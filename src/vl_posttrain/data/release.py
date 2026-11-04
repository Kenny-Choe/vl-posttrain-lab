"""审阅记录→真实图片索引→保护/家族隔离→精确配额与可恢复发布。"""
from __future__ import annotations

from collections import Counter,defaultdict
from dataclasses import replace
from pathlib import Path
import time
import unicodedata
import re
from .schema import QARecord
from .grouping import image_fingerprints,phash_candidates,build_groups,split_and_sample,CandidateBudgetExceeded
from ..io import identity,contained_path,read_json,file_sha256,write_json,write_jsonl

REVIEW_FIELDS=('source','source_revision','native_qa_id','source_split','image_sha256','image_width','image_height',
               'question','answers','task_type','product_family_id','evidence_box_xyxy','annotation_version','evidence_source')
EVAL_SPLITS={'test','val','validation','testmini','matching_test','detection_test','recognition_test'}


def annotation_identity(record:dict) -> str:
    """坐标/答案/原图任一改变都使审阅失效；路径和派生split不改变语义身份。"""
    return identity({key:record.get(key) for key in REVIEW_FIELDS})


def image_uid(file_sha256:str) -> str:
    return identity(['image-file-v1',file_sha256])


def review_errors(record:QARecord,review:dict|None,scope:str) -> list[str]:
    if not review:return ['missing_review']
    reasons=[]
    if review.get('status')!='accepted':reasons.append('review_not_accepted')
    if review.get('annotation_sha256')!=annotation_identity(record.to_dict()):reasons.append('review_annotation_identity_mismatch')
    if not review.get('review_id') or not review.get('reviewer_id'):reasons.append('review_identity_missing')
    required=('question_unique','answer_visible','evidence_tight','family_confirmed')
    if any(review.get(key) is not True for key in required):reasons.append('semantic_or_family_review_incomplete')
    if not record.product_family_id or not review.get('family_review_id'):reasons.append('family_identity_missing')
    if record.annotation_version in {'','unreviewed'} or record.evidence_source in {'','none'}:reasons.append('annotation_provenance_missing')
    if record.quality_flags:reasons.append('unresolved_quality_flags')
    if scope=='synthetic_fixture':
        if record.provenance_kind!='synthetic_fixture' or review.get('reviewer_type')!='synthetic_fixture':
            reasons.append('fixture_scope_mismatch')
    else:
        if record.provenance_kind!='observed' or review.get('reviewer_type') not in {'human','upstream_native'}:
            reasons.append('observed_scope_requires_real_review_provenance')
    if scope=='packaging' and record.source not in {'Unitail-OCR','RPC'}:reasons.append('source_not_in_packaging_recipe')
    if scope=='packaging' and record.source_split.lower()!={'Unitail-OCR':'gallery_train','RPC':'exemplar_train'}.get(record.source):
        reasons.append('source_component_not_in_packaging_train_recipe')
    return reasons


def prepare_release(records:list[dict],reviews:list[dict],image_root:str|Path,counts:dict[str,int],*,
                    scope:str='packaging',protected_images:list[dict]|None=None,reviewed_edges:list[dict]|None=None,
                    seed:int=42,max_qa_per_image:int=4,max_image_pixels:int=40_000_000,max_candidate_pairs:int=50_000) -> dict:
    from PIL import Image
    begin=time.perf_counter()
    if scope not in {'packaging','legacy_regression','synthetic_fixture'}:raise ValueError('未知发布范围')
    if not records or len(records)>100000:raise ValueError('输入QA数量需在1到100000之间')
    if max_qa_per_image<1 or max_candidate_pairs<1:raise ValueError('非法资源/样本上限')
    protected_images=protected_images or [];reviewed_edges=reviewed_edges or []
    review_index={r['sample_id']:r for r in reviews}
    if len(review_index)!=len(reviews):raise ValueError('审阅记录的sample_id重复')
    ids=[r.get('sample_id') for r in records]
    if any(not isinstance(uid,str) or not uid for uid in ids) or len(set(ids))!=len(ids):
        raise ValueError('QA身份缺失或重复')
    config={'scope':scope,'counts':counts,'seed':seed,'max_qa_per_image':max_qa_per_image,
            'max_image_pixels':max_image_pixels,'max_candidate_pairs':max_candidate_pairs,'version':'reviewed-data-release-v1'}
    request_sha=identity([records,reviews,protected_images,reviewed_edges,config])
    index={};cache={};accepted=[];quarantine=[];unused=[];families=defaultdict(set);protected=set();aliases=[]
    def reject(sample_id,reasons):quarantine.append({'sample_id':sample_id,'reasons':reasons})
    for raw in records:
        sid=raw['sample_id']
        try:
            record=QARecord(**raw)
            # 上游eval图也要参与指纹/保护，不能因其不准训练而漏掉图片。
            replace(record,project_split='diagnostic').validate(require_evidence=False)
            path=contained_path(image_root,record.image_path)
            key=(str(path),record.image_sha256)
            if key not in cache:
                with Image.open(path) as im:
                    if im.width*im.height>max_image_pixels:raise ValueError('image_pixel_resource_cap_exceeded')
                cache[key]=image_fingerprints(str(path))
            fingerprint=cache[key]
            if fingerprint['file_sha256']!=record.image_sha256:raise ValueError('image_file_identity_mismatch')
            if (fingerprint['width'],fingerprint['height'])!=(record.image_width,record.image_height):
                raise ValueError('natural_image_dimensions_mismatch')
            uid=image_uid(record.image_sha256)
            index[uid]={'image_id':uid,**fingerprint}
            aliases.append({'sample_id':sid,'provided_image_id':record.image_id,'derived_image_id':uid})
            split_name=record.source_split.lower()
            if split_name in EVAL_SPLITS or re.search(r'(?:^|[_/.-])(test|testing|val|validation|testmini)(?:$|[_/.-])',split_name):
                protected.add(uid);reject(sid,['upstream_evaluation_protected']);continue
            if record.project_split!='train':reject(sid,['input_is_not_new_train_candidate']);continue
            record.validate(require_evidence=True)
            problems=review_errors(record,review_index.get(sid),scope)
            if problems:reject(sid,problems);continue
            families[record.product_family_id].add(uid)
            accepted.append(replace(record,image_id=uid))
        except (ValueError,TypeError,KeyError,OSError) as exc:
            reject(sid,['input_or_image_check_failed:'+str(exc)])
    unknown_protection_versions=[]
    for image in protected_images:
        uid=image['image_id']
        if uid in index and (index[uid]['file_sha256']!=image.get('file_sha256') or image.get('version')=='exif_rgb_white_dct32_v1' and index[uid]['pixel_sha256']!=image.get('pixel_sha256')):
            raise ValueError('保护索引的图像身份冲突')
        if not isinstance(image.get('file_sha256'),str) or not re.fullmatch('[0-9a-f]{64}',image['file_sha256']):raise ValueError('保护图片文件SHA格式错误')
        row=dict(index[uid]) if uid in index else dict(image)
        if uid not in index and image.get('version')!='exif_rgb_white_dct32_v1':
            unknown_protection_versions.append(uid);row['pixel_sha256']=None;row.pop('phash64',None)
        elif uid not in index and (not isinstance(image.get('pixel_sha256'),str) or not re.fullmatch('[0-9a-f]{64}',image['pixel_sha256'])):
            raise ValueError('保护图片像素SHA格式错误')
        index[uid]=row;protected.add(uid)
        if image.get('product_family_id'):families[image['product_family_id']].add(uid)
    family_edges=[]
    for family,members in sorted(families.items()):
        ordered=sorted(members)
        family_edges.extend({'left':ordered[0],'right':uid,'confirmed':True,'basis':'human_reviewed','family_id':family} for uid in ordered[1:])
    all_edges=family_edges+reviewed_edges
    groups=build_groups(list(index.values()),all_edges,protected)
    blocked=set(groups['quarantined_groups']);known_protected_groups=set(blocked)
    pair_reviews={}
    distinct_bases={'human_reviewed_distinct'}|({'fixture_render_provenance_distinct'} if scope=='synthetic_fixture' else set())
    for edge in reviewed_edges:
        if edge['left'] not in index or edge['right'] not in index:raise ValueError('关系引用不在本次图片索引中')
        key=tuple(sorted([edge['left'],edge['right']]))
        if edge.get('reviewed') is True and edge.get('confirmed') is False and edge.get('basis') in distinct_bases:
            pair_reviews[key]='distinct'
    missing_phash=[uid for uid,img in index.items() if not img.get('phash64')]
    scan_complete=not missing_phash and not unknown_protection_versions
    try:near=phash_candidates([r for r in index.values() if r.get('phash64')],max_pairs=max_candidate_pairs)
    except CandidateBudgetExceeded as exc:near=exc.pairs;scan_complete=False
    unresolved=[]
    for pair in near:
        left,right=pair['left'],pair['right'];a,b=groups['mapping'][left],groups['mapping'][right]
        if a==b or pair_reviews.get(tuple(sorted([left,right])))=='distinct':continue
        unresolved.append(pair)
        # 不确定的保护近邻只隔离候选，不把它签成已证泄漏。
        if a in known_protected_groups:blocked.add(b)
        if b in known_protected_groups:blocked.add(a)
    good=[]
    for record in accepted:
        group=groups['mapping'][record.image_id]
        if group in blocked:reject(record.sample_id,['protected_or_unresolved_protection_neighbor']);continue
        good.append(replace(record,image_group_id=group))
    exact=defaultdict(list)
    for record in good:
        question=' '.join(unicodedata.normalize('NFKC',record.question).casefold().split())
        exact[(index[record.image_id]['pixel_sha256'],question)].append(record)
    dedup=[]
    for values in exact.values():
        signatures={identity([r.answers,r.evidence_box_xyxy,r.task_type]) for r in values}
        if len(signatures)>1:
            for r in values:reject(r.sample_id,['conflicting_duplicate_annotations_needs_adjudication'])
            continue
        ordered=sorted(values,key=lambda r:r.sample_id);dedup.append(ordered[0])
        unused.extend({'sample_id':r.sample_id,'reason':'duplicate_QA_same_canonical_pixels_and_question'} for r in ordered[1:])
    per_image=defaultdict(list)
    for r in dedup:per_image[index[r.image_id]['pixel_sha256']].append(r)
    capped=[]
    for values in per_image.values():
        ordered=sorted(values,key=lambda r:identity([seed,'per-image-cap',r.sample_id]));capped.extend(ordered[:max_qa_per_image])
        unused.extend({'sample_id':r.sample_id,'reason':'per_image_QA_cap_not_label_error'} for r in ordered[max_qa_per_image:])
    split=None;assignment_error=None
    try:split=split_and_sample([r.to_dict() for r in capped],counts,seed)
    except ValueError as exc:assignment_error=str(exc)
    splits=split['splits'] if split else {}
    assignment=split['group_assignment'] if split else {}
    cross_split=[]
    for pair in unresolved:
        a=assignment.get(groups['mapping'][pair['left']]);b=assignment.get(groups['mapping'][pair['right']])
        if a and b and a!=b:cross_split.append(pair)
    external_protection_supplied=bool(protected_images)
    counts_met=split is not None
    engineering_ready=counts_met and scan_complete and not cross_split
    real_ready=engineering_ready and scope!='synthetic_fixture' and external_protection_supplied
    target_counts=scope=='packaging' and counts=={'train':8000,'dev':2000,'test':2000}
    status='ready' if real_ready else 'fixture_engineering_passed' if engineering_ready and scope=='synthetic_fixture' else 'partial_needs_review'
    membership={s:[r['sample_id'] for r in rows] for s,rows in splits.items()}
    selected={sid for values in membership.values() for sid in values}
    unused.extend({'sample_id':r.sample_id,'reason':'eligible_not_selected_by_frozen_quota'} for r in capped if r.sample_id not in selected)
    manifest={'status':status,'scope':scope,'request_sha256':request_sha,'config':config,
              'input_QA':len(records),'indexed_images':len(index),'review_accepted_before_grouping':len(accepted),
              'eligible_after_dedup_and_cap':len(capped),'quarantined_QA':len(quarantine),'unused_QA':len(unused),
              'requested_counts':counts,'actual_counts':{k:len(v) for k,v in splits.items()},
              'source_counts':dict(Counter(r['source'] for rows in splits.values() for r in rows)),
              'counts_met':counts_met,'assignment_error':assignment_error,'phash_scan_complete':scan_complete,
              'protected_index_supplied':external_protection_supplied,'protected_images':len(protected),
              'known_cross_split_phash_candidates_pending':len(cross_split),'all_unresolved_phash_candidates':len(unresolved),
              'missing_phash_images':missing_phash,'family_and_image_groups_disjoint':True if split else None,
              'unknown_protection_fingerprint_versions':unknown_protection_versions,
              'reviewed_real_data_release_passed':real_ready,'target_12k_dataset_counts_met':real_ready and target_counts,
              'target_scientific_result_proven':False,'model_weights_loaded':False,'optimizer_steps':0,
              'generation_calls':0,'CPU_wall_seconds':time.perf_counter()-begin,'membership':membership,
              'split_algorithm':split['algorithm'] if split else None}
    manifest['release_sha256']=identity([config,sorted(index.values(),key=lambda r:r['image_id']),membership,quarantine,unused,reviewed_edges])
    return {'manifest':manifest,'splits':splits,'quarantine':quarantine,'unused':unused,'image_index':list(index.values()),
            'phash_review_queue':unresolved,'cross_split_review_queue':cross_split,'image_aliases':aliases,
            'eligible_candidates':[r.to_dict() for r in capped]}


def write_release(output:str|Path,result:dict) -> dict:
    dest=Path(output);manifest=result['manifest']
    existing=dest/'manifest.json'
    if existing.exists():
        old=read_json(existing)
        if old['release_sha256']!=manifest['release_sha256'] or old['request_sha256']!=manifest['request_sha256']:
            raise ValueError('已有发布目录身份不同；使用新attempt目录，不能覆盖旧发布')
        for name,digest in old['file_sha256'].items():
            if file_sha256(dest/name)!=digest:raise ValueError('已有发布文件内容改变，禁止静默复用')
        return old
    dest.mkdir(parents=True,exist_ok=True)
    lock=dest/'release.lock'
    with lock.open('x',encoding='utf-8') as f:f.write(manifest['request_sha256'])
    files=[]
    for split,rows in result['splits'].items():
        private=f'{split}.private.jsonl';public=f'{split}.public.jsonl'
        write_jsonl(dest/private,rows);write_jsonl(dest/public,[QARecord(**r).actor_input() for r in rows]);files.extend([private,public])
    for key in ('quarantine','unused','image_index','phash_review_queue','cross_split_review_queue','image_aliases','eligible_candidates'):
        name=key+'.private.jsonl';write_jsonl(dest/name,result[key]);files.append(name)
    frozen={**manifest,'file_sha256':{name:file_sha256(dest/name) for name in sorted(files)}}
    write_json(existing,frozen);lock.unlink()
    return frozen
