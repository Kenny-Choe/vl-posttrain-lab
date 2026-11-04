"""真实小批候选的CPU闭环；不绕过正式12K发布/完整保护准入。"""
from __future__ import annotations
from collections import Counter
from dataclasses import replace
from pathlib import Path
import time

from .curriculum_build import _emit,audit_replay,LOCAL_STATES,SEARCH_STATES
from .grouping import image_fingerprints,build_groups,split_and_sample,phash_candidates
from .release import annotation_identity
from .schema import QARecord
from ..io import contained_path,file_sha256,identity,write_json,write_jsonl


def validate_CPU_review(record:dict,review:dict):
    if review.get('sample_id')!=record['sample_id'] or review.get('annotation_sha256')!=annotation_identity(record):
        raise ValueError('CPU_review_not_bound_to_actual_annotation')
    if (review.get('reviewer_type')!='assistant_visual' or review.get('status')!='accepted_for_CPU_pilot'
        or review.get('scope')!='real_CPU_pilot_not_formal_training_release'):
        raise ValueError('actual_assistant_visual_CPU_review_required')
    if not review.get('visual_evidence_sha256') or not review.get('native_annotation_provenance'):
        raise ValueError('CPU_review_visual_or_native_provenance_missing')
    if not all(review.get(key) is True for key in ['question_unique_within_image','answer_visible',
                                                 'evidence_corresponds_to_asked_field','family_grouped_within_pilot']):
        raise ValueError('CPU_pilot_semantic_review_incomplete')
    if review.get('formal_full_data_release_approved') is not False:
        raise ValueError('CPU_pilot_must_not_forge_formal_data_approval')


def build_real_pilot(records:list[dict],reviews:list[dict],replay:list[dict],image_root:str|Path,
                     output:str|Path,*,counts:dict[str,int],seed:int=42,wall_cap_seconds:int=300):
    begin=time.perf_counter();dest=Path(output);images=Path(image_root)
    if dest.exists():raise ValueError('CPU_pilot_attempt_exists_do_not_overwrite')
    if not records or len(records)>256 or not 4<=len(replay)<=64 or len(replay)%4:
        raise ValueError('CPU_pilot_requires_1_to_256_domain_QA_and_balanced_replay')
    by_review={r['sample_id']:r for r in reviews}
    if len(by_review)!=len(reviews):raise ValueError('duplicate_CPU_review')
    all_ids=[r['sample_id'] for r in [*records,*replay]]
    if len(set(all_ids))!=len(all_ids):raise ValueError('duplicate_domain_or_replay_QA_identity')
    if Counter(r['source'] for r in replay)!=Counter({'GQA':len(replay)//2,'TextVQA':len(replay)//2}):
        raise ValueError('real_replay_requires_actual_GQA_and_TextVQA_half_each')
    index={};checked=[]
    for raw in [*records,*replay]:
        r=QARecord(**raw).validate(require_evidence=raw['source']=='Unitail-OCR')
        if r.provenance_kind!='observed' or r.project_split!='train':raise ValueError('real_CPU_pilot_input_must_be_observed_train')
        if r.source not in {'Unitail-OCR','GQA','TextVQA'}:raise ValueError('pilot_source_not_in_recipe')
        if r.source=='Unitail-OCR':
            if r.source_split!='gallery_train':raise ValueError('Unitail_eval_not_new_train')
            validate_CPU_review(raw,by_review.get(r.sample_id,{}))
        elif r.source_split not in {'train','train_balanced'} or r.native_gold_match!='native_annotation_verified':
            raise ValueError('replay_requires_native_train_question_answer_identity')
        path=contained_path(images,r.image_path)
        if r.image_sha256 not in index:
            fp=image_fingerprints(str(path))
            if fp['file_sha256']!=r.image_sha256 or (fp['width'],fp['height'])!=(r.image_width,r.image_height):
                raise ValueError('real_pilot_image_file_or_dimensions_changed')
            index[r.image_sha256]={'image_id':r.image_sha256,**fp,'product_family_id':r.product_family_id}
        checked.append(r)
    domain_pixels={index[r.image_sha256]['pixel_sha256'] for r in checked[:len(records)]}
    if any(index[r.image_sha256]['pixel_sha256'] in domain_pixels for r in checked[len(records):]):
        raise ValueError('general_replay_contains_domain_image')
    groups=build_groups(list(index.values()),[],set())
    domain=[replace(r,image_id=r.image_sha256,image_group_id=groups['mapping'][r.image_sha256]) for r in checked[:len(records)]]
    # This is a CPU diagnostic cohort allocation, not formal training admission.
    # Keep all quality/admission flags on the returned full records.
    skeleton=[{'sample_id':r.sample_id,'image_group_id':r.image_group_id,
               'source_split':r.source_split,'project_split':r.project_split} for r in domain]
    allocation=split_and_sample(skeleton,counts,seed)
    full_records={r.sample_id:r.to_dict() for r in domain}
    allocation['splits']={name:[{**full_records[r['sample_id']],'project_split':name} for r in items]
                          for name,items in allocation['splits'].items()}
    train=allocation['splits'].get('train',[])
    if len(train)%2 or len(train)<12:raise ValueError('pilot_train_needs_even_count_and_all_search_states')
    dest.mkdir(parents=True)
    write_json(dest/'request.json',{'scope':'real_CPU_pilot_not_formal_training_release','counts':counts,'seed':seed,
         'input_annotation_sha256':identity(records),'review_sha256':identity(reviews),'replay_sha256':identity(replay),
         'wall_cap_seconds':wall_cap_seconds,'model_weights_loaded':False,'training_started':False})
    for name,rows in allocation['splits'].items():
        write_jsonl(dest/f'{name}.private.jsonl',rows)
        write_jsonl(dest/f'{name}.public.jsonl',[QARecord(**r).actor_input() for r in rows])
    write_jsonl(dest/'reviews.private.jsonl',reviews)
    write_jsonl(dest/'image_index.private.jsonl',list(index.values()))
    near=phash_candidates(list(index.values()),max_pairs=5000)
    cross=[]
    for pair in near:
        a=allocation['group_assignment'].get(groups['mapping'][pair['left']])
        b=allocation['group_assignment'].get(groups['mapping'][pair['right']])
        if a and b and a!=b:cross.append(pair)
    write_jsonl(dest/'cross_split_near_review.private.jsonl',cross)
    course=dest/'curriculum';course.mkdir()
    ordered=sorted(train,key=lambda r:identity([seed,'real-pilot-phase',r['sample_id']]))
    n=len(ordered)//2;plan=[]
    plan.extend((QARecord(**r),'localization',LOCAL_STATES[i%2],1) for i,r in enumerate(ordered[:n]))
    plan.extend((QARecord(**r),'search_stop',SEARCH_STATES[i%6],2) for i,r in enumerate(ordered[n:]))
    for source in ['GQA','TextVQA']:
        selected=sorted((r for r in checked[len(records):] if r.source==source),key=lambda r:identity([seed,'replay-stage',r.sample_id]))
        plan.extend((r,'general_replay','general_answer',1 if i<len(selected)//2 else 2) for i,r in enumerate(selected))
    emitted=[];quarantine=[];crops=injected=0
    for r,phase,state,stage in plan:
        if time.perf_counter()-begin>wall_cap_seconds:raise RuntimeError('real_CPU_pilot_wall_cap')
        try:
            row=_emit(r,phase,state,stage,images,course,fixture=False)
            proof=audit_replay(row,course)
            row['admission_scope']='real_CPU_pilot_not_formal_training_release'
            row['semantic_state_review']='teacher_proposal_requires_state_specific_semantic_review'
            emitted.append(row);crops+=proof['crop_calls'];injected+=proof['injected_context_only_actions']
        except (ValueError,KeyError,OSError) as exc:
            quarantine.append({'sample_id':r.sample_id,'state_kind':state,'reason':str(exc)})
    write_jsonl(course/'trajectories.private.jsonl',emitted)
    write_jsonl(course/'quarantine.private.jsonl',quarantine)
    for stage in [1,2]:
        ids=sorted((r['trajectory_id'] for r in emitted if r['training_stage']==stage),key=lambda tid:identity([seed,stage,tid]))
        write_jsonl(course/f'stage_{stage}.ids.jsonl',[{'trajectory_id':tid} for tid in ids])
    manifest={'status':'real_CPU_pilot_geometry_passed' if not quarantine else 'partial',
        'scope':'real_CPU_pilot_not_formal_training_release','provenance_kind':'gold_guided_teacher_proposals',
        'trajectories':len(emitted),'counts':dict(Counter(r['curriculum_phase'] for r in emitted)),
        'stage_counts':dict(Counter(str(r['training_stage']) for r in emitted)),
        'state_counts':dict(Counter(r['state_kind'] for r in emitted)),
        'actual_crop_replays':crops,'injected_context_only_actions':injected,'quarantined':len(quarantine),
        'real_source_images':True,'fixture':False,'geometric_replay_passed':not quarantine,
        'native_processor_validation':'pending','native_training_mask_validation':'pending',
        'ready_for_training':False,'real_9k_curriculum_completed':False,
        'full_upstream_family_and_external_protection_complete':False,'teacher_state_semantics_fully_reviewed':False,
        'model_weights_loaded':False,'model_generation_calls':0,'optimizer_steps':0,
        'file_sha256':{p.relative_to(course).as_posix():file_sha256(p) for p in course.rglob('*') if p.is_file()}}
    write_json(course/'manifest.json',manifest)
    result={'status':'CPU_pilot_complete_not_formal_data_release' if not quarantine and not cross else 'partial',
        'domain_QA':len(records),'domain_images':len({r.image_sha256 for r in checked[:len(records)]}),
        'domain_splits':{k:len(v) for k,v in allocation['splits'].items()},
        'family_groups_disjoint_within_reviewed_cohort':True,'general_replay_QA':len(replay),
        'known_cross_split_near_candidates':len(cross),'curriculum':manifest,
        'formal_12K_data_release_ready':False,'real_9k_curriculum_completed':False,
        'ready_for_training':False,'CPU_wall_seconds':time.perf_counter()-begin,
        'model_weights_loaded':False,'training_started':False}
    write_json(dest/'receipt.json',result)
    return result
