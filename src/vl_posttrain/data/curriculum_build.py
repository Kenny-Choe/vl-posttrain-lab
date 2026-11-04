"""训练侧QA→金标引导的课程候选。真实像素回放，不执行模型或训练。"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
from pathlib import Path
import time
import unicodedata

from ..agent.geometry import ViewRegistry, area, iou
from ..agent.protocol import PROTOCOL_VERSION, SYSTEM_PROMPT, parse_action
from ..config import Budgets
from ..io import canonical, contained_path, file_sha256, identity, read_json, read_jsonl, write_json, write_jsonl
from .curriculum import final_target
from .grouping import image_fingerprints,phash_candidates,CandidateBudgetExceeded
from .schema import QARecord

LOCAL_STATES=('root_final','crop_local_final')
SEARCH_STATES=('direct_final','coarse_to_fine','wrong_then_correct','earlier_view_final',
               'repeat_then_stop','wrong_repeat_then_correct')
RESERVED=('<|im_start|>','<|im_end|>','<|vision_start|>','<|vision_end|>','<|image_pad|>')


def pixel_sha(image) -> str:
    return hashlib.sha256(str(image.size).encode()+b'|RGB|'+image.tobytes()).hexdigest()


def load_rgb(path:Path,max_pixels:int=40_000_000):
    from PIL import Image,ImageOps
    with Image.open(path) as raw:
        if raw.width*raw.height>max_pixels:raise ValueError('image_pixel_cap_exceeded')
        rgba=ImageOps.exif_transpose(raw).convert('RGBA')
        white=Image.new('RGBA',rgba.size,'white');white.alpha_composite(rgba)
        return white.convert('RGB')


def expanded(rect,width,height,factor):
    x0,y0,x1,y1=rect;cx,cy=(x0+x1)/2,(y0+y1)/2
    w,h=max(32,(x1-x0)*factor),max(32,(y1-y0)*factor)
    # This plans a teacher search window, never repairs a policy prediction.
    return [max(0,cx-w/2),max(0,cy-h/2),min(width,cx+w/2),min(height,cy+h/2)]


def wrong_region(rect,width,height):
    x0,y0,x1,y1=rect
    candidates=[[0,0,x0,height],[x1,0,width,height],[0,0,width,y0],[0,y1,width,height]]
    candidates=[r for r in candidates if r[2]-r[0]>=8 and r[3]-r[1]>=8]
    if not candidates:raise ValueError('no_nonoverlapping_wrong_region_do_not_fabricate_state')
    return max(candidates,key=lambda r:(area(r),r))


def verify_release(directory:Path):
    manifest=read_json(directory/'manifest.json')
    fixture=manifest['scope']=='synthetic_fixture'
    if manifest['status']!=('fixture_engineering_passed' if fixture else 'ready'):
        raise ValueError('领域输入必须是已完成的QA发布，不采用partial候选')
    for name,digest in manifest['file_sha256'].items():
        if file_sha256(contained_path(directory,name))!=digest:
            raise ValueError('QA发布文件身份改变：'+name)
    rows=read_jsonl(directory/'train.private.jsonl')
    if any(row['project_split']!='train' for row in rows):raise ValueError('课程不得读取dev/test作为训练')
    return manifest,rows,read_jsonl(directory/'image_index.private.jsonl'),fixture


def _emit(record:QARecord,phase:str,state_kind:str,stage:int,image_root:Path,dest:Path,*,fixture:bool):
    source=contained_path(image_root,record.image_path)
    if file_sha256(source)!=record.image_sha256:raise ValueError('original_file_sha_mismatch')
    image=load_rgb(source)
    if image.size!=(record.image_width,record.image_height):raise ValueError('natural_dimensions_mismatch')
    root_pixel=pixel_sha(image);root_file=dest/'roots'/f'{root_pixel}.png'
    root_file.parent.mkdir(parents=True,exist_ok=True)
    if not root_file.exists():image.save(root_file)
    trajectory_id=identity(['teacher-curriculum-v1',record.sample_id,phase,state_kind])
    root_ref={'path':root_file.relative_to(dest).as_posix(),'file_sha256':file_sha256(root_file),
              'pixel_sha256':root_pixel,'view_id':'v0'}
    registry=ViewRegistry(record.image_id,*image.size)
    general=phase=='general_replay'
    prompt='你是图文问答助手。直接回答图中问题，保持答案简洁。' if general else SYSTEM_PROMPT
    messages=[{'role':'system','text':prompt,'images':[],'supervised':False},
              {'role':'user','text':record.question if general else f'view_id=v0，自然尺寸={image.size}。{record.question}',
               'images':[root_ref],'supervised':False}]
    actions=[];views={'v0':{'root_box':list(registry.get('v0').root_box),**root_ref}}
    directory=dest/'views'/trajectory_id;directory.mkdir(parents=True,exist_ok=True)
    gt=record.evidence_box_xyxy

    def crop(rect,parent='v0',injected=False):
        target=registry.get(parent).from_root(rect)
        text=canonical({'tool':'crop','view_id':parent,'box':target})
        parse_action(text)
        view=registry.crop(parent,target)
        cropped=image.crop(view.root_box);path=directory/f'{view.view_id}.png';cropped.save(path)
        ref={'path':path.relative_to(dest).as_posix(),'file_sha256':file_sha256(path),
             'pixel_sha256':pixel_sha(cropped),'view_id':view.view_id}
        public={'view_id':view.view_id,'parent_id':parent,'natural_size':list(cropped.size),'root_box':list(view.root_box)}
        messages.extend([{'role':'assistant','text':text,'images':[],'supervised':not injected},
                         {'role':'tool','text':canonical(public),'images':[ref],'supervised':False}])
        actions.append({'kind':'crop','text':text,'supervised':not injected,'injected_wrong_action':injected,
                        'new_view_id':view.view_id,'crop_pixels_sha256':ref['pixel_sha256'],
                        'processor_visual_tokens':None,'processor_status':'not_yet_processed',**public})
        views[view.view_id]={**public,**ref}
        return view.view_id

    final_view='v0'
    if state_kind in {'crop_local_final','wrong_then_correct','earlier_view_final','repeat_then_stop','wrong_repeat_then_correct'}:
        if state_kind in {'wrong_then_correct','wrong_repeat_then_correct'}:
            bad=wrong_region(gt,*image.size);crop(bad,injected=True)
            if state_kind=='wrong_repeat_then_correct':crop(bad,injected=True)
        final_view=crop(expanded(gt,*image.size,2.4))
        if state_kind=='earlier_view_final':crop(wrong_region(gt,*image.size),injected=True)
        if state_kind=='repeat_then_stop':crop(registry.get(final_view).root_box,injected=True)
    elif state_kind=='coarse_to_fine':
        first=crop(expanded(gt,*image.size,4.0))
        final_view=crop(expanded(gt,*image.size,1.6),parent=first)
    elif state_kind not in {'root_final','direct_final','general_answer'}:
        raise ValueError('未知课程状态')
    final=record.answers[0] if general else final_target(record.to_dict(),registry,final_view)
    final_root=None if general else list(registry.get(final_view).to_root(parse_action(final)['final']['evidence']['box']))
    if not general and iou(final_root,gt)<.5:raise ValueError('teacher_target_quantization_failed_IoU_gate')
    messages.append({'role':'assistant','text':final,'images':[],'supervised':True})
    actions.append({'kind':'final','text':final,'supervised':True})
    return {'trajectory_id':trajectory_id,'sample_id':record.sample_id,'image_group_id':record.image_group_id,
            'product_family_id':record.product_family_id,'source':record.source,'project_split':'train',
            'curriculum_phase':phase,'training_stage':stage,'state_kind':state_kind,
            'provenance_kind':'synthetic_fixture' if fixture else 'gold_guided_teacher_proposal',
            'teacher_actions_not_model_rollouts':True,'qa_sha256':identity(record.to_dict()),
            'messages':messages,'actions':actions,'views':views,'protocol_version':PROTOCOL_VERSION,
            'private_supervision':{'answers':record.answers,'evidence_box_xyxy':gt},
            'actual_teacher_final_root':final_root,'native_processor_validation':'pending',
            'native_training_mask_validation':'pending','semantic_state_review':'programmatic_fixture' if fixture else 'pending',
            'model_weights_loaded':False,'model_generation_calls':0,'optimizer_steps':0}


def audit_replay(row:dict,root:Path) -> dict:
    """从保存的规范原图独立执行动作，核对像素/视图/监督归属。"""
    first=row['messages'][1]['images'][0]
    image=load_rgb(contained_path(root,first['path']))
    if pixel_sha(image)!=first['pixel_sha256']:raise ValueError('root_pixels_changed')
    registry=ViewRegistry('replay-root',*image.size);crop_count=0;injected=0;final_seen=False
    assistant=[m for m in row['messages'] if m['role']=='assistant']
    if len(assistant)!=len(row['actions']):raise ValueError('action_message_count_mismatch')
    for index,action in enumerate(row['actions']):
        if final_seen:raise ValueError('action_after_final')
        if action['text']!=assistant[index]['text'] or action['supervised'] is not assistant[index]['supervised']:
            raise ValueError('action_message_supervision_mismatch')
        if action['kind']=='crop':
            parsed=parse_action(action['text']);view=registry.crop(parsed['view_id'],parsed['box'])
            crop_count+=1
            if action.get('injected_wrong_action'):
                injected+=1
                if action['supervised'] is not False:raise ValueError('injected_wrong_action_is_supervised')
            elif action['supervised'] is not True:raise ValueError('teacher_correct_action_not_supervised')
            ref=row['views'][view.view_id]
            if list(view.root_box)!=ref['root_box'] or action['new_view_id']!=view.view_id:raise ValueError('view_geometry_mismatch')
            pixels=image.crop(view.root_box)
            path=contained_path(root,ref['path'])
            if file_sha256(path)!=ref['file_sha256'] or pixel_sha(load_rgb(path))!=pixel_sha(pixels):
                raise ValueError('saved_crop_not_actual_root_pixels')
            if action['crop_pixels_sha256']!=pixel_sha(pixels):raise ValueError('crop_receipt_pixel_mismatch')
        elif action['kind']=='final':
            final_seen=True
            if action['supervised'] is not True:raise ValueError('final_is_not_supervised')
            if row['curriculum_phase']!='general_replay':
                parsed=parse_action(action['text']);value=parsed['final']
                actual=list(registry.get(value['evidence']['view_id']).to_root(value['evidence']['box']))
                if actual!=row['actual_teacher_final_root']:raise ValueError('final_geometry_mismatch')
                if value['answer'] not in row['private_supervision']['answers']:raise ValueError('teacher_answer_mismatch')
            elif action['text'] not in row['private_supervision']['answers']:raise ValueError('general_answer_mismatch')
        else:raise ValueError('unknown_teacher_action')
    if not final_seen:raise ValueError('teacher_trajectory_has_no_final')
    cap=1 if row['curriculum_phase']=='localization' else 3
    if crop_count>cap:raise ValueError('curriculum_crop_cap_exceeded')
    for message in row['messages']:
        if message['role']!='assistant' and message['supervised'] is not False:raise ValueError('non_assistant_supervised')
        for ref in message['images']:
            if ref['view_id'] not in row['views']:raise ValueError('unknown_image_view')
    return {'crop_calls':crop_count,'injected_context_only_actions':injected,'pixel_replay_passed':True}


def build_curriculum(domain_release:str|Path,domain_images:str|Path,replay_rows:list[dict],replay_images:str|Path,
                     output:str|Path,*,localization:int=4000,search_stop:int=4000,general_replay:int=1000,
                     seed:int=42,wall_cap_seconds:int=900,disk_cap_bytes:int=512*1024**2) -> dict:
    begin=time.perf_counter();directory=Path(domain_release);dest=Path(output)
    release,domain,index,fixture=verify_release(directory)
    if any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in [localization,search_stop,general_replay,wall_cap_seconds,disk_cap_bytes]) or general_replay%4:
        raise ValueError('课程数量必须为正；回放数须能按两源两阶段等分')
    if len(domain)!=localization+search_stop:raise ValueError('领域train必须恰好分为两阶段；不静默舍弃或复制')
    if len(replay_rows)!=general_replay:raise ValueError('回放输入须为冻结的准确配额')
    if any(r['provenance_kind']!=('synthetic_fixture' if fixture else 'observed') for r in [*domain,*replay_rows]):
        raise ValueError('真实/fixture来源混用')
    all_ids=[r['sample_id'] for r in [*domain,*replay_rows]]
    if len(set(all_ids))!=len(all_ids):raise ValueError('领域/回放重复QA身份')
    sources={'GQA':'synthetic_GQA_fixture','TextVQA':'synthetic_TextVQA_fixture'} if fixture else {'GQA':'GQA','TextVQA':'TextVQA'}
    if Counter(r['source'] for r in replay_rows)!=Counter({v:general_replay//2 for v in sources.values()}):
        raise ValueError('回放来源必须按GQA/TextVQA各半，不能伪造或替换来源')
    forbidden_files={r['file_sha256'] for r in index};forbidden_pixels={r.get('pixel_sha256') for r in index}
    seen_files={};validated_replay=[];replay_signatures=set()
    for raw in replay_rows:
        record=QARecord(**raw).validate()
        if record.project_split!='train':raise ValueError('通用回放仅可来自train')
        if record.source_split.lower() not in {'train','train_balanced','balanced_train'}:raise ValueError('通用回放上游组件必须明确为训练集')
        if not fixture and record.native_gold_match not in {'question_choices_image_checked','native_annotation_verified','human_reviewed_answer'}:
            raise ValueError('通用回放缺可追踪金答案核验')
        path=contained_path(replay_images,record.image_path)
        if record.image_sha256 not in seen_files:seen_files[record.image_sha256]=image_fingerprints(str(path))
        fingerprint=seen_files[record.image_sha256]
        if fingerprint['file_sha256']!=record.image_sha256:raise ValueError('replay_image_identity_mismatch')
        if fingerprint['file_sha256'] in forbidden_files or fingerprint['pixel_sha256'] in forbidden_pixels:
            raise ValueError('通用回放与领域/保护索引存在同图，禁止额外暴露')
        signature=(fingerprint['pixel_sha256'],' '.join(unicodedata.normalize('NFKC',record.question).casefold().split()))
        if signature in replay_signatures:raise ValueError('通用回放存在同图同题重复，不能只换ID凑数')
        replay_signatures.add(signature)
        validated_replay.append(record)
    replay_index=[{'image_id':'replay-'+digest,**fp} for digest,fp in seen_files.items()]
    replay_ids={r['image_id'] for r in replay_index};near_complete=True
    try:near=phash_candidates([*index,*replay_index],max_pairs=50000)
    except CandidateBudgetExceeded as exc:near=exc.pairs;near_complete=False
    replay_near=[p for p in near if (p['left'] in replay_ids)!=(p['right'] in replay_ids)]
    domain=sorted(domain,key=lambda r:identity([seed,'domain-phase',r['sample_id']]))
    plan=[(QARecord(**r).validate(require_evidence=True),'localization',LOCAL_STATES[i%len(LOCAL_STATES)],1,Path(domain_images)) for i,r in enumerate(domain[:localization])]
    plan.extend((QARecord(**r).validate(require_evidence=True),'search_stop',SEARCH_STATES[i%len(SEARCH_STATES)],2,Path(domain_images)) for i,r in enumerate(domain[localization:]))
    for source in sources.values():
        rows=sorted((r for r in validated_replay if r.source==source),key=lambda r:identity([seed,'general-stage',r.sample_id]))
        plan.extend((r,'general_replay','general_answer',1 if i<general_replay//4 else 2,Path(replay_images)) for i,r in enumerate(rows))
    if dest.exists():raise ValueError('课程输出已存在，保留原attempt，使用新目录')
    dest.mkdir(parents=True);(dest/'curriculum.lock').write_text(identity([release['release_sha256'],all_ids,seed]),encoding='utf-8')
    counts=Counter();states=Counter();stages={1:[],2:[]};quarantine=[];records=[];injected=0;calls=0;json_bytes=0
    launch={'scope':'synthetic_fixture' if fixture else 'gold_guided_teacher_proposals','domain_release_sha256':release['release_sha256'],
            'counts':{'localization':localization,'search_stop':search_stop,'general_replay':general_replay},'seed':seed,
            'wall_cap_seconds':wall_cap_seconds,'disk_cap_bytes':disk_cap_bytes,'protocol':PROTOCOL_VERSION,'budgets':asdict(Budgets()),
            'teacher_planning_not_model_generation':True}
    write_json(dest/'launch_manifest.json',launch)
    try:
        for number,(record,phase,state,stage,images) in enumerate(plan):
            if any(marker in text for text in [record.question,*record.answers] for marker in RESERVED):
                quarantine.append({'sample_id':record.sample_id,'reason':'reserved_chat_marker_in_source_text'});continue
            try:
                row=_emit(record,phase,state,stage,images,dest,fixture=fixture)
                proof=audit_replay(row,dest)
                injected+=proof['injected_context_only_actions'];calls+=proof['crop_calls']
                records.append(row);counts[phase]+=1;states[state]+=1;stages[stage].append(row['trajectory_id'])
                json_bytes+=len(canonical(row).encode())+1
            except (ValueError,OSError,KeyError) as exc:
                quarantine.append({'sample_id':record.sample_id,'phase':phase,'state':state,'reason':str(exc)})
            if number%64==0:
                elapsed=time.perf_counter()-begin
                size=sum(p.stat().st_size for p in dest.rglob('*') if p.is_file())
                if elapsed>wall_cap_seconds or size+json_bytes>disk_cap_bytes:
                    raise RuntimeError('curriculum_CPU_wall_or_disk_cap')
        write_jsonl(dest/'trajectories.private.jsonl',records)
        write_jsonl(dest/'quarantine.private.jsonl',quarantine)
        write_jsonl(dest/'replay_near_protection_candidates.private.jsonl',replay_near)
        for stage,ids in stages.items():
            ordered=sorted(ids,key=lambda tid:identity([seed,stage,'stage-order',tid]))
            write_jsonl(dest/f'stage_{stage}.ids.jsonl',[{'trajectory_id':tid} for tid in ordered])
        expected=launch['counts'];geometry_passed=dict(counts)==expected and not quarantine
        result={'status':'fixture_geometry_passed' if fixture and geometry_passed else 'teacher_proposals_geometry_passed' if geometry_passed else 'partial',
                'provenance_kind':'synthetic_fixture' if fixture else 'gold_guided_teacher_proposals',
                'counts':dict(counts),'expected_counts':expected,'stage_counts':{str(k):len(v) for k,v in stages.items()},
                'state_counts':dict(states),'trajectories':len(records),'actual_crop_replays':calls,
                'injected_context_only_actions':injected,'quarantined':len(quarantine),
                'geometric_replay_passed':geometry_passed,'native_processor_validation':'pending','native_training_mask_validation':'pending',
                'replay_protection_phash_scan_complete':near_complete,'replay_protection_near_candidates':len(replay_near),
                'semantic_state_review':'fixture_only' if fixture else 'pending','ready_for_training':False,
                'real_9k_curriculum_completed':False,'model_weights_loaded':False,'model_generation_calls':0,'optimizer_steps':0,
                'CPU_wall_seconds':time.perf_counter()-begin,'launch_manifest_sha256':file_sha256(dest/'launch_manifest.json')}
        result['file_sha256']={p.relative_to(dest).as_posix():file_sha256(p) for p in dest.rglob('*') if p.is_file() and p.name!='curriculum.lock'}
        write_json(dest/'manifest.json',result);(dest/'curriculum.lock').unlink()
        return result
    except Exception as exc:
        write_json(dest/'failure.json',{'error_type':type(exc).__name__,'error':str(exc),'completed_rows':len(records),
                                     'model_weights_loaded':False,'ready_for_training':False,'CPU_wall_seconds':time.perf_counter()-begin})
        raise
