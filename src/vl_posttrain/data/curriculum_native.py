"""CPU原生processor展开与两种独立span核验；无模型权重、无训练。"""
from __future__ import annotations
from collections import defaultdict
import copy
import importlib.metadata
from pathlib import Path
import time

from ..agent.backend import resize_for_budget
from ..config import Budgets
from ..io import contained_path,file_sha256,identity,read_json,read_jsonl,write_json,write_jsonl
from .curriculum_build import audit_replay,load_rgb,RESERVED


def assistant_spans(ids:list[int],header:list[int],end_id:int,newline:list[int],flags:list[bool]):
    """结构oracle：角色头不监督，body和真实im_end后缀换行属于动作目标。"""
    starts=[i for i in range(len(ids)-len(header)+1) if ids[i:i+len(header)]==header]
    if len(starts)!=len(flags):raise ValueError('native_assistant_header_count_mismatch')
    spans=[];mask=[False]*len(ids)
    for start,supervised in zip(starts,flags):
        if type(supervised) is not bool:raise ValueError('监督标记必须为真实布尔值')
        body=start+len(header)
        try:end=ids.index(end_id,body)
        except ValueError as exc:raise ValueError('assistant_end_token_missing') from exc
        stop=end+1+len(newline)
        if ids[end+1:stop]!=newline:raise ValueError('native_end_newline_suffix_changed')
        spans.append({'header_start':start,'body_start':body,'end_token':end,'stop':stop,'supervised':supervised})
        if supervised:mask[body:stop]=[True]*(stop-body)
    return spans,mask


class ProcessorOnly:
    def __init__(self,model_metadata:str|Path):
        from transformers import AutoProcessor
        path=Path(model_metadata)
        self.processor=AutoProcessor.from_pretrained(path,local_files_only=True,trust_remote_code=False,use_fast=False)
        self.budgets=Budgets()
        names=['config.json','tokenizer.json','tokenizer_config.json','preprocessor_config.json','chat_template.json']
        self.identity={'class':type(self.processor).__name__,
                       'metadata_sha256':{name:file_sha256(path/name) for name in names if (path/name).is_file()},
                       'model_weights_loaded':False,'versions':{}}
        for name in ['torch','transformers','Pillow','tokenizers','torchvision']:
            try:self.identity['versions'][name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:self.identity['versions'][name]=None

    def prepare(self,messages:list[dict],root:Path,*,add_generation_prompt:bool=False):
        images=[];clean=[];view_ids=[]
        for message in messages:
            content=[]
            for ref in message['images']:
                path=contained_path(root,ref['path'])
                if file_sha256(path)!=ref['file_sha256']:raise ValueError('processor_input_image_file_changed')
                image=load_rgb(path)
                cap=self.budgets.initial_visual_tokens if ref['view_id']=='v0' else self.budgets.crop_visual_tokens
                image=resize_for_budget(image,cap)
                images.append(image);view_ids.append(ref['view_id'])
                content.append({'type':'image','image':image})
            content.append({'type':'text','text':message['text']})
            clean.append({'role':message['role'],'content':content})
        text=self.processor.apply_chat_template(clean,tokenize=False,add_generation_prompt=add_generation_prompt)
        encoded=self.processor(text=[text],images=images,padding=False,return_tensors='pt')
        ids=encoded['input_ids'][0].tolist();grid=encoded.get('image_grid_thw')
        if grid is None or len(grid)!=len(images):raise ValueError('native_image_count_mismatch')
        merge=self.processor.image_processor.merge_size
        counts=[int(row.prod().item())//(merge*merge) for row in grid]
        for view,count in zip(view_ids,counts):
            if count>(self.budgets.initial_visual_tokens if view=='v0' else self.budgets.crop_visual_tokens):
                raise ValueError('native_per_image_visual_budget_exceeded')
        image_id=self.processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
        if ids.count(image_id)!=sum(counts):raise ValueError('image_pad_and_grid_counts_differ')
        if len(ids)>self.budgets.context_tokens:raise ValueError('native_curriculum_context_overflow')
        if sum(counts)>self.budgets.new_visual_tokens:raise ValueError('native_curriculum_visual_total_overflow')
        return {'ids':ids,'image_grid_thw':grid.tolist(),'view_ids':view_ids,'visual_counts':counts,
                'context_tokens':len(ids),'input_ids_sha256':identity(ids)}


def check_native_row(row:dict,root:Path,processor:ProcessorOnly):
    proof=audit_replay(row,root)
    messages=row['messages'];assistant=[(i,m) for i,m in enumerate(messages) if m['role']=='assistant']
    for message in messages:
        if any(token in message['text'] for token in RESERVED):raise ValueError('reserved_chat_marker_in_curriculum_text')
    native=processor.prepare(messages,root)
    tokenizer=processor.processor.tokenizer
    start_id=tokenizer.convert_tokens_to_ids('<|im_start|>');end_id=tokenizer.convert_tokens_to_ids('<|im_end|>')
    header=[start_id,*tokenizer.encode('assistant\n',add_special_tokens=False)]
    newline=tokenizer.encode('\n',add_special_tokens=False)
    spans,mask=assistant_spans(native['ids'],header,end_id,newline,[m['supervised'] for _,m in assistant])
    prefix_mask=[False]*len(native['ids']);prefix_prepares=0
    for (index,message),span in zip(assistant,spans):
        before=processor.prepare(messages[:index],root,add_generation_prompt=True)
        through=processor.prepare(messages[:index+1],root)
        prefix_prepares+=2
        a,b=len(before['ids']),len(through['ids'])
        if before['ids']!=native['ids'][:a] or through['ids']!=native['ids'][:b]:
            raise ValueError('native_prefix_not_a_full_sequence_prefix')
        if a!=span['body_start'] or b!=span['stop']:raise ValueError('independent_prefix_and_header_oracles_disagree')
        if tokenizer.decode(native['ids'][a:span['end_token']],skip_special_tokens=False)!=message['text']:
            raise ValueError('native_assistant_body_does_not_equal_teacher_text')
        if message['supervised']:prefix_mask[a:b]=[True]*(b-a)
    if prefix_mask!=mask:raise ValueError('full_position_supervision_mismatch')
    labels=[token if use else -100 for token,use in zip(native['ids'],mask)]
    if not any(mask) or mask[0]:raise ValueError('no_supervision_or_invalid_first_token_target')
    image_id=tokenizer.convert_tokens_to_ids('<|image_pad|>')
    if any(use and token in {image_id,start_id} for token,use in zip(native['ids'],mask)):
        raise ValueError('visual_or_role_start_token_is_supervised')
    action_tokens=sum(span['stop']-span['body_start'] for span in spans)
    if action_tokens>processor.budgets.generated_tokens:raise ValueError('teacher_action_token_budget_exceeded')
    return {'trajectory_id':row['trajectory_id'],'sample_id':row['sample_id'],'state_kind':row['state_kind'],
            'curriculum_phase':row['curriculum_phase'],'source':row['source'],**proof,
            'native_context_tokens':native['context_tokens'],'visual_counts':native['visual_counts'],
            'image_grid_thw':native['image_grid_thw'],'view_ids':native['view_ids'],
            'input_ids_sha256':native['input_ids_sha256'],'labels_sha256':identity(labels),
            'input_ids':native['ids'],'labels':labels,'assistant_spans':spans,
            'supervised_tokens':sum(mask),'all_teacher_assistant_tokens':action_tokens,
            'supervised_end_tokens':sum(s['supervised'] for s in spans),'end_suffix_ids':[end_id,*newline],
            'prefix_prepares':prefix_prepares,'native_mask_oracles_agree':True,
            'sampled_token_ids':False,'token_provenance':'actual_tokenizer_encoding_of_teacher_not_model_generation',
            'model_weights_loaded':False,'model_generation_calls':0,'optimizer_steps':0}


def audit_native_curriculum(curriculum:str|Path,model_metadata:str|Path,output:str|Path,*,
                            max_records:int=16,wall_cap_seconds:int=300):
    begin=time.perf_counter();root=Path(curriculum);dest=Path(output)
    manifest=read_json(root/'manifest.json')
    for name,digest in manifest['file_sha256'].items():
        if file_sha256(contained_path(root,name))!=digest:raise ValueError('frozen_curriculum_files_changed')
    rows=read_jsonl(root/'trajectories.private.jsonl')
    if not 1<=max_records<=20000:raise ValueError('native审核条数须在1到20000之间')
    buckets=defaultdict(list)
    for row in rows:
        key=(row['curriculum_phase'],row['state_kind'],row['source'] if row['curriculum_phase']=='general_replay' else '')
        buckets[key].append(row)
    first=[sorted(items,key=lambda r:r['trajectory_id'])[0] for _,items in sorted(buckets.items())]
    if max_records<len(first):raise ValueError('max_records不足以覆盖所有课程状态及回放来源')
    used={r['trajectory_id'] for r in first}
    selected=first+sorted((r for r in rows if r['trajectory_id'] not in used),key=lambda r:r['trajectory_id'])[:max_records-len(first)]
    if dest.exists():raise ValueError('native审核attempt已存在，保留旧证据，不覆盖')
    dest.mkdir(parents=True)
    request={'curriculum_manifest_sha256':file_sha256(root/'manifest.json'),'selected_trajectory_ids':[r['trajectory_id'] for r in selected],
             'max_records':max_records,'wall_cap_seconds':wall_cap_seconds,'model_weights_loaded':False}
    write_json(dest/'launch_manifest.json',request)
    receipts=[]
    try:
        processor=ProcessorOnly(model_metadata)
        for row in selected:
            if time.perf_counter()-begin>wall_cap_seconds:raise RuntimeError('native_curriculum_CPU_wall_cap')
            receipts.append(check_native_row(row,root,processor))
        write_jsonl(dest/'native_masks.private.jsonl',receipts)
        result={'status':'passed','scope':'native_CPU_processor_masks_on_selected_teacher_records',
                'curriculum_records':len(rows),'records_checked':len(receipts),'full_curriculum_checked':len(receipts)==len(rows),
                'fixture':manifest['provenance_kind']=='synthetic_fixture','processor_identity':processor.identity,
                'supervised_tokens':sum(r['supervised_tokens'] for r in receipts),
                'context_max':max(r['native_context_tokens'] for r in receipts),
                'visual_count_max':max(v for r in receipts for v in r['visual_counts']),
                'state_kinds':sorted({r['state_kind'] for r in receipts}),
                'end_suffix_ids':receipts[0]['end_suffix_ids'],'native_mask_oracles_agree':True,
                'actual_Swift_trainer_mask_integration_verified':False,'ready_for_training':False,
                'model_weights_loaded':False,'model_generation_calls':0,'optimizer_steps':0,
                'CPU_wall_seconds':time.perf_counter()-begin,'request_sha256':identity(request),
                'native_mask_file_sha256':file_sha256(dest/'native_masks.private.jsonl')}
        write_json(dest/'receipt.json',result)
        return result
    except Exception as exc:
        write_jsonl(dest/'partial_native_masks.private.jsonl',receipts)
        write_json(dest/'failure.json',{'status':'failed','error_type':type(exc).__name__,'error':str(exc),
                   'records_checked':len(receipts),'CPU_wall_seconds':time.perf_counter()-begin,
                   'model_weights_loaded':False,'ready_for_training':False})
        raise
