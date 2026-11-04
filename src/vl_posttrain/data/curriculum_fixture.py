"""独立的通用回放渲染fixture；不把程序生成的数据改名为真实GQA/TextVQA。"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from .demo import create_fixture
from .schema import QARecord
from ..io import write_jsonl,write_json


def create_replay_fixture(output:str|Path,count:int=1000,seed:int=77123) -> list[dict]:
    if isinstance(count,bool) or not isinstance(count,int) or count<4 or count%4:
        raise ValueError('fixture回放数必须为正且能按两源两阶段等分')
    root=Path(output)
    if root.exists():raise ValueError('保留原fixture，使用新目录')
    root.mkdir(parents=True)
    total=((count+95)//96)*96
    records,_=create_fixture(root,total,seed)
    selected=[]
    for i,row in enumerate(records[:count]):
        source='synthetic_GQA_fixture' if i<count//2 else 'synthetic_TextVQA_fixture'
        record=QARecord(**row)
        sid='replay-fixture-'+record.sample_id
        changed=replace(record,sample_id=sid,native_qa_id=sid,source=source,
                        image_id='replay-'+record.image_id,image_group_id='replay-'+record.product_family_id,
                        product_family_id='replay-'+record.product_family_id,evidence_box_xyxy=None)
        selected.append(changed.to_dict())
    write_jsonl(root/'replay.private.jsonl',selected)
    write_json(root/'fixture_manifest.json',{'provenance_kind':'synthetic_fixture','selected_QA':count,
        'rendered_QA':total,'extra_rendered_QA_not_selected':total-count,
        'real_GQA_or_TextVQA_acquired':False,'model_weights_loaded':False,'optimizer_steps':0})
    return selected
