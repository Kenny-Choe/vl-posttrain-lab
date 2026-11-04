"""可控规模的渲染数据fixture，永不冒充Unitail/RPC或真实人工标注。"""
from __future__ import annotations

import random
from pathlib import Path
from .schema import QARecord
from .release import annotation_identity,prepare_release,write_release
from ..io import file_sha256,write_json,write_jsonl


def create_fixture(root:str|Path,qa_count:int=96,seed:int=42) -> tuple[list[dict],list[dict]]:
    from PIL import Image,ImageDraw
    if qa_count<96 or qa_count%96 or qa_count>99936:raise ValueError('fixture规模必须是96的倍数，范围96到99936')
    root=Path(root);images=root/'images';images.mkdir(parents=True,exist_ok=True)
    records=[];reviews=[]
    positions=[(5,5),(69,5),(5,69),(69,69)]
    for number in range(qa_count//4):
        rng=random.Random(seed+number)
        texture=Image.frombytes('RGB',(8,8),rng.randbytes(8*8*3))
        image=texture.resize((128,128),Image.Resampling.NEAREST);draw=ImageDraw.Draw(image)
        fields=[]
        for field,(x,y) in enumerate(positions):
            answer=f'{number:05d}{field}'
            rectangle=list(draw.textbbox((x,y),answer))
            draw.rectangle((rectangle[0]-2,rectangle[1]-2,rectangle[2]+2,rectangle[3]+2),fill='white')
            draw.text((x,y),answer,fill='black');fields.append((answer,rectangle))
        filename=f'image_{number:05d}.png';path=images/filename;image.save(path);digest=file_sha256(path)
        for field,(answer,rectangle) in enumerate(fields):
            sid=f'synthetic-qa-{number:05d}-{field}'
            record=QARecord(sample_id=sid,source='synthetic_packaging_fixture',source_revision='render-v1',native_qa_id=sid,
                source_split='train',image_id=f'fixture-image-{number}',image_path='images/'+filename,image_sha256=digest,
                image_width=128,image_height=128,question=f"What is the rendered code at the {['top-left','top-right','bottom-left','bottom-right'][field]} corner?",answers=[answer],
                image_group_id='pending_derived_group',project_split='train',task_type='synthetic_code_reading',
                product_family_id=f'fixture-family-{number//4}',evidence_box_xyxy=rectangle,annotation_version='synthetic-render-v1',
                evidence_source='programmatic_textbbox',provenance_kind='synthetic_fixture',native_gold_match='renderer_known_not_human')
            row=record.to_dict();records.append(row)
            reviews.append({'sample_id':sid,'review_id':'fixture-review-'+sid,'reviewer_id':'programmatic_renderer_not_human',
                'reviewer_type':'synthetic_fixture','annotation_sha256':annotation_identity(row),'status':'accepted',
                'question_unique':True,'answer_visible':True,'evidence_tight':True,'family_confirmed':True,
                'family_review_id':'fixture-family-rule-v1'})
    write_jsonl(root/'input_records.fixture.jsonl',records);write_jsonl(root/'input_reviews.fixture.jsonl',reviews)
    return records,reviews


def dataset_demo(output:str|Path,qa_count:int=96,seed:int=42) -> dict:
    dest=Path(output)
    records,reviews=create_fixture(dest,qa_count,seed)
    counts={'train':qa_count*2//3,'dev':qa_count//6,'test':qa_count//6}
    result=prepare_release(records,reviews,dest,counts,scope='synthetic_fixture',seed=seed)
    edges=[]
    if result['phash_review_queue']:
        # 独立渲染来源的已知fixture关系；绝不伪装成人工审阅真实照片。
        edges=[{**pair,'confirmed':False,'reviewed':True,'basis':'fixture_render_provenance_distinct'} for pair in result['phash_review_queue']]
        result=prepare_release(records,reviews,dest,counts,scope='synthetic_fixture',seed=seed,reviewed_edges=edges)
    write_jsonl(dest/'distinct_edges.fixture.jsonl',edges)
    manifest=write_release(dest/'release',result)
    report={'status':manifest['status'],'data_kind':'synthetic_rendered_fixture_not_real_packaging_dataset',
            'actual_fixture_QA':qa_count,'actual_fixture_images':qa_count//4,'actual_counts':manifest['actual_counts'],
            'actual_family_groups':len({r['image_group_id'] for rows in result['splits'].values() for r in rows}),
            'release_sha256':manifest['release_sha256'],'fixture_distinct_pair_reviews':len(edges),
            'reviewed_real_data_release_passed':False,'target_12k_dataset_counts_met':False,
            'model_weights_loaded':False,'model_generation_calls':0,'optimizer_steps':0,
            'phase_CPU_wall_seconds':manifest['CPU_wall_seconds']}
    write_json(dest/'demo_manifest.json',report)
    return report
