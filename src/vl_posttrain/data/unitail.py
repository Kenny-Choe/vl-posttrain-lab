"""Unitail-OCR gallery v2原生区域适配；候选不自动升级为已审核QA。"""
from __future__ import annotations
from collections import Counter

from .geometry_bridge import checked_text_region
from ..io import identity

UNREADABLE={'','*','###','???'}


def gallery_regions(coco:dict,revision:str) -> dict:
    if not revision:raise ValueError('upstream_archive_revision_required')
    images={row['id']:row for row in coco['images']}
    if len(images)!=len(coco['images']):raise ValueError('duplicate_native_gallery_image_id')
    annotations=coco['annotations'];ids=[row['id'] for row in annotations]
    if len(set(ids))!=len(ids):raise ValueError('duplicate_native_text_region_id')
    candidates=[];quarantine=[]
    for raw in annotations:
        iid=raw['image_id'];aid=raw['id']
        if iid not in images:raise ValueError('native_region_references_missing_image')
        image=images[iid];text=raw.get('text-words')
        if not isinstance(text,str) or text.strip() in UNREADABLE:
            quarantine.append({'native_region_id':aid,'image_id':iid,'reason':'native_unreadable_marker'});continue
        try:
            segments=raw['segmentation']
            if len(segments)!=1:raise ValueError('multiple_native_polygons_not_a_single_tight_region')
            values=segments[0]
            if len(values)<6 or len(values)%2:raise ValueError('invalid_native_polygon')
            polygon=[values[i:i+2] for i in range(0,len(values),2)]
            rectangle=checked_text_region(polygon,image['width'],image['height'])
        except (ValueError,KeyError,TypeError) as exc:
            quarantine.append({'native_region_id':aid,'image_id':iid,'reason':'native_geometry_needs_review:'+str(exc)});continue
        candidates.append({'candidate_id':identity(['Unitail-OCR',revision,iid,aid]),'source':'Unitail-OCR',
            'source_revision':revision,'source_split':'gallery_train','native_region_id':aid,'image_id':iid,
            'image_path':image['file_name'],'image_width':image['width'],'image_height':image['height'],
            'transcription':text.strip(),'native_polygon':polygon,'region_box_xyxy':rectangle,
            'native_annotation_sha256':identity(raw),'question_created':False,
            'answer_and_semantic_review':'pending','product_family_review':'pending',
            'native_word_annotation_does_not_guarantee_correct_transcription':True})
    return {'candidates':candidates,'quarantine':quarantine,
        'summary':{'source':'Unitail-OCR','source_revision':revision,'images':len(images),
            'text_regions':len(annotations),'non_ignore_candidates_with_valid_geometry':len(candidates),
            'quarantined':len(quarantine),'reasons':dict(Counter(row['reason'].split(':')[0] for row in quarantine)),
            'QA_created':0,'reviewed_real_QA':0,'model_weights_loaded':False}}
