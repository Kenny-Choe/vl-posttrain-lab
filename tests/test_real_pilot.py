import copy
import tempfile
import unittest
from pathlib import Path
from vl_posttrain.data.real_pilot import validate_CPU_review,build_real_pilot
from vl_posttrain.data.release import annotation_identity
from vl_posttrain.data.schema import QARecord
from vl_posttrain.io import file_sha256


def review_for(record):
    return {'sample_id':record['sample_id'],'annotation_sha256':annotation_identity(record),
            'reviewer_type':'assistant_visual','status':'accepted_for_CPU_pilot',
            'scope':'real_CPU_pilot_not_formal_training_release','visual_evidence_sha256':{'unit_fixture':'a'*64},
            'native_annotation_provenance':{'unit_fixture':True},'question_unique_within_image':True,
            'answer_visible':True,'evidence_corresponds_to_asked_field':True,
            'family_grouped_within_pilot':True,'formal_full_data_release_approved':False}


class RealPilotTests(unittest.TestCase):
    def test_annotation_change_or_claiming_formal_release_invalidates_CPU_review(self):
        record={'sample_id':'q','question':'question','answers':['one'],'evidence_box_xyxy':[1,2,3,4]}
        proof=review_for(record);validate_CPU_review(record,proof)
        changed=copy.deepcopy(record);changed['answers']=['two']
        with self.assertRaises(ValueError):validate_CPU_review(changed,proof)
        proof['formal_full_data_release_approved']=True
        with self.assertRaises(ValueError):validate_CPU_review(record,proof)

    def test_CPU_only_flags_preserved_during_cohort_split_and_never_training_ready(self):
        try:from PIL import Image,ImageDraw
        except ImportError:self.skipTest('real pixel contract needs Pillow')
        scratch=Path(__file__).resolve().parents[1]/'outputs'/'test_scratch'
        scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            root=Path(temporary);domain=[];replay=[]
            self.assertTrue(root.resolve().is_relative_to(scratch.resolve()))
            for i in range(12):
                path=root/f'{i}.png';image=Image.new('RGB',(96,96),'white');draw=ImageDraw.Draw(image)
                draw.text((12,18),f'item{i}',fill='black');draw.rectangle((i+1,60,i+12,80),fill=(i*19,20,60));image.save(path)
                source='Unitail-OCR' if i<8 else 'GQA' if i<10 else 'TextVQA'
                for q in range(2 if i<8 else 1):
                    record=QARecord(sample_id=f'q{i}-{q}',source=source,source_revision='unit-test-fixture',native_qa_id=f'{i}-{q}',
                        source_split='gallery_train' if i<8 else 'train',image_id=str(i),image_path=path.name,
                        image_sha256=file_sha256(path),image_width=96,image_height=96,question=f'question {q}',answers=['item'],
                        image_group_id=f'group{i}',product_family_id=f'family{i}',project_split='train',task_type='unit_fixture',
                        evidence_box_xyxy=[10,16,45,32] if i<8 else None,annotation_version='unit-fixture',evidence_source='unit-fixture',
                        quality_flags=['CPU_only_pending_full_protection'] if i<8 else [],
                        native_gold_match='native_annotation_verified').to_dict()
                    (domain if i<8 else replay).append(record)
            result=build_real_pilot(domain,[review_for(r) for r in domain],replay,root,root/'out',counts={'train':12,'dev':2,'test':2})
            self.assertFalse(result['ready_for_training'])
            self.assertFalse(result['formal_12K_data_release_ready'])
            from vl_posttrain.io import read_jsonl
            for split in ['train','dev','test']:
                rows=read_jsonl(root/'out'/f'{split}.private.jsonl')
                self.assertTrue(all(r['quality_flags']==['CPU_only_pending_full_protection'] for r in rows))
                self.assertTrue(all(r['project_split']==split for r in rows))
            self.assertEqual(result['curriculum']['trajectories'],16)
