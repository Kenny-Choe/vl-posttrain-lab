import copy
from collections import Counter
from pathlib import Path
import tempfile
import unittest

from vl_posttrain.io import read_json,read_jsonl,write_json
from vl_posttrain.data.demo import dataset_demo
from vl_posttrain.data.curriculum_fixture import create_replay_fixture
from vl_posttrain.data.curriculum_build import build_curriculum,audit_replay,SEARCH_STATES,LOCAL_STATES

ROOT=Path(__file__).resolve().parents[1]


class CurriculumTests(unittest.TestCase):
    def setUp(self):
        base=ROOT/'outputs/test-temp';base.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=base);self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.domain=self.root/'domain';dataset_demo(self.domain,96)
        self.general=self.root/'general';self.replay=create_replay_fixture(self.general,8)

    def build(self,name='course',**kwargs):
        return build_curriculum(self.domain/'release',self.domain,self.replay,self.general,self.root/name,
                                localization=32,search_stop=32,general_replay=8,**kwargs)

    def test_full_teacher_fixture_replays_real_pixels_and_all_behavior_states(self):
        result=self.build();rows=read_jsonl(self.root/'course/trajectories.private.jsonl')
        self.assertEqual(result['counts'],{'localization':32,'search_stop':32,'general_replay':8})
        self.assertEqual(result['stage_counts'],{'1':36,'2':36})
        self.assertEqual(set(result['state_counts']),set(SEARCH_STATES)|set(LOCAL_STATES)|{'general_answer'})
        self.assertFalse(result['ready_for_training']);self.assertFalse(result['real_9k_curriculum_completed'])
        self.assertFalse(result['model_weights_loaded'])
        for row in rows:
            proof=audit_replay(row,self.root/'course')
            self.assertTrue(proof['pixel_replay_passed'])
            if row['state_kind']=='wrong_repeat_then_correct':self.assertEqual(proof['crop_calls'],3)

    def test_replay_keeps_plain_QA_and_balances_both_sources_within_each_stage(self):
        self.build();rows=read_jsonl(self.root/'course/trajectories.private.jsonl')
        for stage in [1,2]:
            general=[r for r in rows if r['training_stage']==stage and r['curriculum_phase']=='general_replay']
            self.assertEqual(Counter(r['source'] for r in general),{'synthetic_GQA_fixture':2,'synthetic_TextVQA_fixture':2})
            for row in general:
                self.assertEqual(len(row['actions']),1)
                self.assertEqual(row['actions'][0]['text'],row['private_supervision']['answers'][0])
                self.assertNotIn('evidence',row['actions'][0]['text'])

    def test_injected_error_is_context_only_and_early_view_final_really_references_earlier_view(self):
        self.build();rows=read_jsonl(self.root/'course/trajectories.private.jsonl')
        row=next(r for r in rows if r['state_kind']=='earlier_view_final')
        self.assertIn('"view_id":"v1"',row['actions'][-1]['text'])
        self.assertIn('v2',row['views'])
        self.assertFalse(row['actions'][-2]['supervised'])
        corrupt=copy.deepcopy(row);corrupt['actions'][-2]['supervised']=True
        corrupt['messages'][-3]['supervised']=True
        with self.assertRaises(ValueError):audit_replay(corrupt,self.root/'course')

    def test_observation_metadata_does_not_contain_gold_fields(self):
        self.build();rows=read_jsonl(self.root/'course/trajectories.private.jsonl')
        import json
        for row in rows:
            for message in row['messages']:
                if message['role']=='tool':
                    self.assertEqual(set(json.loads(message['text'])),{'view_id','parent_id','natural_size','root_box'})
                    self.assertFalse(message['supervised'])

    def test_crop_file_tampering_is_detected_by_independent_replay(self):
        self.build();rows=read_jsonl(self.root/'course/trajectories.private.jsonl')
        row=next(r for r in rows if 'v1' in r['views'])
        (self.root/'course'/row['views']['v1']['path']).write_bytes(b'changed')
        with self.assertRaises((ValueError,OSError)):audit_replay(row,self.root/'course')

    def test_existing_output_and_modified_published_input_are_not_silently_overwritten(self):
        self.build()
        with self.assertRaises(ValueError):self.build()
        path=self.domain/'release/train.private.jsonl';path.write_text(path.read_text(encoding='utf-8')+'\n',encoding='utf-8')
        with self.assertRaises(ValueError):self.build('after_tamper')

    def test_replay_heldout_and_cross_domain_photo_are_rejected(self):
        self.replay[0]['project_split']='dev'
        with self.assertRaises(ValueError):self.build()
        self.replay[0]['project_split']='train';self.replay[0]['source_split']='test_extra'
        with self.assertRaises(ValueError):self.build()
        self.replay[0]['source_split']='train'
        other=read_jsonl(self.domain/'release/test.private.jsonl')[0]
        self.replay[0].update(image_sha256=other['image_sha256'],image_path=other['image_path'])
        from shutil import copyfile
        copyfile(self.domain/other['image_path'],self.general/other['image_path'])
        with self.assertRaises(ValueError):self.build()

    def test_replay_same_photo_same_question_cannot_be_counted_twice_by_new_ID(self):
        a,b=self.replay[:2]
        b.update(image_path=a['image_path'],image_sha256=a['image_sha256'],question=a['question'])
        with self.assertRaises(ValueError):self.build()

    def test_missing_replay_and_changed_source_recipe_are_not_padded(self):
        removed=self.replay.pop()
        with self.assertRaises(ValueError):self.build()
        self.replay.append(removed);self.replay[0]['source']='GQA'
        with self.assertRaises(ValueError):self.build()

    def test_partial_reserved_marker_is_quarantined_without_changing_remaining_quota(self):
        # Construct a new, internally hash-valid fixture release containing a bad
        # reserved marker. The curriculum builder must isolate it, not substitute.
        path=self.domain/'release/train.private.jsonl';rows=read_jsonl(path)
        rows[0]['question']='bad <|im_start|> source marker'
        from vl_posttrain.io import write_jsonl,file_sha256
        write_jsonl(path,rows);manifest=read_json(self.domain/'release/manifest.json')
        manifest['file_sha256']['train.private.jsonl']=file_sha256(path);write_json(self.domain/'release/manifest.json',manifest)
        result=self.build()
        self.assertEqual(result['status'],'partial');self.assertEqual(result['quarantined'],1)
        self.assertEqual(result['trajectories'],71)
