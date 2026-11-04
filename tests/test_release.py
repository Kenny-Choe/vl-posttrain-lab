import copy
import tempfile
import unittest
from pathlib import Path
from vl_posttrain.data.demo import create_fixture
from vl_posttrain.data.release import prepare_release,write_release,annotation_identity,image_uid
from vl_posttrain.data.grouping import image_fingerprints,split_and_sample,phash_candidates,CandidateBudgetExceeded
from vl_posttrain.io import read_jsonl,read_json
from vl_posttrain.data.schema import QARecord


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        base=Path(__file__).resolve().parents[1]/'outputs/test-temp';base.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=base);self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        assert self.root.resolve().is_relative_to(base.resolve())
        self.records,self.reviews=create_fixture(self.root,96)
    def prepare(self,**kwargs):
        return prepare_release(self.records,self.reviews,self.root,{'train':64,'dev':16,'test':16},scope='synthetic_fixture',**kwargs)
    def resign(self,index):
        self.reviews[index]['annotation_sha256']=annotation_identity(self.records[index])
    def test_end_to_end_fixture_release_counts_and_family_isolation(self):
        result=self.prepare()
        self.assertTrue(result['manifest']['counts_met'])
        self.assertFalse(result['manifest']['reviewed_real_data_release_passed'])
        self.assertFalse(result['manifest']['target_12k_dataset_counts_met'])
        split_ids=[{r['image_group_id'] for r in result['splits'][key]} for key in ('train','dev','test')]
        self.assertFalse(split_ids[0]&split_ids[1] or split_ids[1]&split_ids[2] or split_ids[0]&split_ids[2])
        families={}
        for split,rows in result['splits'].items():
            for row in rows:
                self.assertEqual(families.setdefault(row['product_family_id'],split),split)
    def test_review_bound_to_answers_and_tight_box(self):
        self.records[0]['answers']=['changed']
        result=self.prepare()
        flagged=next(q for q in result['quarantine'] if q['sample_id']==self.records[0]['sample_id'])
        self.assertIn('review_annotation_identity_mismatch',flagged['reasons'])
        self.assertFalse(result['manifest']['counts_met'])
    def test_image_content_mismatch_is_detected_not_relabeled(self):
        (self.root/self.records[0]['image_path']).write_bytes(b'not_an_image')
        result=self.prepare()
        self.assertGreaterEqual(result['manifest']['quarantined_QA'],4)
        self.assertFalse(result['manifest']['counts_met'])
    def test_protected_member_quarantines_entire_family(self):
        first=self.records[0]
        protected={'image_id':'protected-alias',**image_fingerprints(str(self.root/first['image_path']))}
        result=self.prepare(protected_images=[protected])
        bad={q['sample_id'] for q in result['quarantine']}
        self.assertTrue({r['sample_id'] for r in self.records if r['product_family_id']==first['product_family_id']}<=bad)
        self.assertFalse(result['manifest']['counts_met'])
    def test_upstream_eval_without_bbox_still_protects_photo(self):
        self.records[0]['source_split']='matching_test';self.records[0]['evidence_box_xyxy']=None
        result=self.prepare()
        self.assertEqual(result['manifest']['protected_images'],1)
        flagged=next(q for q in result['quarantine'] if q['sample_id']==self.records[0]['sample_id'])
        self.assertEqual(flagged['reasons'],['upstream_evaluation_protected'])
    def test_image_alias_cannot_bypass_max_four_questions(self):
        extra=copy.deepcopy(self.records[0]);extra['sample_id']='extra';extra['image_id']='different-id-same-pixels';extra['question']='Additional fifth question?'
        review=copy.deepcopy(self.reviews[0]);review.update(sample_id='extra',annotation_sha256=annotation_identity(extra))
        self.records.append(extra);self.reviews.append(review)
        result=self.prepare()
        counts={}
        for row in result['eligible_candidates']:counts[row['image_id']]=counts.get(row['image_id'],0)+1
        self.assertLessEqual(max(counts.values()),4)
        self.assertTrue(any(r['reason']=='per_image_QA_cap_not_label_error' for r in result['unused']))
    def test_conflicting_duplicate_labels_both_require_adjudication(self):
        extra=copy.deepcopy(self.records[0]);extra['sample_id']='duplicate-conflict';extra['answers']=['different']
        review=copy.deepcopy(self.reviews[0]);review.update(sample_id=extra['sample_id'],annotation_sha256=annotation_identity(extra))
        self.records.append(extra);self.reviews.append(review)
        result=self.prepare()
        reasons={r['sample_id']:r['reasons'] for r in result['quarantine']}
        self.assertIn('conflicting_duplicate_annotations_needs_adjudication',reasons[self.records[0]['sample_id']])
        self.assertIn('conflicting_duplicate_annotations_needs_adjudication',reasons['duplicate-conflict'])
    def test_synthetic_review_cannot_publish_observed_recipe(self):
        result=prepare_release(self.records,self.reviews,self.root,{'train':1},scope='packaging')
        self.assertFalse(result['manifest']['reviewed_real_data_release_passed'])
        self.assertFalse(result['manifest']['counts_met'])
    def test_writer_keeps_actor_private_fields_out_and_preserves_release(self):
        result=self.prepare();dest=self.root/'release';first=write_release(dest,result)
        public=read_jsonl(dest/'train.public.jsonl')
        self.assertEqual(set(public[0]),{'task_id','question','image_id','image_path','image_sha256'})
        self.assertEqual(write_release(dest,self.prepare())['release_sha256'],first['release_sha256'])
        (dest/'train.public.jsonl').write_text('changed')
        with self.assertRaises(ValueError):write_release(dest,self.prepare())
    def test_writer_rejects_new_input_reusing_old_release_directory(self):
        write_release(self.root/'release',self.prepare())
        self.records[0]['answers']=['different']
        with self.assertRaises(ValueError):write_release(self.root/'release',self.prepare())
    def test_unknown_protection_fingerprint_is_partial_not_false_clean(self):
        protected={'image_id':'other-protected','file_sha256':'1'*64,'pixel_sha256':'2'*64,'version':'unknown'}
        result=self.prepare(protected_images=[protected])
        self.assertFalse(result['manifest']['phash_scan_complete'])
        self.assertEqual(result['manifest']['status'],'partial_needs_review')
        self.assertIn('other-protected',result['manifest']['unknown_protection_fingerprint_versions'])
    def test_capacity_reservation_finds_feasible_case_previous_greedy_missed(self):
        rows=[{'sample_id':f'{g}-{i}','image_group_id':g} for g,size in [('a',4),('b',3),('c',3)] for i in range(size)]
        result=split_and_sample(rows,{'train':6,'dev':4})
        self.assertEqual({k:len(v) for k,v in result['splits'].items()},{'train':6,'dev':4})
    def test_phash_scan_has_explicit_budget_and_validates_64bit_input(self):
        rows=[{'image_id':str(i),'phash64':'0'*16} for i in range(10)]
        with self.assertRaises(CandidateBudgetExceeded) as context:phash_candidates(rows,max_pairs=3)
        self.assertEqual(len(context.exception.pairs),3)
        with self.assertRaises(ValueError):phash_candidates([{'image_id':'a','phash64':'abc'}])
    def test_record_rejects_answer_string_not_list(self):
        bad=copy.deepcopy(self.records[0]);bad['answers']='not_a_reference_list'
        with self.assertRaises(ValueError):QARecord(**bad).validate()
