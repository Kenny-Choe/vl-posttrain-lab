import unittest
from vl_posttrain.data.grouping import build_groups, split_and_sample
from vl_posttrain.data.selection import largest_remainder, select
from vl_posttrain.data.nll import target_ids, shifted_nll, modality_conditions, visual_dependency
from vl_posttrain.data.schema import normalize_native_qa, match_native_gold
from vl_posttrain.data.preferences import build_preference
from vl_posttrain.data.sources import rpc_exemplar_candidates, unitail_reviewed_region


class DataTests(unittest.TestCase):
    def test_only_confirmed_edges_union_and_family_protection_propagates(self):
        images = [{"image_id":"a","product_family_id":"family1"},{"image_id":"b","product_family_id":"family1"},{"image_id":"c"}]
        result = build_groups(images,[{"left":"b","right":"c","confirmed":False}],{"a"})
        self.assertEqual(result["mapping"]["a"], result["mapping"]["b"])
        self.assertNotEqual(result["mapping"]["a"], result["mapping"]["c"])
        self.assertIn(result["mapping"]["b"],result["quarantined_groups"])

    def test_group_id_independent_of_input_order_and_unknown_protection_rejected(self):
        images = [{"image_id":"b"},{"image_id":"a"}]
        edges = [{"left":"a","right":"b","confirmed":True,"basis":"known_crop"}]
        self.assertEqual(build_groups(images,edges),build_groups(images[::-1],edges))
        with self.assertRaises(ValueError):
            build_groups(images,[],{"not_indexed"})

    def test_split_does_not_duplicate_group_or_pad_shortfall(self):
        rows=[{"sample_id":str(i),"image_group_id":str(i//2)} for i in range(20)]
        result=split_and_sample(rows,{"train":8,"dev":2,"test":2})
        sets=[{r["image_group_id"] for r in result["splits"][key]} for key in ("train","dev","test")]
        self.assertFalse(sets[0]&sets[1] or sets[1]&sets[2] or sets[0]&sets[2])
        with self.assertRaises(ValueError):
            split_and_sample(rows,{"train":30})

    def test_largest_remainders_have_exact_total(self):
        self.assertEqual(sum(largest_remainder({"a":13,"b":17,"c":22},31).values()),31)
        with self.assertRaises(ValueError):
            largest_remainder({"a":2},3)

    def test_native_test_cannot_be_laundered_by_new_project_split(self):
        with self.assertRaises(ValueError):
            split_and_sample([{"sample_id":"q","image_group_id":"g","source_split":"test"}],{"train":1})

    def test_nll_uses_previous_logit_not_same_position(self):
        import math
        value,n=shifted_nll([[0,2],[2,0],[0,0]],[-100,1,-100])
        self.assertEqual(n,1)
        self.assertAlmostEqual(value,math.log1p(math.exp(-2)))
        self.assertEqual(target_ids([5,6,7],[-100,-100,7],[False,False,True]),[7])
        with self.assertRaises(ValueError):
            target_ids([5,6],[5,-100],[True,False])

    def test_modality_rebuild_preserves_answer_and_target_identity(self):
        messages=[{"role":"user","content":[{"type":"image","image":"image"},{"type":"text","text":"q"}]},
                  {"role":"assistant","content":[{"type":"text","text":"answer"}]}]
        left,right=modality_conditions(messages)
        self.assertEqual(left[-1],right[-1])
        self.assertEqual(len(right[0]["content"]),1)
        self.assertEqual(visual_dependency(1.,2.,[3],[3]),1.)
        with self.assertRaises(ValueError):
            visual_dependency(1.,2.,[3],[4])

    def test_source_images_string_and_unverified_gold(self):
        record=normalize_native_qa({"images":"a.png","conversations":[{"role":"user","content":"<image> Q"},{"role":"assistant","content":"generated"}]},"cold","rev")
        self.assertEqual(record["image_path"],"a.png")
        self.assertEqual(record["native_gold_match"],"unverified")

    def test_gold_matching_rejects_same_image_different_question_and_remaps_options(self):
        original={"source":"s","image_id":"i","native_qa_id":"1","question":"lowest?","choices":["Tue","Wed"],"answer":"A"}
        candidate={"source":"s","image_id":"i","question":"lowest?","choices":["Wed","Tue"]}
        self.assertEqual(match_native_gold(candidate,[original])["answer"],"B")
        candidate["question"]="highest?"
        with self.assertRaises(ValueError):
            match_native_gold(candidate,[original])

    def test_preference_requires_actual_quality_difference(self):
        row={"project_split":"train","native_gold_match":"question_choices_image_checked","sample_id":"q","image_group_id":"g",
             "candidates":[{"candidate_id":"1","text":"good","verified_correct":True,"valid":True},
                           {"candidate_id":"2","text":"bad","verified_correct":False}]}
        self.assertEqual(build_preference(row)["chosen"],"good")
        row["candidates"][1]["verified_correct"]=True
        with self.assertRaises(ValueError):
            build_preference(row)

    def test_rpc_product_bbox_cannot_become_evidence_qa(self):
        coco={"images":[{"id":1,"file_name":"1.jpg","width":100,"height":100}],
              "annotations":[{"id":2,"image_id":1,"category_id":3,"bbox":[1,2,80,90]}],"categories":[]}
        row=rpc_exemplar_candidates(coco,"r1")[0]
        self.assertNotIn("answers",row)
        self.assertNotIn("evidence_box_xyxy",row)

    def test_unitail_test_components_do_not_enter_training(self):
        row={"image_id":"i","image_path":"i.jpg","width":100,"height":100,"region_id":"1", "polygon":[[1,1],[90,1],[90,90],[1,90]],
             "transcription":"500g","component":"matching_test","review_receipt":"review1"}
        with self.assertRaises(ValueError):
            unitail_reviewed_region(row,"r")


try:
    import numpy
except ImportError:
    numpy=None


@unittest.skipIf(numpy is None,"PCG64 选数需 NumPy 可选依赖")
class SelectionTests(unittest.TestCase):
    def test_selection_exact_quota_reproducible_and_input_order_independent(self):
        rows=[{"sample_id":str(i),"source":str(i%3),"task_type":"qa","project_split":"train","quality_flags":[],"visual_dependency":i/30} for i in range(60)]
        left=select(rows,20,"visual_mixture")
        right=select(rows[::-1],20,"visual_mixture")
        self.assertEqual(left,right)
        self.assertEqual(sum(left["high_quotas"].values()),10)
        self.assertEqual(len(left["rows"]),20)

    def test_nan_score_or_eval_input_rejected(self):
        row={"sample_id":"q","source":"s","task_type":"qa","project_split":"train","quality_flags":[],"visual_dependency":float("nan")}
        with self.assertRaises(ValueError):
            select([row],1,"visual_mixture")
