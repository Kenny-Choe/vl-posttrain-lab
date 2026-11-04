import unittest
from vl_posttrain.evaluation.scoring import reward, answer_matches, repeated_crop, token_f1
from vl_posttrain.evaluation.metrics import aggregate, pass_at_k
from vl_posttrain.evaluation.bootstrap import paired_group_bootstrap
from vl_posttrain.evaluation.benchmarks import anls, relaxed_numeric, retention_delta


def row(sample="q", group="g", correct=False):
    return {"run_id":"r", "sample_id":sample, "group_id":group, "answer_correct":correct,
            "joint_success":correct, "iou":1. if correct else 0., "valid":True, "crop_calls":2,
            "generated_tokens":100, "model_seconds":2., "latency_seconds":2., "repeated_crop":False}


class EvaluationTests(unittest.TestCase):
    def test_reward_and_joint_are_different_objectives(self):
        value=reward(True,[0,0,49,100],[0,0,100,100],100,100,True)
        self.assertGreater(value["reward"],1.)
        self.assertFalse(value["joint_success"])

    def test_invalid_output_has_zero_reward_even_if_answer_and_box_correct(self):
        result=reward(True,[0,0,100,100],[0,0,100,100],100,100,False)
        self.assertEqual(result["reward"],0.)
        self.assertFalse(result["joint_success"])

    def test_all_wrong_can_have_localization_signal_but_not_iou_answer_bonus(self):
        near=reward(False,[10,10,20,20],[10,10,20,20],100,100,True)
        far=reward(False,[80,80,90,90],[10,10,20,20],100,100,True)
        self.assertAlmostEqual(near["reward"],.05)
        self.assertLess(far["reward"],near["reward"])

    def test_answer_aliases_explicit_not_fuzzy_units(self):
        self.assertTrue(answer_matches("５００ g",["500g"]))
        self.assertFalse(answer_matches("0.5kg",["500g"]))
        self.assertTrue(answer_matches("0.5kg",["500g"],{"0.5kg":"500g"}))
        self.assertFalse(answer_matches("500mg",["500g"]))

    def test_repeat_requires_root_overlap_area_and_effective_resolution(self):
        crops=[{"root_box":[0,0,100,100],"effective_resolution":1.}, {"root_box":[0,0,100,100],"effective_resolution":1.05}]
        self.assertTrue(repeated_crop(crops))
        crops[-1]["effective_resolution"]=1.2
        self.assertFalse(repeated_crop(crops))
        crops[-1].pop("effective_resolution")
        with self.assertRaises(ValueError):
            repeated_crop(crops)

    def test_failed_attempts_in_success_cost_numerator_and_zero_success_unknown(self):
        result=aggregate([row("a",correct=True),row("b")])
        self.assertEqual(result["generated_tokens_per_success"],200)
        self.assertEqual(result["joint_success_rate"],.5)
        self.assertIsNone(aggregate([row()])["generated_tokens_per_success"])

    def test_duplicate_predictions_rejected(self):
        with self.assertRaises(ValueError):
            aggregate([row(),row()])

    def test_metrics_reject_boolean_strings_and_contradictory_joint(self):
        invalid=row()
        invalid["answer_correct"]="false"
        with self.assertRaises(ValueError):
            aggregate([invalid])
        invalid=row(correct=True)
        invalid["joint_success"]=False
        with self.assertRaises(ValueError):
            aggregate([invalid])

    def test_pass_at_k_requires_independent_real_samples(self):
        rows=[{**row(correct=i==0),"rollout_index":i,"independent_sample":True} for i in range(4)]
        self.assertEqual(pass_at_k(rows,4)["pass_at_k"],1.)
        self.assertEqual(pass_at_k(rows,1)["pass_at_k"],.25)
        rows[-1]["independent_sample"]=False
        with self.assertRaises(ValueError):
            pass_at_k(rows,4)

    def test_group_bootstrap_resamples_whole_groups_and_is_paired(self):
        left=[row("a","g1"),row("b","g1"),row("c","g2")]
        right=[row("a","g1",True),row("b","g1",True),row("c","g2")]
        value=paired_group_bootstrap(left,right,iterations=100)
        self.assertEqual(value["groups"],2)
        self.assertAlmostEqual(value["difference_right_minus_left"],2/3)
        self.assertEqual(value,paired_group_bootstrap(left,right,iterations=100))
        with self.assertRaises(ValueError):
            paired_group_bootstrap(left,right[:-1])

    def test_reference_scorers_do_not_conflate_metrics(self):
        self.assertEqual(relaxed_numeric("105",["100"]),1.)
        self.assertEqual(relaxed_numeric("106",["100"]),0.)
        self.assertEqual(anls("abcd",["abce"]),.75)
        self.assertEqual(token_f1("净含量 500g","净含量 500g"),1.)

    def test_ocr_retention_needs_same_full_set(self):
        value=retention_delta([{"sample_id":"a","score":1.}],[{"sample_id":"a","score":0.}])
        self.assertEqual(value["delta_percentage_points"],-100.)
        with self.assertRaises(ValueError):
            retention_delta([{"sample_id":"a","score":1.}],[{"sample_id":"b","score":1.}])

    def test_multi_seed_bootstrap_keeps_unique_question_count(self):
        from vl_posttrain.evaluation.bootstrap import fixed_seed_mean_bootstrap
        baseline=[row("a","g1"),row("b","g2")]
        candidates={17:[row("a","g1",True),row("b","g2")],42:[row("a","g1",True),row("b","g2",True)],73:[row("a","g1"),row("b","g2")]}
        value=fixed_seed_mean_bootstrap(baseline,candidates,iterations=100)
        self.assertEqual(value["questions"],2)
        self.assertAlmostEqual(value["difference_right_minus_left"],.5)
        self.assertEqual(value["training_seeds"],[17,42,73])
