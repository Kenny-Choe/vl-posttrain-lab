import unittest
from vl_posttrain.data.curriculum_native import assistant_spans


class NativeMaskContractTests(unittest.TestCase):
    def test_headers_and_visual_context_are_excluded_but_real_end_newline_is_included(self):
        # Explicit token fixture, not model output. Tool/user image placeholders
        # precede each assistant header; only supervised assistant spans count.
        ids=[902,5,900,2,3,20,21,901,3,902,6,900,2,3,30,901,3,902,7,900,2,3,40,41,901,3]
        spans,mask=assistant_spans(ids,[900,2,3],901,[3],[True,False,True])
        self.assertEqual([s['body_start'] for s in spans],[5,14,22])
        self.assertTrue(mask[5]);self.assertTrue(mask[7]);self.assertTrue(mask[8])
        self.assertFalse(any(mask[11:17]))
        self.assertFalse(any(use for token,use in zip(ids,mask) if token in {900,902}))
        self.assertEqual(sum(mask),8)

    def test_missing_end_or_template_newline_is_not_silently_accepted(self):
        for ids in [[900,2,3,20,3],[900,2,3,20,901,8]]:
            with self.assertRaises(ValueError):assistant_spans(ids,[900,2,3],901,[3],[True])

    def test_extra_assistant_header_or_missing_flag_is_a_hard_mismatch(self):
        ids=[900,2,3,20,901,3,900,2,3,21,901,3]
        with self.assertRaises(ValueError):assistant_spans(ids,[900,2,3],901,[3],[True])

    def test_numeric_supervision_flag_is_not_interpreted_as_verified_boolean(self):
        with self.assertRaises(ValueError):assistant_spans([900,2,3,20,901,3],[900,2,3],901,[3],[1])
