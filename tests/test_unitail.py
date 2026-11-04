import copy
import unittest
from vl_posttrain.data.unitail import gallery_regions
from vl_posttrain.data.sources import unitail_reviewed_region


class UnitailTests(unittest.TestCase):
    def base(self):
        return {'images':[{'id':7,'file_name':'imgs/7.jpg','width':100,'height':80}],
                'annotations':[{'id':9,'image_id':7,'text-words':'500g',
                                'segmentation':[[10,20,40,20,40,30,10,30]]}]}

    def test_actual_unitail_star_is_unreadable_not_a_gold_word(self):
        coco=self.base();coco['annotations'][0]['text-words']=' * '
        result=gallery_regions(coco,'actual-native-version')
        self.assertEqual(result['summary']['QA_created'],0)
        self.assertEqual(result['candidates'],[])
        self.assertEqual(result['quarantine'][0]['reason'],'native_unreadable_marker')
        export={'image_id':'7','image_path':'imgs/7.jpg','width':100,'height':80,'region_id':'9',
                'polygon':[[10,20],[40,20],[40,30],[10,30]],'transcription':'*',
                'component':'gallery_train','review_receipt':'actual-review'}
        with self.assertRaises(ValueError):unitail_reviewed_region(export,'revision')

    def test_polygon_not_product_bbox_and_semantics_still_pending(self):
        coco=self.base();coco['annotations'][0]['bbox']=[0,0,100,80]
        row=gallery_regions(coco,'revision')['candidates'][0]
        self.assertEqual(row['region_box_xyxy'],[10,20,40,30])
        self.assertFalse(row['question_created'])
        self.assertEqual(row['answer_and_semantic_review'],'pending')
        self.assertNotIn('answers',row)

    def test_outside_polygon_is_quarantined_not_clamped(self):
        coco=self.base();coco['annotations'][0]['segmentation'][0][0]=-1
        result=gallery_regions(coco,'revision')
        self.assertEqual(result['candidates'],[])
        self.assertIn('geometry_needs_review',result['quarantine'][0]['reason'])

    def test_duplicate_and_missing_native_ids_rejected(self):
        coco=self.base();coco['annotations']*=2
        with self.assertRaises(ValueError):gallery_regions(coco,'revision')
        coco=self.base();coco['annotations'][0]['image_id']=8
        with self.assertRaises(ValueError):gallery_regions(coco,'revision')
