import copy
import tempfile
import unittest
from pathlib import Path
from vl_posttrain.data.swift_bridge import swift_teacher_record


class SwiftBridgeTests(unittest.TestCase):
    def row(self):
        return {'messages':[{'role':'system','text':'system','images':[],'supervised':False},
            {'role':'user','text':'question','images':[],'supervised':False},
            {'role':'assistant','text':'wrong crop','images':[],'supervised':False},
            {'role':'tool','text':'actual observation','images':[],'supervised':False},
            {'role':'assistant','text':'answer','images':[],'supervised':True}],
            'private_supervision':{'secret_gold':'must not leak'}}

    def test_error_action_loss_false_tool_wrapped_and_private_gold_absent(self):
        result=swift_teacher_record(self.row(),'.')
        self.assertFalse(result['messages'][2]['loss'])
        self.assertTrue(result['messages'][4]['loss'])
        self.assertEqual(result['messages'][3],{'role':'user','content':'<tool_response>\nactual observation\n</tool_response>'})
        self.assertNotIn('private_supervision',result)
        self.assertNotIn('must not leak',repr(result))

    def test_non_boolean_or_supervised_observation_refused(self):
        row=self.row();row['messages'][2]['supervised']=0
        with self.assertRaises(ValueError):swift_teacher_record(row,'.')
        row=self.row();row['messages'][3]['supervised']=True
        with self.assertRaises(ValueError):swift_teacher_record(row,'.')

    def test_changed_image_not_silently_used(self):
        scratch=Path(__file__).resolve().parents[1]/'outputs/test_scratch';scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            root=Path(temporary).resolve();self.assertTrue(root.is_relative_to(scratch.resolve()))
            (root/'image.bin').write_bytes(b'changed image')
            row=self.row();row['messages'][1]['images']=[{'path':'image.bin','file_sha256':'0'*64,'view_id':'v0'}]
            with self.assertRaisesRegex(ValueError,'identity_changed'):swift_teacher_record(row,root)
