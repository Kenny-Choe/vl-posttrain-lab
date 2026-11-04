import contextlib
import io
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from vl_posttrain.cli import main
from vl_posttrain.io import read_json

ROOT=Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def call(self,*argv):
        stream=io.StringIO()
        with contextlib.redirect_stdout(stream):
            main(list(argv))
        return json.loads(stream.getvalue())

    def test_plan_and_doctor_never_load_models(self):
        plan=self.call("plan","--config",str(ROOT/"configs/route.target.json"))
        self.assertFalse(plan["training_started"])
        self.assertEqual(plan["curriculum_total_updates"],500)
        self.assertFalse(self.call("doctor")["models_loaded"])

    def test_no_training_command_exists(self):
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as result:
            main(["train"])
        self.assertNotEqual(result.exception.code,0)

    def test_resource_rejection_has_nonzero_exit_and_keeps_receipt(self):
        base=ROOT/'outputs/test-temp';base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as directory:
            model=Path(directory)/'model';model.mkdir()
            (model/'config.json').write_text(json.dumps({'text_config':{'dtype':'bfloat16'}}))
            (model/'one.safetensors').write_bytes(b'not_model_weights')
            (model/'model.safetensors.index.json').write_text(json.dumps({'metadata':{'total_size':16*1024**3},'weight_map':{'w':'one.safetensors'}}))
            receipt=Path(directory)/'admission.json'
            with patch('vl_posttrain.resources.memory_snapshot',return_value={'effective_available_bytes':1024**3}),contextlib.redirect_stdout(io.StringIO()),self.assertRaises(SystemExit) as exit:
                main(['model-preflight','--model',str(model),'--device','cpu','--output',str(receipt)])
            self.assertEqual(exit.exception.code,2)
            self.assertFalse(read_json(receipt)['admitted'])
            self.assertFalse(read_json(receipt)['model_weights_loaded'])

    def test_dataset_demo_is_not_reported_as_real_12k_data(self):
        base=ROOT/'outputs/test-temp';base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as directory:
            result=self.call('dataset-demo','--output',directory)
            self.assertEqual(result['actual_fixture_QA'],96)
            self.assertFalse(result['target_12k_dataset_counts_met'])
            self.assertFalse(result['reviewed_real_data_release_passed'])

    def test_demo_and_metrics_keep_fixture_identity(self):
        try:
            import PIL
        except ImportError:
            self.skipTest("Pillow optional")
        temp_root=ROOT/"outputs/test-temp"
        temp_root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            path=Path(directory)
            self.assertTrue(path.resolve().is_relative_to(temp_root.resolve()))
            demo=self.call("demo","--output",str(path))
            self.assertFalse(demo["model_loaded"])
            self.assertEqual(demo["root_evidence_actual"],[779,1120,1040,1220])
            metrics=self.call("metrics","--input",str(path/"metrics_fixture.jsonl"),"--output",str(path/"metrics.json"))
            self.assertEqual(metrics["input_provenance_kinds"],["synthetic_fixture_not_model_inference"])
            self.assertIsNone(metrics["model_seconds_all_attempts"])
