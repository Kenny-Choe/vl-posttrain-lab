import json
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
from vl_posttrain.resources import linux_memory_snapshot,model_admission,GiB


class ResourceTests(unittest.TestCase):
    def setUp(self):
        base=Path(__file__).resolve().parents[1]/'outputs/test-temp';base.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base);self.addCleanup(self.temp.cleanup);self.path=Path(self.temp.name)
        assert self.path.resolve().is_relative_to(base.resolve())
    def model(self):
        root=self.path/'model';root.mkdir()
        (root/'config.json').write_text(json.dumps({'torch_dtype':'bfloat16'}))
        (root/'one.safetensors').write_bytes(b'fixture_not_real_weights')
        (root/'model.safetensors.index.json').write_text(json.dumps({'metadata':{'total_size':16*GiB},'weight_map':{'p':'one.safetensors'}}))
        return root
    def test_host_memory_does_not_override_container_remaining_memory(self):
        proc=self.path/'proc';group=self.path/'cgroup';proc.mkdir();group.mkdir()
        (proc/'meminfo').write_text('MemTotal: 528000000 kB\nMemAvailable: 400000000 kB\n')
        (group/'memory.max').write_text(str(4*GiB));(group/'memory.current').write_text(str(3*GiB));(group/'cpu.max').write_text('200000 100000')
        snapshot=linux_memory_snapshot(proc,group)
        self.assertEqual(snapshot['effective_available_bytes'],GiB)
        self.assertEqual(snapshot['cpu_quota_cores'],2)
        self.assertFalse(model_admission(self.model(),'cpu',snapshot)['admitted'])
    def test_CPU_admission_accounts_for_float32_expansion(self):
        result=model_admission(self.model(),'cpu',{'effective_available_bytes':33*GiB})
        self.assertEqual(result['estimated_required_device_bytes'],34*GiB)
        self.assertFalse(result['admitted'])
    def test_actual_qwen_nested_dtype_metadata_is_used_before_weight_load(self):
        model=self.model()
        (model/'config.json').write_text(json.dumps({'model_type':'qwen3_vl','text_config':{'dtype':'bfloat16'}}))
        result=model_admission(model,'cpu',{'effective_available_bytes':GiB})
        self.assertEqual(result['weight_inventory']['dtype_metadata_location'],'text_config.dtype')
        self.assertEqual(result['weight_inventory']['stored_dtype'],'bfloat16')
        self.assertEqual(result['estimated_required_device_bytes'],34*GiB)
        self.assertEqual(result['reasons'],['effective_CPU_memory_below_estimated_weights_plus_headroom'])
    def test_missing_GPU_rejected_before_importing_or_loading_model(self):
        result=model_admission(self.model(),'cuda:0',{'effective_available_bytes':8*GiB},[])
        self.assertFalse(result['admitted']);self.assertIn('requested_GPU_not_visible',result['reasons'])
        self.assertFalse(result['model_weights_loaded'])
    def test_positive_GPU_memory_check_is_not_peak_guarantee(self):
        result=model_admission(self.model(),'cuda:0',{'effective_available_bytes':8*GiB},[{'index':0,'free_bytes':24*GiB}])
        self.assertTrue(result['admitted']);self.assertIn('not_peak_memory_guarantee',result['estimate_scope'])
    def test_missing_shard_does_not_trigger_download(self):
        model=self.model();(model/'one.safetensors').unlink()
        with self.assertRaises(ValueError):model_admission(model,'cpu',{'effective_available_bytes':64*GiB})
    def test_visible_GPU_is_logical_not_physical_ordinal(self):
        from vl_posttrain.resources import gpu_snapshot
        process=SimpleNamespace(returncode=0,stdout='0, GPU-aaa, 24576, 512\n1, GPU-bbb, 24576, 24000\n')
        with patch('vl_posttrain.resources.subprocess.run',return_value=process),patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':'1'}):
            values=gpu_snapshot()
        self.assertEqual(values[0]['index'],0);self.assertEqual(values[0]['physical_index'],1)
        self.assertEqual(values[0]['free_bytes'],24000*1024**2)
