import tempfile
import unittest
from pathlib import Path

from vl_posttrain.agent.geometry import ViewRegistry, iou
from vl_posttrain.agent.protocol import parse_action
from vl_posttrain.config import Budgets
from vl_posttrain.io import canonical, file_sha256


class GeometryTests(unittest.TestCase):
    def test_native_pixel_example_and_rounding(self):
        registry = ViewRegistry("a", 1200, 1600)
        v1 = registry.crop("v0", [500, 600, 950, 850])
        self.assertEqual(v1.root_box, (600, 960, 1140, 1360))
        self.assertEqual(v1.natural_size, (540, 400))
        self.assertEqual(v1.to_root([333.33,400,814.81,650]), (779,1120,1040,1220))

    def test_nested_and_earlier_view_reference(self):
        registry = ViewRegistry("a", 1000, 1000)
        one = registry.crop("v0", [100,100,900,900])
        two = registry.crop(one.view_id, [250,250,750,750])
        self.assertEqual(two.root_box, (300,300,700,700))
        self.assertEqual(registry.get("v0").to_root([400,400,600,600]), (400,400,600,600))

    def test_frame_checks_reject_invalid_no_clamp(self):
        registry = ViewRegistry("a", 100, 100)
        for rect in ([0,0,1001,1000], [5,5,4,6], [0,0,float("nan"),5], [False,0,1,1], [0,0,0,1]):
            with self.subTest(rect=rect), self.assertRaises(ValueError):
                registry.crop("v0", rect)
        with self.assertRaises(ValueError):
            registry.crop("other-task:v1", [0,0,1000,1000])

    def test_iou_geometry(self):
        self.assertEqual(iou([0,0,10,10],[0,0,10,10]),1)
        self.assertAlmostEqual(iou([0,0,10,10],[5,0,15,10]),1/3)

    def test_strict_json_rejects_duplicate_extra_and_nonfinite(self):
        for text in ['{"tool":"crop","tool":"crop","view_id":"v0","box":[0,0,1,1]}',
                     '{"tool":"crop","view_id":"v0","box":[0,0,1,1],"reward":1}',
                     '{"tool":"crop","view_id":"v0","box":[0,0,NaN,1]}', '```json\n{}\n```']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_action(text)

    def test_budget_contract(self):
        self.assertEqual(Budgets().initial_visual_tokens + 3 * Budgets().crop_visual_tokens,1920)
        with self.assertRaises(ValueError):
            Budgets(new_visual_tokens=1000)


try:
    from PIL import Image
except ImportError:
    Image = None


@unittest.skipIf(Image is None, "真实 crop 测试需要 Pillow")
class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        temp_root = Path(__file__).resolve().parents[1] / "outputs" / "test-temp"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=temp_root)
        if not Path(self.tmp.name).resolve().is_relative_to(temp_root.resolve()):
            raise AssertionError("测试临时目录必须在工作区内")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        image = Image.new("RGB", (200,200))
        for x in range(200):
            for y in range(200):
                image.putpixel((x,y),(x,y,(x+y)%256))
        image.save(self.root / "image.png")
        self.task = {"task_id":"q", "image_id":"i", "question":"q", "image_path":"image.png", "image_sha256":file_sha256(self.root/"image.png")}

    def env(self, budgets=None):
        from vl_posttrain.agent.environment import CropEnvironment
        return CropEnvironment(self.task,self.root,self.root/"out",budgets)

    def test_pixels_are_real_and_nested_reads_from_root(self):
        env = self.env()
        env.step(canonical({"tool":"crop","view_id":"v0","box":[100,100,900,900]}),[10],"stop")
        env.step(canonical({"tool":"crop","view_id":"v1","box":[250,250,750,750]}),[11],"stop")
        expected = env.root_image.crop((60,60,140,140))
        self.assertEqual(env.images["v2"].tobytes(), expected.tobytes())
        self.assertEqual(env.messages[-1]["content"][1]["type"],"image")

    def test_actor_rejects_private_labels(self):
        self.task["answers"] = ["gold"]
        with self.assertRaises(ValueError):
            self.env()

    def test_invalid_calls_count_once_and_do_not_create_images(self):
        env = self.env()
        for text in ['not_json', '{"tool":"crop","view_id":"bad","box":[0,0,1000,1000]}']:
            env.step(text,[10],"stop")
        self.assertEqual(env.calls,2)
        self.assertEqual(len(env.images),1)

    def test_whole_episode_generation_budget_not_reset_each_turn(self):
        env = self.env(Budgets(generated_tokens=3))
        env.step("invalid",[1,2],"stop")
        env.step("invalid",[1,2],"stop")
        self.assertTrue(env.done)
        self.assertIn("over_budget",env.termination)
        self.assertEqual(env.generated,4)

    def test_actual_visual_cap_and_full_context_stop(self):
        env = self.env()
        self.assertEqual(env.admit_processor(6144,[32]),0)
        self.assertTrue(env.done)
        other = self.env()
        self.assertEqual(other.admit_processor(10,[769]),0)
        self.assertEqual(other.termination,"per_image_visual_budget")

    def test_final_can_reference_earlier_view(self):
        env = self.env()
        env.step(canonical({"tool":"crop","view_id":"v0","box":[100,100,900,900]}),[10],"stop")
        env.step(canonical({"final":{"answer":"x","evidence":{"view_id":"v0","box":[200,200,400,400]}}}),[11],"stop")
        self.assertEqual(env.final["root_box"],[40,40,80,80])
        with self.assertRaises(ValueError):
            env.step("{}",[12],"stop")

    def test_backend_resize_preserves_visual_ceiling(self):
        from vl_posttrain.agent.backend import resize_for_budget
        for size in [(1200,1600),(20000,80),(32,32),(200,700)]:
            image = resize_for_budget(Image.new("RGB",size),384)
            self.assertLessEqual(image.width // 32 * (image.height // 32),384)

    def test_alpha_handling_matches_canonical_pixel_fingerprint(self):
        from vl_posttrain.data.grouping import image_fingerprints
        Image.new("RGBA",(20,20),(0,0,0,0)).save(self.root/"transparent.png")
        self.task["image_path"]="transparent.png"
        self.task["image_sha256"]=file_sha256(self.root/"transparent.png")
        env=self.env()
        self.assertEqual(env.root_image.getpixel((0,0)),(255,255,255))
        import hashlib
        actual=hashlib.sha256(str(env.root_image.size).encode()+b"|RGB|"+env.root_image.tobytes()).hexdigest()
        try:
            import numpy
        except ImportError:
            self.skipTest("像素指纹对照需要NumPy")
        self.assertEqual(actual,image_fingerprints(str(self.root/"transparent.png"))["pixel_sha256"])
