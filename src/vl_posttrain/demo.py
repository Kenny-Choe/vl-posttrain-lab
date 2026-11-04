"""完全隔离的合成 CPU 工程示例，不生成任何实测模型成绩。"""
from __future__ import annotations

from pathlib import Path
from .agent.environment import CropEnvironment
from .io import canonical, file_sha256, write_json, write_jsonl
from .evaluation.scoring import reward
from .algorithms.objectives import gradient_probe, advantages


def run_demo(output: str | Path) -> dict:
    from PIL import Image, ImageDraw
    dest = Path(output)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "synthetic_package.png"
    image = Image.new("RGB", (1200, 1600), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((780, 1120, 1039, 1219), outline="black", width=3)
    draw.text((800, 1150), "500g", fill="black")
    image.save(path)
    env = CropEnvironment({"task_id": "synthetic-fixture-001", "question": "What is the net weight in the marked sample?",
                           "image_id": "synthetic-package", "image_path": path.name,
                           "image_sha256": file_sha256(path)}, dest, dest / "views")
    # 这些 token ID 是明示的 fixture 标识，既非 tokenizer 输出，也非模型生成。
    env.admit_processor(100, [4])
    env.step(canonical({"tool": "crop", "view_id": "v0", "box": [500, 600, 950, 850]}), [900001], "fixture_stop")
    env.admit_processor(200, [4, 4])
    final = {"final": {"answer": "500g", "evidence": {"view_id": "v1", "box": [333.33, 400, 814.81, 650]}}}
    env.step(canonical(final), [900002], "fixture_stop")
    receipt = env.receipt()
    score = reward(True, receipt["final"]["root_box"], [780,1120,1040,1220], 1200,1600,True)
    result = {"provenance_kind": "synthetic_fixture_not_model_inference", "model_loaded": False,
              "native_processor_executed": False, "training": False,
              "fixture_processor_counts": True, "fixture_token_ids": True,
              "scope": "geometry_protocol_budget_and_reward_integration_only",
              "root_evidence_actual": receipt["final"]["root_box"], "score": score,
              "gradient_probe": gradient_probe(),
              "all_wrong_distance_advantages": advantages([.01] * 4 + [.04] * 4)}
    write_json(dest / "demo_receipt.json", {**result, "environment": receipt})
    write_jsonl(dest / "metrics_fixture.jsonl", [{"run_id":"synthetic-cpu-demo", "sample_id":"synthetic-fixture-001", "group_id":"synthetic-package",
                 "answer_correct":score["answer_correct"], "joint_success":score["joint_success"], "iou":score["iou"],
                 "valid":True, "crop_calls":receipt["crop_calls"], "generated_tokens":receipt["generated_tokens"],
                 "model_seconds":None, "latency_seconds":None, "repeated_crop":False,
                 "provenance_kind":"synthetic_fixture_not_model_inference"}])
    return result
