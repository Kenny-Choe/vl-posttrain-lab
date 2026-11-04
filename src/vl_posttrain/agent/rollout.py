from __future__ import annotations

import time
from pathlib import Path
from ..io import identity, read_json, write_json
from ..data.schema import QARecord
from ..evaluation.scoring import answer_matches, reward, repeated_crop
from .environment import CropEnvironment


def run_episode(record: QARecord, backend, image_root, output, run_id: str, protocol_sha256: str,
                budgets=None, max_wall_seconds: float = 600) -> dict:
    record.validate(require_evidence=True)
    destination = Path(output)
    if (destination / "receipt.json").exists():
        old = read_json(destination / "receipt.json")
        if old["sample_id"] != record.sample_id or old["run_id"] != run_id or old["protocol_sha256"] != protocol_sha256 or old["model_identity"] != backend.identity or old["record_sha256"] != identity(record.to_dict()):
            raise ValueError("输出目录身份冲突；不得复用其他模型/协议结果")
        return old
    lock = destination / "episode.lock"
    destination.mkdir(parents=True, exist_ok=True)
    # 独占文件，防止两个控制器生成同一题。失败锁保留，先检查而非自动重启。
    with lock.open("x", encoding="utf-8") as f:
        f.write(identity([run_id, record.sample_id, backend.identity, protocol_sha256]))
    start = time.perf_counter()
    environment = None
    try:
        environment = CropEnvironment(record.actor_input(), image_root, destination, budgets)
        while not environment.done:
            if time.perf_counter() - start > max_wall_seconds:
                environment.stop("episode_wallclock_budget")
                break
            prep = backend.prepare(environment.messages)
            room = environment.admit_processor(prep["context_tokens"], prep["visual_counts"])
            if environment.events and environment.events[-1]["type"] == "prefill":
                environment.events[-1]["processor_receipt"] = prep
            write_json(destination / "progress.json", {"receipt": environment.receipt(), "last_processor": prep})
            if not room:
                break
            generated = backend.generate(room)
            environment.step(generated["text"], generated["token_ids"], generated["finish_reason"], generated["latency_seconds"])
            environment.events[-1]["generation_receipt"] = {k: v for k, v in generated.items() if k not in {"text", "token_ids"}}
            if time.perf_counter() - start > max_wall_seconds:
                environment.final = None
                environment.stop("episode_wallclock_budget")
        receipt = environment.receipt()
        final = receipt.get("final")
        valid = receipt["status"] == "final"
        scored = reward(answer_matches(final["answer"], record.answers) if final else False,
                        final["root_box"] if final else None, record.evidence_box_xyxy,
                        record.image_width, record.image_height, valid)
        crops = [e["crop"] for e in receipt["events"] if e.get("crop")]
        repeat = None
        for crop in crops:
            count = environment.seen_visual.get(crop["view_id"])
            crop["visual_tokens"] = count
            if count:
                rect = crop["root_box"]
                crop["effective_resolution"] = (count / ((rect[2] - rect[0]) * (rect[3] - rect[1]))) ** .5
        if all(c.get("effective_resolution") for c in crops):
            repeat = repeated_crop(crops)
        seconds = time.perf_counter() - start
        result = {**receipt, **scored, "sample_id": record.sample_id, "group_id": record.image_group_id,
                  "run_id": run_id, "protocol_sha256": protocol_sha256, "model_identity": backend.identity,
                  "record_sha256": identity(record.to_dict()),
                  "provenance_kind": "observed_existing_model_inference", "parameter_updates": 0,
                  "model_seconds": seconds, "latency_seconds": seconds, "repeated_crop": repeat}
        write_json(destination / "receipt.json", result)
        lock.unlink()
        return result
    except Exception as exc:
        write_json(destination / "failure.json", {"error_type": type(exc).__name__, "error": str(exc),
                   "partial": environment.receipt() if environment is not None else None, "elapsed_seconds": time.perf_counter() - start,
                   "training": False, "automatic_retry": False})
        raise
