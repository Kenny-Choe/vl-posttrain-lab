from __future__ import annotations

import math
from collections import defaultdict
from ..io import finite


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    p = (len(values) - 1) * q
    lo = math.floor(p)
    return values[lo] + (values[math.ceil(p)] - values[lo]) * (p - lo)


def aggregate(rows: list[dict]) -> dict:
    """分母为全部尝试，失败成本保留；0 成功时成本为 null 而非0。"""
    if not rows:
        raise ValueError("空结果")
    for row in rows:
        for key in ("joint_success", "answer_correct", "valid"):
            if not isinstance(row[key], bool):
                raise ValueError(f"{key} 必须为bool，不能将字符串false当成真")
        overlap = finite(row["iou"], "iou")
        if not 0 <= overlap <= 1:
            raise ValueError("IoU 必须在0到1之间")
        if row["joint_success"] != bool(row["answer_correct"] and row["valid"] and overlap >= .5):
            raise ValueError("joint与答案/IoU/合法性矛盾")
        for key in ("generated_tokens", "crop_calls"):
            if isinstance(row[key], bool) or not isinstance(row[key], int) or row[key] < 0:
                raise ValueError(f"{key} 必须非负整数")
        for key in ("latency_seconds", "model_seconds"):
            if row.get(key) is not None and finite(row[key], key) < 0:
                raise ValueError("成本不能为负")
    if len({r.get("model") for r in rows if r.get("model") is not None}) > 1:
        raise ValueError("请分别汇总每个模型，不混算不同模型结果")
    ids = [(r["run_id"], r["sample_id"], r.get("rollout_index", 0)) for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("重复结果身份；补评不可双计")
    n = len(rows)
    success = sum(bool(r["joint_success"]) for r in rows)
    latency = [r["latency_seconds"] for r in rows if r.get("latency_seconds") is not None]
    total_tokens = sum(r["generated_tokens"] for r in rows)
    total_seconds = sum(r["model_seconds"] for r in rows) if all(r.get("model_seconds") is not None for r in rows) else None
    repeat_known = all(r.get("repeated_crop") is not None for r in rows)
    kinds = sorted({r.get("provenance_kind", "unspecified_input_provenance") for r in rows})
    return {"attempts": n, "successes": success,
            "input_provenance_kinds": kinds,
            "answer_accuracy": sum(bool(r["answer_correct"]) for r in rows) / n,
            "grounding_accuracy_iou_0_5": sum(r["iou"] >= .5 for r in rows) / n,
            "joint_success_rate": success / n,
            "valid_completion_rate": sum(bool(r["valid"]) for r in rows) / n,
            "mean_iou": sum(r["iou"] for r in rows) / n,
            "mean_crop_calls": sum(r["crop_calls"] for r in rows) / n,
            "mean_crop_calls_success_only": sum(r["crop_calls"] for r in rows if r["joint_success"]) / success if success else None,
            "repeated_crop_task_rate": sum(bool(r["repeated_crop"]) for r in rows) / n if repeat_known else None,
            "generated_tokens_all_attempts": total_tokens,
            "generated_tokens_per_success": total_tokens / success if success else None,
            "model_seconds_all_attempts": total_seconds,
            "model_seconds_per_success": total_seconds / success if success and total_seconds is not None else None,
            "latency_p50_seconds": quantile(latency, .5), "latency_p95_seconds": quantile(latency, .95),
            "latency_measured_attempts": len(latency),
            "pass_at_1": success / n if all(r.get("rollout_index", 0) == 0 for r in rows) else None,
            "metric_version": "packaging-agent-metrics-v1"}


def pass_at_k(rows: list[dict], k: int) -> dict:
    """无放回抽取 k 个独立真实样本的 unbiased pass@k；不能复制 greedy 结果。"""
    if k < 1:
        raise ValueError("k 必须为正")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["run_id"], row["sample_id"])].append(row)
    estimates = []
    for key, values in groups.items():
        indices = [r["rollout_index"] for r in values]
        if len(indices) != len(set(indices)) or len(values) < k or not all(r.get("independent_sample") is True for r in values):
            raise ValueError(f"{key} 没有足够的独立采样")
        n, c = len(values), sum(bool(r["joint_success"]) for r in values)
        estimates.append(1.0 if n - c < k else 1 - math.comb(n - c, k) / math.comb(n, k))
    if not estimates:
        raise ValueError("无采样")
    return {"k": k, "questions": len(estimates), "pass_at_k": sum(estimates) / len(estimates)}
