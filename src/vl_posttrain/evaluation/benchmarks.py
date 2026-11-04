"""透明的 CPU 评分参考。官方榜单仍须固定版本官方 scorer 和完整 split。"""
from __future__ import annotations

import re


def numeric_value(text: str) -> float | None:
    value = text.strip().replace(",", "")
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)%?", value):
        return None
    return float(value[:-1]) / 100 if value.endswith("%") else float(value)


def relaxed_numeric(prediction: str, references: list[str], tolerance: float = .05) -> float:
    for answer in references:
        p, r = numeric_value(prediction), numeric_value(answer)
        if p is not None and r is not None:
            if (abs(p - r) <= abs(r) * tolerance) if r != 0 else p == 0:
                return 1.
        elif prediction.strip().casefold() == answer.strip().casefold():
            return 1.
    return 0.


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def anls(prediction: str, references: list[str]) -> float:
    p = prediction.casefold().strip()
    result = 0.
    for answer in references:
        r = answer.casefold().strip()
        denominator = max(len(p), len(r))
        normalized = edit_distance(p, r) / denominator if denominator else 0.
        result = max(result, 1 - normalized if normalized < .5 else 0.)
    return result


def retention_delta(baseline: list[dict], candidate: list[dict]) -> dict:
    a = {r["sample_id"]: r["score"] for r in baseline}
    b = {r["sample_id"]: r["score"] for r in candidate}
    if len(a) != len(baseline) or len(b) != len(candidate) or set(a) != set(b) or not a:
        raise ValueError("OCR 保留比较需要同一完整题集和评分口径")
    return {"questions": len(a), "baseline": sum(a.values()) / len(a), "candidate": sum(b.values()) / len(b),
            "delta_percentage_points": 100 * (sum(b.values()) - sum(a.values())) / len(a)}
