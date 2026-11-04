from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from ..agent.geometry import iou, area


def normalize_answer(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def answer_matches(answer: str, references: list[str], aliases: dict[str, str] | None = None) -> bool:
    aliases = {normalize_answer(k): normalize_answer(v) for k, v in (aliases or {}).items()}
    def norm(text):
        value = normalize_answer(text)
        return aliases.get(value, value)
    return norm(answer) in {norm(ref) for ref in references}


def reward(answer_correct: bool, predicted_box, gold_box, width: int, height: int,
           valid: bool, variant: str = "target", distance_weight: float = 0.05) -> dict:
    overlap = iou(predicted_box, gold_box) if predicted_box else 0.0
    distance = 0.0
    if predicted_box:
        cp = ((predicted_box[0] + predicted_box[2]) / 2, (predicted_box[1] + predicted_box[3]) / 2)
        cg = ((gold_box[0] + gold_box[2]) / 2, (gold_box[1] + gold_box[3]) / 2)
        distance = max(0.0, 1 - math.dist(cp, cg) / math.hypot(width, height))
    a = float(answer_correct)
    if variant == "legacy_A":
        raw = a
    elif variant == "legacy_B":
        raw = a * (1 + 0.2 * overlap)
    elif variant == "answer_only":
        raw = a
    elif variant == "no_distance":
        raw = a + 0.5 * a * overlap
    elif variant == "target":
        raw = a + 0.5 * a * overlap + distance_weight * distance
    else:
        raise ValueError("未知奖励版本")
    value = float(valid) * raw
    return {"reward": value, "answer_correct": bool(answer_correct), "iou": overlap,
            "distance_feedback": distance, "valid": bool(valid),
            "joint_success": bool(valid and answer_correct and overlap >= 0.5), "reward_version": variant}


def repeated_crop(crops: list[dict]) -> bool:
    for i, left in enumerate(crops):
        for right in crops[i + 1:]:
            a, b = left["root_box"], right["root_box"]
            ratio = area(a) / area(b) if area(b) else math.inf
            ra, rb = left.get("effective_resolution"), right.get("effective_resolution")
            if ra is None or rb is None:
                raise ValueError("新协议重复率需要实际有效分辨率；不能只用框 IoU 替代")
            improved = max(ra, rb) / min(ra, rb) > 1.1 if min(ra, rb) > 0 else True
            if iou(a, b) >= 0.9 and 0.8 <= ratio <= 1.25 and not improved:
                return True
    return False


def token_f1(prediction: str, reference: str) -> float:
    from collections import Counter
    def tokens(text):
        return re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+|[^\w\s]", unicodedata.normalize("NFKC", text).casefold())
    p, r = tokens(prediction), tokens(reference)
    if not p or not r:
        return float(p == r)
    common = sum((Counter(p) & Counter(r)).values())
    return 2 * common / (len(p) + len(r))
