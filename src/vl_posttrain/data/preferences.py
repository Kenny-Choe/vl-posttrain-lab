from __future__ import annotations

from ..io import identity


def build_preference(row: dict) -> dict:
    if row.get("project_split") != "train" or row.get("native_gold_match") != "question_choices_image_checked":
        raise ValueError("偏好对须来自有原生答案核验的训练样本")
    candidates = row["candidates"]
    chosen = [c for c in candidates if c.get("verified_correct") is True and c.get("valid") is True]
    rejected = [c for c in candidates if c.get("verified_correct") is False]
    if not chosen or not rejected:
        raise ValueError("没有可验证偏好差异；同分/不确定不强行配对")
    good = min(chosen, key=lambda c: c["candidate_id"])
    bad = min(rejected, key=lambda c: c["candidate_id"])
    if good["text"] == bad["text"]:
        raise ValueError("chosen 与 rejected 相同")
    return {"pair_id": identity([row["sample_id"], good["candidate_id"], bad["candidate_id"]]),
            "sample_id": row["sample_id"], "image_group_id": row["image_group_id"],
            "chosen": good["text"], "rejected": bad["text"],
            "chosen_id": good["candidate_id"], "rejected_id": bad["candidate_id"],
            "project_split": "train", "native_gold_match": row["native_gold_match"]}
