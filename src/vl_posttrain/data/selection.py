from __future__ import annotations

import math
from collections import defaultdict
from ..io import identity, finite


def largest_remainder(sizes: dict[str, int], total: int) -> dict[str, int]:
    if total < 0 or total > sum(sizes.values()) or any(n < 0 for n in sizes.values()):
        raise ValueError("候选不足或配额为负")
    if total == 0:
        return {key: 0 for key in sizes}
    denominator = sum(sizes.values())
    quotas = {key: total * n // denominator for key, n in sizes.items()}
    order = sorted(sizes, key=lambda key: (-(total * sizes[key] % denominator), key))
    for key in order[:total - sum(quotas.values())]:
        quotas[key] += 1
    return quotas


def select(rows: list[dict], total: int, arm: str, seed: int = 42) -> dict:
    import numpy as np
    if arm not in {"filtered_random", "visual_mixture"}:
        raise ValueError("未知抽样臂")
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("候选 sample_id 重复")
    layers = defaultdict(list)
    for row in rows:
        if row.get("project_split") != "train" or row.get("quality_flags") or row.get("protected") or str(row.get("source_split", "train")).lower() in {"test", "val", "validation", "testmini"}:
            raise ValueError("候选必须通过训练隔离与质量检查")
        finite(row["visual_dependency"], "visual_dependency")
        layers[f'{row["source"]}::{row["task_type"]}'].append(row)
    quotas = largest_remainder({key: len(value) for key, value in layers.items()}, total)
    high = {key: n // 2 for key, n in quotas.items()}
    for key in sorted(key for key, n in quotas.items() if n % 2)[:total // 2 - sum(high.values())]:
        high[key] += 1
    rng = np.random.Generator(np.random.PCG64(seed))
    chosen = []
    for key in sorted(layers):
        candidates = sorted(layers[key], key=lambda r: r["sample_id"])
        if arm == "visual_mixture":
            ranked = sorted(candidates, key=lambda r: (-r["visual_dependency"], r["sample_id"]))
            top = ranked[:high[key]]
            candidates = sorted(ranked[high[key]:], key=lambda r: r["sample_id"])
        else:
            top = []
        take = quotas[key] - len(top)
        chosen.extend(top + [candidates[int(i)] for i in rng.choice(len(candidates), take, replace=False)])
    if len(chosen) != total or len({r["sample_id"] for r in chosen}) != total:
        raise AssertionError("抽样配额/身份不一致")
    return {"rows": chosen, "arm": arm, "quotas": quotas, "high_quotas": high if arm == "visual_mixture" else {},
            "input_sha256": identity(sorted(rows, key=lambda r: r["sample_id"])), "seed": seed,
            "rng": "numpy.PCG64", "numpy_version": np.__version__, "count": total}
