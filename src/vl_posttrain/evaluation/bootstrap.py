from __future__ import annotations

import random
from collections import defaultdict
from .metrics import quantile


def paired_group_bootstrap(left: list[dict], right: list[dict], metric: str = "joint_success",
                           iterations: int = 2000, seed: int = 42) -> dict:
    """配对 cluster bootstrap：重采样商品/图像组，保留组内所有 QA。"""
    if iterations < 1:
        raise ValueError("bootstrap 次数必须为正")
    def index(rows):
        result = {}
        for r in rows:
            key = r["sample_id"]
            if key in result:
                raise ValueError("每模型每题只能一个结果；训练种子须单独对照")
            result[key] = r
        return result
    a, b = index(left), index(right)
    if set(a) != set(b) or not a:
        raise ValueError("比较必须覆盖同一完整题集，不得悄悄取交集")
    groups = defaultdict(list)
    for key in sorted(a):
        if a[key]["group_id"] != b[key]["group_id"]:
            raise ValueError("组身份不一致")
        groups[a[key]["group_id"]].append(float(b[key][metric]) - float(a[key][metric]))
    sums = {key: sum(v) for key, v in groups.items()}
    ids, rng, replicates = sorted(groups), random.Random(seed), []
    for _ in range(iterations):
        selected = rng.choices(ids, k=len(ids))
        replicates.append(sum(sums[key] for key in selected) / sum(len(groups[key]) for key in selected))
    return {"difference_right_minus_left": sum(sums.values()) / len(a),
            "ci95": [quantile(replicates, .025), quantile(replicates, .975)],
            "questions": len(a), "groups": len(ids), "iterations": iterations, "seed": seed,
            "scope": "question_group_sampling_uncertainty_not_training_seed_uncertainty"}


def fixed_seed_mean_bootstrap(baseline: list[dict], candidates: dict[int, list[dict]], metric: str="joint_success",
                              iterations: int=2000, seed: int=42) -> dict:
    """同一组重采样应用于各固定种子，再取种子均值；不把三seed当独立6K题。"""
    if len(candidates)<2:
        raise ValueError("需多个已实际执行的独立训练种子")
    baseline_index={r["sample_id"]:r for r in baseline}
    if len(baseline_index)!=len(baseline) or not baseline:
        raise ValueError("baseline身份不唯一或为空")
    seed_effects={}
    indices={}
    for training_seed,rows in sorted(candidates.items()):
        indices[training_seed]={r["sample_id"]:r for r in rows}
        seed_effects[str(training_seed)]=paired_group_bootstrap(baseline,rows,metric,iterations=1,seed=seed)["difference_right_minus_left"]
    # 逐题先对固定种子求平均，与共享重采样中逐seed再平均代数等价。
    grouped_candidates=[]
    baseline_float=[]
    for key,r in sorted(baseline_index.items()):
        grouped_candidates.append({"sample_id":key,"group_id":r["group_id"],
                                   metric:sum(float(indices[s][key][metric]) for s in candidates)/len(candidates)})
        baseline_float.append({"sample_id":key,"group_id":r["group_id"],metric:float(r[metric])})
    result=paired_group_bootstrap(baseline_float,grouped_candidates,metric,iterations,seed)
    return {**result,"training_seeds":sorted(candidates),"per_seed_point_differences":seed_effects,
            "scope":"question_group_uncertainty_of_fixed_training_seed_mean_not_training_seed_population_CI"}
