"""路线身份、参数一致性与只读执行计划；不提供训练启动命令。"""
from __future__ import annotations

import math
from dataclasses import dataclass
from .io import finite, identity


@dataclass(frozen=True)
class Budgets:
    crop_calls: int = 3
    generated_tokens: int = 2048
    initial_visual_tokens: int = 768
    crop_visual_tokens: int = 384
    new_visual_tokens: int = 2048
    context_tokens: int = 6144

    def __post_init__(self):
        for key, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == "crop_calls" else 1):
                raise ValueError(f"非法预算 {key}")
        if self.initial_visual_tokens + self.crop_calls * self.crop_visual_tokens > self.new_visual_tokens:
            raise ValueError("原图加全部 crop 的视觉预算超过总额")


def validate_route(route: dict) -> dict:
    if route.get("execution_mode") != "existing_models_only_no_training":
        raise ValueError("本次路线必须保持不训练")
    p1, p2 = route["project1"], route["project2"]
    if p2["parent"] != "C0" or p1["dpo_parent"] != "C2" or p1["grpo_parent"] != "C2":
        raise ValueError("模型谱系不符合独立分支")
    if sum(p2["split_counts"].values()) != p2["qa_target"]:
        raise ValueError("QA 划分总量不一致")
    if p2["curriculum"]["domain_localization"] + p2["curriculum"]["domain_search_stop"] != p2["split_counts"]["train"]:
        raise ValueError("领域课程规模不一致")
    Budgets(**p2["budgets"])
    rl = p2["rl"]
    batch = rl["world_size"] * rl["micro_batch"] * rl["accumulation"]
    if batch != rl["questions_per_batch"] * rl["group_size"]:
        raise ValueError("完成轨迹 global batch 与题数/G 不一致")
    if rl["fresh_batches"] * rl["reuse"] != rl["updates"]:
        raise ValueError("新 rollout 批次与有效更新数量不一致")
    if rl["questions_per_batch"] * rl["fresh_batches"] > p2["split_counts"]["train"]:
        raise ValueError("当前不放回日程超出训练池")
    if set(rl["seeds"]) != {17, 42, 73} or len(rl["seeds"]) != 3:
        raise ValueError("目标为三个独立训练种子")
    for key in ("lr", "beta_kl", "tau_pos", "tau_neg"):
        if finite(rl[key], key) <= 0:
            raise ValueError(f"{key} 必须大于0")
    if rl["sampling"] != {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0}:
        raise ValueError("目标公平采样分布已改变")
    return {"status": "configuration_checked_not_executed", "route_sha256": identity(route),
            "global_trajectories": batch,
            "new_trajectories_per_rl_run": rl["fresh_batches"] * batch,
            "question_presentations_per_rl_run": rl["fresh_batches"] * rl["questions_per_batch"],
            "updates_per_rl_run": rl["updates"],
            "training_started": False}


def training_plan(route: dict) -> dict:
    result = validate_route(route)
    c = route["project2"]["curriculum"]
    stage = c["domain_localization"] + c["general_replay"] // 2
    global_batch = c["world_size"] * c["micro_batch"] * c["accumulation"]
    updates = math.ceil(stage / global_batch) * c["epochs_per_stage"]
    result.update({"p1_general_target": route["project1"]["general_sft"],
                   "p1_reasoning_target": route["project1"]["reasoning_sft"],
                   "p1_dpo_pairs_target": route["project1"]["dpo_pairs"],
                   "curriculum_stage_examples": stage, "curriculum_global_batch": global_batch,
                   "curriculum_stage_updates": updates, "curriculum_total_updates": 2 * updates,
                   "weights_retrained": False, "scope": "target_plan_only"})
    return result
