from __future__ import annotations

from dataclasses import dataclass
from ..io import identity


@dataclass
class FrozenRolloutBatch:
    behavior_checkpoint: str
    reference_checkpoint: str
    rows: list[dict]
    max_reuse: int = 4
    completed_updates: int = 0

    def __post_init__(self):
        if not self.rows or not self.behavior_checkpoint or not self.reference_checkpoint:
            raise ValueError("完整 rollout/old/ref 身份必需")
        required = {"sample_id", "token_ids", "old_logps", "ref_logps", "advantage", "action_mask", "prefix_sha256"}
        for r in self.rows:
            if not required <= set(r):
                raise ValueError("缺少多轮回放字段")
            if len({len(r[key]) for key in ("token_ids", "old_logps", "ref_logps", "action_mask")}) != 1:
                raise ValueError("逐位置回放长度不一致")
        self._locked = self.fingerprint()

    def fingerprint(self):
        return identity([self.behavior_checkpoint, self.reference_checkpoint, self.rows])

    def next_reuse(self) -> dict:
        if self.fingerprint() != self._locked:
            raise ValueError("复用期间 old/ref/轨迹/优势被修改")
        if self.completed_updates >= self.max_reuse:
            raise ValueError("本批已消费，须生成新批，不能冒充新轨迹")
        self.completed_updates += 1
        return {"reuse": self.completed_updates, "batch_sha256": self._locked,
                "behavior_checkpoint": self.behavior_checkpoint, "new_generations": 0}


def verify_checkpoint_sidecar(sidecar: dict, previous_cost: dict | None = None) -> dict:
    required = {"policy_sha256", "ref_sha256", "optimizer_sha256", "scheduler_sha256", "rng_sha256",
                "logical_batch_index", "reuse_index", "schedule_sha256", "cost"}
    if not required <= set(sidecar):
        raise ValueError("恢复元数据不完整")
    if sidecar["reuse_index"] != 0 and not sidecar.get("frozen_rollout_sha256"):
        raise ValueError("批次中间恢复需要保存原 old/ref/轨迹")
    for key, value in (previous_cost or {}).items():
        if sidecar["cost"].get(key, -1) < value:
            raise ValueError("累计成本不可回退")
    return {"contract_checked": True, "native_8b_resume_tested": False}
