from __future__ import annotations

import math
from ..io import finite, identity


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    z = math.exp(value)
    return z / (1 + z)


def advantages(rewards: list[float], epsilon: float = 1e-8) -> list[float]:
    if not rewards or epsilon <= 0:
        raise ValueError("空奖励/非法 epsilon")
    rewards = [finite(v, "reward") for v in rewards]
    mean = math.fsum(rewards) / len(rewards)
    std = math.sqrt(math.fsum((v - mean) ** 2 for v in rewards) / len(rewards))
    return [(v - mean) / (std + epsilon) for v in rewards]


def surrogate(logp: float, old_logp: float, advantage: float, method: str = "sapo",
              epsilon: float = .2, tau_pos: float = 1., tau_neg: float = 1.05) -> dict:
    for v in (logp, old_logp, advantage, epsilon, tau_pos, tau_neg):
        finite(v)
    if tau_pos <= 0 or tau_neg <= 0 or not 0 < epsilon < 1:
        raise ValueError("非法温度或 clip 参数")
    ratio = math.exp(logp - old_logp)
    if method == "sapo":
        tau = tau_pos if advantage > 0 else tau_neg
        p = sigmoid(tau * (ratio - 1))
        loss = -advantage * 4 / tau * p
        weight = 4 * p * (1 - p)
        grad = -advantage * ratio * weight
        effective_clip = False
    elif method == "grpo":
        clipped = min(1 + epsilon, max(1 - epsilon, ratio))
        loss = -min(ratio * advantage, clipped * advantage)
        effective_clip = (advantage > 0 and ratio > 1 + epsilon) or (advantage < 0 and ratio < 1 - epsilon)
        weight = 0. if effective_clip else 1.
        grad = -advantage * ratio * weight
    else:
        raise ValueError("方法只能为 sapo/grpo")
    if not all(math.isfinite(v) for v in (ratio, loss, grad)):
        raise ValueError("非有限比率/目标，须查输入与数值稳定性")
    return {"loss": loss, "d_loss_d_logp": grad, "ratio": ratio,
            "gradient_weight": weight, "effective_clip": effective_clip,
            "method": method}


def k3(logp: float, ref_logp: float) -> dict:
    diff = ref_logp - logp
    value = math.expm1(diff) - diff
    derivative = 1 - math.exp(diff)
    if not math.isfinite(value):
        raise ValueError("非有限 k3")
    return {"value": value, "d_loss_d_logp": derivative,
            "interpretation": "sampled_token_estimator_not_proof_of_true_KL_bound"}


def dpo(policy_chosen: float, policy_rejected: float, ref_chosen: float, ref_rejected: float,
        beta: float = .1) -> dict:
    if beta <= 0:
        raise ValueError("DPO beta 必须为正")
    margin = beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
    loss = max(0., -margin) + math.log1p(math.exp(-abs(margin)))
    return {"loss": loss, "margin": margin,
            "d_loss_d_policy_chosen": beta * (sigmoid(margin) - 1),
            "d_loss_d_policy_rejected": beta * (1 - sigmoid(margin))}


def gradient_probe() -> dict:
    cases = []
    for a, ratio in [(1., 1.), (-1., 1.), (.5, 1.35), (-.5, .65), (0., 1.5)]:
        for method in ("grpo", "sapo"):
            x, step = math.log(ratio), 1e-6
            expected = surrogate(x, 0., a, method)
            numerical = (surrogate(x + step, 0., a, method)["loss"] - surrogate(x - step, 0., a, method)["loss"]) / (2 * step)
            if abs(numerical - expected["d_loss_d_logp"]) > 1e-6:
                raise AssertionError("导数与中心差分不一致")
            cases.append({"advantage": a, "ratio": ratio, **expected, "finite_difference": numerical})
    return {"provenance_kind": "analytic_cpu_fixture", "not_model_training": True,
            "cases": cases, "status": "passed", "fingerprint": identity(cases)}
