"""离线检查真实导出的token概率；不从整体grad_norm猜裁剪因果。"""
from __future__ import annotations

import math
from ..evaluation.metrics import quantile
from .objectives import surrogate, k3


def token_diagnostics(current, old, reference, advantages, mask, full_distribution_entropy=None) -> dict:
    sizes=[len(v) for v in (current,old,reference,advantages,mask)]
    if len(set(sizes))!=1 or not sizes[0]:
        raise ValueError("轨迹批量不一致或为空")
    ratios,clipped,weights,zero,kls,sampled_surprisals=[],[],[],[],[],[]
    for cur,prev,ref,adv,active in zip(current,old,reference,advantages,mask):
        if len({len(cur),len(prev),len(ref),len(active)})!=1 or not any(active):
            raise ValueError("token位置或动作mask不一致")
        for c,o,r,m in zip(cur,prev,ref,active):
            if m not in (0,1,False,True):
                raise ValueError("mask必须二值")
            if not m:
                continue
            if any(not math.isfinite(v) or v>1e-6 for v in (c,o,r)):
                raise ValueError("真实token logp必须有限且不大于0")
            grpo=surrogate(c,o,adv,"grpo")
            sapo=surrogate(c,o,adv,"sapo")
            ratios.append(grpo["ratio"])
            clipped.append(grpo["effective_clip"])
            weights.append(sapo["gradient_weight"])
            zero.append(adv==0)
            kls.append(k3(c,r)["value"])
            sampled_surprisals.append(-c)
    n=len(ratios)
    if full_distribution_entropy is not None:
        if len(full_distribution_entropy)!=n or not all(math.isfinite(v) and v>=0 for v in full_distribution_entropy):
            raise ValueError("entropy须为每个有效位置完整词表分布计算的值")
        entropy=sum(full_distribution_entropy)/n
    else:
        entropy=None
    selected=[w for w,c in zip(weights,clipped) if c]
    return {"valid_action_tokens":n,"ratio_p05":quantile(ratios,.05),"ratio_p95":quantile(ratios,.95),
            "ordinary_outside_clip_band_fraction":sum(r<.8 or r>1.2 for r in ratios)/n,
            "sign_effective_clip_fraction":sum(clipped)/n,"zero_advantage_token_fraction":sum(zero)/n,
            "mean_k3":sum(kls)/n,"mean_sapo_gate_derivative_on_grpo_clipped_positions":sum(selected)/len(selected) if selected else None,
            "mean_sampled_negative_logp_not_entropy":sum(sampled_surprisals)/n,"mean_full_distribution_entropy":entropy,
            "interpretation":"token_counts_and_local_gate_weights_not_total_parameter_gradient_norm_or_ability_gain"}
