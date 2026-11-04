"""独立可微损失参考。需可选 Torch；不是已验证的 4卡 Swift 完整 Trainer。"""
from __future__ import annotations


def policy_loss(current, old, reference, advantages, mask, method="sapo", beta=.04,
                epsilon=.2, tau_pos=1., tau_neg=1.05):
    import torch
    if current.ndim != 2 or old.shape != current.shape or reference.shape != current.shape or mask.shape != current.shape:
        raise ValueError("logps/mask 必须同为 [轨迹,token]")
    if advantages.ndim != 1 or len(advantages) != current.shape[0]:
        raise ValueError("advantages 必须每轨迹一个值")
    if not current.requires_grad or old.requires_grad or reference.requires_grad or advantages.requires_grad:
        raise ValueError("current 必须可微；old/ref/advantages 必须 detach")
    if beta < 0 or tau_pos <= 0 or tau_neg <= 0:
        raise ValueError("非法系数")
    if not torch.all((mask == 0) | (mask == 1)) or not torch.all(mask.sum(-1) > 0):
        raise ValueError("二值动作 mask 且每轨迹非空")
    valid = mask.bool()
    if not torch.isfinite(advantages).all() or not all(torch.isfinite(t[valid]).all() for t in (current, old, reference)):
        raise ValueError("有效位置出现 NaN/Inf")
    # 先屏蔽 padding 再 exp，防止 NaN*0 污染损失与梯度。
    cur, prev, ref = [torch.where(valid, t, torch.zeros_like(t)).float() for t in (current, old, reference)]
    ratio = (cur - prev).exp()
    adv = advantages.float()[:, None]
    if method == "sapo":
        tau = torch.where(adv > 0, tau_pos, tau_neg)
        gate = 4 / tau * torch.sigmoid(tau * (ratio - 1))
        task = -gate * adv
    elif method == "grpo":
        task = -torch.minimum(ratio * adv, ratio.clamp(1 - epsilon, 1 + epsilon) * adv)
    else:
        raise ValueError("未知 surrogate")
    delta = ref - cur
    kl = delta.exp() - delta - 1
    per_token = task + beta * kl
    if not torch.isfinite(per_token[valid]).all():
        raise ValueError("非有限损失，禁止隐式 clamp 改变目标")
    reduce = lambda value: (torch.where(valid, value, torch.zeros_like(value)).sum(-1) / mask.sum(-1)).mean()
    loss = reduce(per_token)
    return loss, {"task_loss": reduce(task).detach(), "kl": reduce(kl).detach(),
                  "ratio": ratio.detach(), "normalization": "per_trajectory_then_batch_mean"}


def chunked_target_logps(logits, labels, chunk_size=64):
    import torch
    if logits.ndim != 3 or labels.shape != logits.shape[:2] or (labels[:, 0] != -100).any():
        raise ValueError("causal 张量形状/首目标错误")
    positions = (labels[:, 1:] != -100).nonzero(as_tuple=False)
    if not len(positions):
        raise ValueError("无目标 token")
    chunks = []
    for pos in positions.split(chunk_size):
        batch, time = pos.unbind(-1)
        selected = logits[batch, time].float()
        target = labels[batch, time + 1]
        chunks.append(selected.gather(-1, target[:, None]).squeeze(-1) - torch.logsumexp(selected, -1))
    return torch.cat(chunks), positions
