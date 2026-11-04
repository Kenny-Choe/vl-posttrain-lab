from __future__ import annotations

import copy
import math


def modality_conditions(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    """删除结构化 image 内容而非伪造零像素图；两分支须重新 apply_chat_template。"""
    visual, text = copy.deepcopy(messages), copy.deepcopy(messages)
    count = 0
    for message in text:
        if isinstance(message["content"], list):
            count += sum(item.get("type") == "image" for item in message["content"])
            message["content"] = [item for item in message["content"] if item.get("type") != "image"]
    if not count:
        raise ValueError("缺少结构化图片")
    return visual, text


def target_ids(input_ids: list[int], labels: list[int], allowed: list[bool]) -> list[int]:
    if len(input_ids) != len(labels) or len(labels) != len(allowed):
        raise ValueError("位置 mask 长度不一致")
    result = []
    for i, (token, label, ok) in enumerate(zip(input_ids, labels, allowed)):
        if label == -100:
            continue
        if i == 0 or not ok or token != label:
            raise ValueError("非 assistant/首位置/目标错位")
        result.append(label)
    if not result:
        raise ValueError("无监督目标")
    return result


def shifted_nll(logits, labels: list[int]) -> tuple[float, int]:
    """小型 CPU 参考实现；t 的 logits 预测 t+1，不先构造全词表 softmax。"""
    if len(logits) != len(labels) or not labels or labels[0] != -100:
        raise ValueError("causal NLL 输入不正确")
    total, count = 0.0, 0
    for t in range(len(labels) - 1):
        target = labels[t + 1]
        if target == -100:
            continue
        row = [float(x) for x in logits[t]]
        if not row or not all(math.isfinite(x) for x in row) or not 0 <= target < len(row):
            raise ValueError("非法 logits/target")
        maximum = max(row)
        total += maximum + math.log(sum(math.exp(v - maximum) for v in row)) - row[target]
        count += 1
    if count == 0:
        raise ValueError("无 shifted 目标")
    return total / count, count


def visual_dependency(nll_with: float, nll_without: float, ids_with: list[int], ids_without: list[int]) -> float:
    if not ids_with or ids_with != ids_without:
        raise ValueError("两模态条件的目标 token 不同")
    if not all(math.isfinite(v) for v in (nll_with, nll_without)):
        raise ValueError("非有限 NLL")
    return nll_without - nll_with
