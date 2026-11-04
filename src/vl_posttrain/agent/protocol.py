from __future__ import annotations

from ..io import strict_json
from .geometry import box

PROTOCOL_VERSION = "packaging-view-1000-v1"
SYSTEM_PROMPT = """你是包装细节视觉助手。所有图片以 view_id 注册。可以输出一次 crop 请求，或一次最终答案。
crop 格式：{"tool":"crop","view_id":"v0","box":[左,上,右,下]}。
final 格式：{"final":{"answer":"答案","evidence":{"view_id":"v0","box":[左,上,右,下]}}}。
每个框都相对引用视图，坐标为0到1000的边缘坐标，右下大于左上。粗搜索窗口不等于最终紧致证据框。
根据当前可见信息决定继续或结束；可以引用较早视图。输出唯一 JSON 对象，不输出代码围栏或额外解释。
最多3次crop；全轨迹最多2048生成token、新视觉2048token、上下文6144token。错误调用也消耗次数。
"""


def parse_action(text: str) -> dict:
    value = strict_json(text.strip())
    if not isinstance(value, dict):
        raise ValueError("动作必须为 JSON 对象")
    if set(value) == {"tool", "view_id", "box"} and value["tool"] == "crop":
        if not isinstance(value["view_id"], str):
            raise ValueError("view_id 类型错误")
        box(value["box"], 1000, 1000)
        return value
    if set(value) == {"final"} and isinstance(value["final"], dict):
        final = value["final"]
        if set(final) != {"answer", "evidence"} or not isinstance(final["answer"], str) or not final["answer"].strip():
            raise ValueError("final 字段不完整")
        evidence = final["evidence"]
        if not isinstance(evidence, dict) or set(evidence) != {"view_id", "box"} or not isinstance(evidence["view_id"], str):
            raise ValueError("evidence 格式错误")
        box(evidence["box"], 1000, 1000)
        return value
    raise ValueError("非法动作/多余字段")
