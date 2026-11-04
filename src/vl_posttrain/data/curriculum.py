from __future__ import annotations

from ..agent.geometry import ViewRegistry
from ..agent.protocol import parse_action
from ..io import canonical, identity


def final_target(record: dict, registry: ViewRegistry, view_id: str) -> str:
    if record.get("project_split") != "train" or not record.get("evidence_box_xyxy"):
        raise ValueError("课程只能使用有证据的训练样本")
    local = registry.get(view_id).from_root(record["evidence_box_xyxy"])
    action = {"final": {"answer": record["answers"][0], "evidence": {"view_id": view_id, "box": local}}}
    parse_action(canonical(action))
    return canonical(action)


def validate_curriculum(rows: list[dict], expected: dict | None = None) -> dict:
    ids, phase = set(), {"localization": 0, "search_stop": 0, "general_replay": 0}
    kinds = set()
    for row in rows:
        if row["trajectory_id"] in ids or row["project_split"] != "train":
            raise ValueError("轨迹身份重复或跨集")
        ids.add(row["trajectory_id"])
        kind = row["curriculum_phase"]
        if kind not in phase:
            raise ValueError("未知课程阶段")
        phase[kind] += 1
        kinds.add(row["state_kind"])
        views = {"v0"}
        actions = row["actions"]
        if not actions or actions[-1].get("kind") != "final":
            raise ValueError("演示必须正常 final，不用截断作正例")
        calls = 0
        for action in actions:
            if action["kind"] == "crop":
                calls += 1
                parsed = parse_action(action["text"])
                if parsed.get("view_id") not in views:
                    raise ValueError("演示引用未来/未知视图")
                if not action.get("crop_pixels_sha256") or not action.get("processor_visual_tokens"):
                    raise ValueError("crop 演示缺少真实像素/processor收据")
                if action.get("injected_wrong_action") and action.get("supervised") is not False:
                    raise ValueError("程序注入的错误动作不能作为正确动作监督")
                if not action.get("new_view_id") or action["new_view_id"] in views:
                    raise ValueError("新 view 身份错误")
                views.add(action["new_view_id"])
            elif action["kind"] == "final" and kind != "general_replay":
                parsed = parse_action(action["text"])
                if parsed["final"]["evidence"]["view_id"] not in views:
                    raise ValueError("final 引用不存在视图")
            if action["kind"] in {"final", "crop"} and not action.get("injected_wrong_action") and action.get("supervised") is not True:
                raise ValueError("模型目标和终止必须受到监督")
        if kind == "localization" and calls > 1 or kind == "search_stop" and calls > 3:
            raise ValueError("课程工具预算不符")
    if expected and any(phase.get(key) != value for key, value in expected.items()):
        raise ValueError(f"实际课程计数 {phase} 未达到目标 {expected}")
    return {"trajectories": len(ids), "phase_counts": phase, "state_kinds": sorted(kinds),
            "manifest_sha256": identity(rows), "native_mask_validation": "requires_processor_replay"}
