from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
from ..config import Budgets
from ..io import contained_path, file_sha256, identity
from .geometry import ViewRegistry
from .protocol import SYSTEM_PROMPT, PROTOCOL_VERSION, parse_action


class CropEnvironment:
    """环境只接收公开 task，不持有答案/金框。所有子图读取规范原图。"""
    def __init__(self, task: dict, image_root: str | Path, output: str | Path, budgets: Budgets | None = None):
        from PIL import Image, ImageOps
        allowed = {"task_id", "question", "image_id", "image_path", "image_sha256"}
        if set(task) != allowed:
            raise ValueError("actor task 必须严格使用公开字段白名单")
        self.task = dict(task)
        self.budgets = budgets or Budgets()
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.root_path = contained_path(image_root, task["image_path"])
        if file_sha256(self.root_path) != task["image_sha256"]:
            raise ValueError("原图身份不一致")
        with Image.open(self.root_path) as source:
            rgba = ImageOps.exif_transpose(source).convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            background.alpha_composite(rgba)
            self.root_image = background.convert("RGB")
        self.registry = ViewRegistry(task["image_id"], *self.root_image.size)
        self.images = {"v0": self.root_image}
        self.calls = self.generated = self.new_visual = 0
        self.seen_visual = {}
        self.done = False
        self.termination = None
        self.final = None
        self.events = []
        prompt = SYSTEM_PROMPT + "\n本次实际预算：" + str(asdict(self.budgets))
        self.messages = [{"role": "system", "content": [{"type": "text", "text": prompt}]},
                         {"role": "user", "content": [{"type": "image", "image": self.root_image, "view_id": "v0"},
                          {"type": "text", "text": f'view_id=v0，原图自然尺寸={self.root_image.size}。{task["question"]}'}]}]

    def stop(self, reason: str):
        self.done, self.termination = True, reason

    def admit_processor(self, context_tokens: int, visual_counts: list[int]) -> int:
        """后端真实 processor 计数；历史视图不重复算新增，但每次 prefill 都记录。"""
        if self.done:
            raise ValueError("任务已结束")
        if len(visual_counts) != len(self.images) or any(not isinstance(n, int) or n <= 0 for n in visual_counts):
            raise ValueError("processor 图像数/视觉 token 错误")
        for (view_id, image), count in zip(self.images.items(), visual_counts):
            cap = self.budgets.initial_visual_tokens if view_id == "v0" else self.budgets.crop_visual_tokens
            if count > cap:
                self.stop("per_image_visual_budget")
                return 0
            if view_id not in self.seen_visual:
                self.new_visual += count
                self.seen_visual[view_id] = count
            elif self.seen_visual[view_id] != count:
                raise ValueError("同一视图的 processor 视觉身份发生改变")
        if self.new_visual > self.budgets.new_visual_tokens:
            self.stop("new_visual_budget")
            return 0
        room = min(self.budgets.generated_tokens - self.generated, self.budgets.context_tokens - context_tokens)
        if room <= 0:
            self.stop("generation_or_context_budget")
            return 0
        self.events.append({"type": "prefill", "context_tokens": context_tokens, "visual_counts": visual_counts,
                            "processed_historical_visual_tokens": sum(visual_counts), "new_visual_total": self.new_visual})
        return room

    def step(self, text: str, token_ids: list[int], finish_reason: str, latency_seconds: float = 0.0) -> dict:
        if self.done:
            raise ValueError("任务已结束，不允许重复执行")
        if not token_ids or any(isinstance(v, bool) or not isinstance(v, int) for v in token_ids):
            raise ValueError("必须保存真实采样 token ID，包括实际终止 token")
        self.generated += len(token_ids)
        event = {"type": "action", "text": text, "token_ids": token_ids, "finish_reason": finish_reason,
                 "latency_seconds": latency_seconds, "generated_total": self.generated}
        self.messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
        self.events.append(event)
        if self.generated > self.budgets.generated_tokens or finish_reason == "length":
            self.stop("generation_truncated_or_over_budget")
            return {"error": self.termination}
        attempt_counted = False
        try:
            action = parse_action(text)
            if "final" in action:
                final = action["final"]
                view = self.registry.get(final["evidence"]["view_id"])
                root_box = view.to_root(final["evidence"]["box"])
                self.final = {**final, "root_box": list(root_box)}
                self.stop("final")
                event["final"] = self.final
                return {"final": self.final}
            self.calls += 1
            attempt_counted = True
            if self.calls > self.budgets.crop_calls:
                self.stop("crop_budget_exhausted")
                return {"error": self.termination}
            view = self.registry.crop(action["view_id"], action["box"])
            # 永远从原图裁剪，避免多级 crop 反复放大低清缩略图。
            image = self.root_image.crop(view.root_box)
            self.images[view.view_id] = image
            path = self.output / f"{view.view_id}.png"
            image.save(path)
            observation = {"view_id": view.view_id, "parent_id": view.parent_id,
                           "natural_size": list(image.size), "root_box": list(view.root_box)}
            pixel_hash = hashlib.sha256(str(image.size).encode() + b"|RGB|" + image.tobytes()).hexdigest()
            event["crop"] = {**observation, "pixel_sha256": pixel_hash,
                             "saved_file_sha256": file_sha256(path), "visual_tokens": None}
            self.messages.append({"role": "tool", "content": [{"type": "text", "text": str(observation)},
                                  {"type": "image", "image": image, "view_id": view.view_id}]})
            return observation
        except (ValueError, TypeError, KeyError) as exc:
            # 非法 crop 在进入几何校验前可能已计费；每个失败 turn 仅消费一次。
            if not attempt_counted:
                self.calls += 1
            event["error"] = str(exc)
            if self.calls > self.budgets.crop_calls:
                self.stop("invalid_action_budget_exhausted")
            self.messages.append({"role": "tool", "content": [{"type": "text", "text": str({"error": str(exc), "remaining_calls": max(0, self.budgets.crop_calls - self.calls)})}]})
            return {"error": str(exc)}

    def receipt(self) -> dict:
        return {"task_id": self.task["task_id"], "image_id": self.task["image_id"],
                "protocol_version": PROTOCOL_VERSION, "root_image_sha256": self.task["image_sha256"],
                "status": self.termination or "running", "final": self.final, "crop_calls": self.calls,
                "generated_tokens": self.generated, "new_visual_tokens": self.new_visual,
                "views": {key: asdict(value) for key, value in self.registry.views.items()},
                "events": self.events}
