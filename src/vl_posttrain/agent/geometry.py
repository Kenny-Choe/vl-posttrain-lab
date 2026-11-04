from __future__ import annotations

import math
from dataclasses import dataclass
from ..io import finite


def box(values, width: float, height: float) -> tuple[float, float, float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError("框必须为四个数")
    x0, y0, x1, y1 = [finite(v, "box") for v in values]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("越界、反向或空框；禁止静默修正")
    return x0, y0, x1, y1


def area(rect) -> float:
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])


def iou(left, right) -> float:
    intersection = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    union = area(left) + area(right) - intersection
    return intersection / union if union > 0 else 0.0


@dataclass(frozen=True)
class View:
    view_id: str
    root_image_id: str
    root_box: tuple[int, int, int, int]
    parent_id: str | None

    @property
    def natural_size(self) -> tuple[int, int]:
        x0, y0, x1, y1 = self.root_box
        return x1 - x0, y1 - y0

    def to_root(self, normalized) -> tuple[int, int, int, int]:
        u0, v0, u1, v1 = box(normalized, 1000, 1000)
        x0, y0, x1, y1 = self.root_box
        return (math.floor(x0 + u0 / 1000 * (x1 - x0)), math.floor(y0 + v0 / 1000 * (y1 - y0)),
                math.ceil(x0 + u1 / 1000 * (x1 - x0)), math.ceil(y0 + v1 / 1000 * (y1 - y0)))

    def from_root(self, rectangle) -> list[float]:
        x0, y0, x1, y1 = rectangle
        left, top, right, bottom = self.root_box
        if not (left <= x0 < x1 <= right and top <= y0 < y1 <= bottom):
            raise ValueError("证据不在引用视图内")
        return [(x0 - left) / (right - left) * 1000, (y0 - top) / (bottom - top) * 1000,
                (x1 - left) / (right - left) * 1000, (y1 - top) / (bottom - top) * 1000]


class ViewRegistry:
    def __init__(self, image_id: str, width: int, height: int):
        if width <= 0 or height <= 0:
            raise ValueError("非法原图尺寸")
        self.image_id = image_id
        self.views = {"v0": View("v0", image_id, (0, 0, width, height), None)}

    def get(self, view_id: str) -> View:
        if not isinstance(view_id, str) or view_id not in self.views:
            raise ValueError("引用不存在或其他轨迹的 view")
        return self.views[view_id]

    def crop(self, view_id: str, normalized) -> View:
        parent = self.get(view_id)
        rectangle = parent.to_root(normalized)
        child_id = f"v{len(self.views)}"
        child = View(child_id, self.image_id, rectangle, view_id)
        self.views[child_id] = child
        return child
