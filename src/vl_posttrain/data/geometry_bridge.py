from ..agent.geometry import box
from ..io import finite


def checked_text_region(polygon, width: int, height: int) -> list[float]:
    if not isinstance(polygon, list) or len(polygon) < 3 or any(not isinstance(p, (tuple, list)) or len(p) != 2 for p in polygon):
        raise ValueError("polygon 至少三个二维点")
    xs = [finite(p[0]) for p in polygon]
    ys = [finite(p[1]) for p in polygon]
    if any(not (0 <= x <= width and 0 <= y <= height) for x, y in zip(xs, ys)):
        raise ValueError("polygon 越界")
    return list(box([min(xs), min(ys), max(xs), max(ys)], width, height))
