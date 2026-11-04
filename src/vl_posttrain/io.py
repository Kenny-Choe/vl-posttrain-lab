"""确定性 JSON/JSONL、文件身份和原子写入。"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def identity(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def strict_json(text: str) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"重复 JSON 字段：{key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"非有限 JSON 常量：{value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=reject_constant)


def read_json(path: str | Path) -> Any:
    return strict_json(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8-sig") as f:
        for number, line in enumerate(f, 1):
            if line.strip():
                value = strict_json(line)
                if not isinstance(value, dict):
                    raise ValueError(f"第{number}行必须为对象")
                rows.append(value)
    return rows


def atomic_text(path: str | Path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(prefix=f".{p.name}.", dir=p.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path: str | Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    atomic_text(path, "".join(canonical(row) + "\n" for row in rows))


def finite(value: Any, name: str = "number") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} 必须为有限数值")
    return float(value)


def contained_path(root: str | Path, relative: str) -> Path:
    base = Path(root).resolve()
    path = (base / relative).resolve()
    if path == base or not path.is_relative_to(base):
        raise ValueError("文件必须位于指定目录内")
    return path
