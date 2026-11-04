"""按公开清单打包；私人材料即使仍在本地也不会进入源码包。"""
from __future__ import annotations
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vl_posttrain.io import file_sha256, read_json, write_json


def release_files(root: Path | None = None):
    root = (root or ROOT).resolve()
    policy = read_json(root / "configs/publication.json")
    excluded = policy["excluded_paths"]
    def eligible(path):
        relative = path.relative_to(root).as_posix()
        if any(relative == p or relative.startswith(p.rstrip("/") + "/") for p in excluded):
            return False
        if "__pycache__" in path.parts or any(part == "private" for part in path.relative_to(root).parts):
            return False
        if ".private." in path.name or ".local." in path.name:
            return False
        if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
            raise ValueError("公开目录不接受符号链接或junction")
        if not path.resolve().is_relative_to(root):
            raise ValueError("公开文件超出仓库")
        return True
    candidates = set()
    for folder, extensions in policy["recursive_directories"].items():
        directory = (root / folder).resolve()
        if not directory.is_relative_to(root):
            raise ValueError("公开目录超出仓库")
        if directory.is_symlink() or getattr(directory, "is_junction", lambda: False)():
            raise ValueError("公开目录不接受链接")
        for path in directory.rglob("*"):
            if eligible(path) and path.is_file() and path.suffix.lower() in extensions:
                candidates.add(path)
    for name in policy["exact_files"]:
        path = root / name
        if eligible(path) and path.is_file():
            candidates.add(path)
    return sorted(candidates, key=lambda p: p.relative_to(root).as_posix())


def check_files(files, root: Path | None = None):
    root = (root or ROOT).resolve()
    patterns = {
        "private_key": r"-----BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----",
        "personal_windows_path": r"[A-Z]:[\\/]+Users[\\/]+[^\s\"<>]+",
        "live_remote_root": r"/root/" + r"autodl-tmp/",
        "email_address": r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b",
        "github_access_token": r"(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,})",
        "hf_access_token": r"hf_[A-Za-z0-9]{24,}",
        "api_key": r"sk-[A-Za-z0-9_-]{28,}",
        "aws_access_id": r"AKIA[A-Z0-9]{16}",
    }
    for path in files:
        if path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("公开文件过大：" + str(path.relative_to(root)))
        content = path.read_text(encoding="utf-8-sig")
        for label, pattern in patterns.items():
            if re.search(pattern, content, re.I if label == "personal_windows_path" else 0):
                raise ValueError(f"公开文件触发{label}，请本地审查：{path.relative_to(root)}")
    return {"files": len(files), "raw_images_included": False, "weights_included": False,
            "private_workspace_included": False, "personal_reports_included": False,
            "credential_pattern_checks": "passed"}


def package(output: Path | None = None):
    files = release_files()
    check = check_files(files)
    output = output or ROOT / "dist/vl-posttrain-lab-source.zip"
    if not output.resolve().is_relative_to((ROOT / "dist").resolve()):
        raise ValueError("发布包必须输出到本仓库dist")
    output.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo("vl-posttrain-lab/" + relative, (2026, 11, 4, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
            entries.append({"path": relative, "sha256": file_sha256(path), "bytes": path.stat().st_size})
    result = {**check, "archive": output.name, "sha256": file_sha256(output), "bytes": output.stat().st_size,
              "entries": entries, "scope": "public_code_and_technical_documentation_only",
              "pushed_to_github": False, "native_models_evaluated_this_run": False}
    write_json(output.parent / "release_manifest.json", result)
    return result


if __name__ == "__main__":
    result = package()
    print(f"公开源码包：{result['files']}文件，{result['bytes']}字节；不含个人报告，未推送。")
