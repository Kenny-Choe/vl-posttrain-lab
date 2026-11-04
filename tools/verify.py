"""公开仓库的便携CPU检查，不读取私人报告或历史成绩。"""
from __future__ import annotations
import ast
import importlib.metadata
import importlib.util
import io
from pathlib import Path
import re
import struct
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if (ROOT / ".deps/torch-cpu").is_dir() and sys.platform == "win32" and sys.version_info[:2] == (3, 12) and struct.calcsize("P") == 8:
    sys.path.insert(0, str(ROOT / ".deps/torch-cpu"))
from vl_posttrain.io import read_json, write_json, file_sha256
from vl_posttrain.config import training_plan
from package_release import release_files, check_files


def verify():
    began = time.perf_counter()
    files = release_files()
    selected = {path.relative_to(ROOT).as_posix() for path in files}
    policy = read_json(ROOT / "configs/publication.json")
    for name in policy["exact_files"]:
        if name not in selected:
            raise ValueError("缺少公开文件：" + name)
    for path in files:
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8"), filename=path.relative_to(ROOT).as_posix())
    sys.path.insert(0, str(ROOT / "tests"))
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for path in files:
        if path.parent == ROOT / "tests" and path.name.startswith("test_") and path.suffix == ".py":
            suite.addTests(loader.loadTestsFromName(path.stem))
    stream = io.StringIO()
    tests = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    if not tests.wasSuccessful():
        print(stream.getvalue())
        raise SystemExit(1)
    links = 0
    for path in files:
        if path.suffix != ".md":
            continue
        for link in re.findall(r"\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if "://" in link or link.startswith("#"):
                continue
            target = (path.parent / link.split("#")[0]).resolve()
            if not target.exists() or not target.is_relative_to(ROOT.resolve()):
                raise ValueError(f"公开文档链接不可用：{path.name} → {link}")
            if target.is_file() and target.relative_to(ROOT.resolve()).as_posix() not in selected:
                raise ValueError("公开文档链接到私人文件：" + link)
            if target.is_dir() and not any(p.resolve().is_relative_to(target) for p in files):
                raise ValueError("公开文档链接到未发布目录：" + link)
            links += 1
    report = {"status": "passed", "scope": "public_CPU_tests_configuration_links_and_publication_boundary",
              "tests_run": tests.testsRun, "tests_passed": tests.testsRun - len(tests.skipped),
              "skipped": [{"test": str(t), "reason": why} for t, why in tests.skipped],
              "failures": len(tests.failures), "errors": len(tests.errors),
              "local_markdown_links_checked": links, "release_hygiene": check_files(files),
              "route_arithmetic": training_plan(read_json(ROOT / "configs/route.target.json")),
              "file_sha256": {p.relative_to(ROOT).as_posix(): file_sha256(p) for p in files},
              "elapsed_seconds": time.perf_counter() - began, "model_weights_loaded": False,
              "training_started": False, "GPU_jobs_started": 0,
              "private_artifacts_required": False, "target_scientific_results_proven": False}
    write_json(ROOT / "outputs/verification/cpu_verification.json", report)
    print(f"公开版CPU核验：{report['tests_passed']}/{report['tests_run']}通过，{len(tests.skipped)}跳过；{links}链接；不依赖私人材料。")
    return report


if __name__ == "__main__":
    verify()
