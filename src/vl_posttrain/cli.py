from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

from .io import read_json, read_jsonl, write_json, write_jsonl, identity


def main(argv=None):
    parser = argparse.ArgumentParser(prog="vl-lab", description="数据治理、主动视觉与离线评测；不启动训练")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="检查可选依赖，不下载或加载模型")
    plan = sub.add_parser("plan", help="核算目标配置与步数，不执行")
    plan.add_argument("--config", type=Path, default=Path("configs/route.target.json"))
    demo = sub.add_parser("demo", help="隔离的合成 CPU 工程示例")
    demo.add_argument("--output", type=Path, default=Path("outputs/synthetic_demo"))
    probe = sub.add_parser("gradient-probe", help="纯 CPU 数学导数检查")
    probe.add_argument("--output", type=Path)
    processor = sub.add_parser("processor-preflight", help="已有本地processor的真实CPU/crop/上下文预检，不加载权重")
    processor.add_argument("--model", required=True)
    processor.add_argument("--output", type=Path, default=Path("outputs/native_processor_preflight"))
    admission=sub.add_parser('model-preflight',help='读取已有模型索引和容器/GPU内存，权重加载前准入')
    admission.add_argument('--model',type=Path,required=True)
    admission.add_argument('--device',default='cpu')
    admission.add_argument('--output',type=Path)
    score = sub.add_parser("metrics", help="汇总已有逐题结果，不生成模型输出")
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    compare = sub.add_parser("compare", help="对同一完整题集做商品组配对 bootstrap")
    compare.add_argument("--left", type=Path, required=True)
    compare.add_argument("--right", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    select = sub.add_parser("select", help="分层随机/视觉依赖混采")
    select.add_argument("--input", type=Path, required=True)
    select.add_argument("--count", type=int, required=True)
    select.add_argument("--arm", choices=["filtered_random", "visual_mixture"], required=True)
    select.add_argument("--seed", type=int, default=42)
    select.add_argument("--output", type=Path, required=True)
    groups = sub.add_parser("group", help="确认边、文件/像素身份与家族组隔离")
    groups.add_argument("--images", type=Path, required=True)
    groups.add_argument("--edges", type=Path, required=True)
    groups.add_argument("--protected", type=Path)
    groups.add_argument("--output", type=Path, required=True)
    split = sub.add_parser("split", help="同源组划分并按配额抽样，不复制凑数")
    split.add_argument("--input", type=Path, required=True)
    split.add_argument("--counts", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--seed", type=int, default=42)
    native = sub.add_parser("normalize", help="单轮 LLaVA/原生 QA 归一化，缺金标仍标 unverified")
    native.add_argument("--input", type=Path, required=True)
    native.add_argument("--source", required=True)
    native.add_argument("--revision", required=True)
    native.add_argument("--output", type=Path, required=True)
    dataset=sub.add_parser('prepare-packaging',help='审阅+真实图片核验+保护/家族隔离+配额发布，不训练')
    dataset.add_argument('--records',type=Path,required=True);dataset.add_argument('--reviews',type=Path,required=True)
    dataset.add_argument('--image-root',type=Path,required=True);dataset.add_argument('--output',type=Path,required=True)
    dataset.add_argument('--counts',type=Path);dataset.add_argument('--protected-index',type=Path);dataset.add_argument('--edges',type=Path)
    dataset.add_argument('--scope',choices=['packaging','legacy_regression','synthetic_fixture'],default='packaging')
    dataset.add_argument('--seed',type=int,default=42)
    dataset_demo=sub.add_parser('dataset-demo',help='隔离的96或12K规模CPU渲染fixture，非真实数据成果')
    dataset_demo.add_argument('--output',type=Path,default=Path('outputs/synthetic_dataset_demo'))
    dataset_demo.add_argument('--qa-count',type=int,default=96)
    curriculum=sub.add_parser('build-curriculum',help='已发布train QA→真实像素课程候选，不训练')
    curriculum.add_argument('--domain-release',type=Path,required=True);curriculum.add_argument('--image-root',type=Path,required=True)
    curriculum.add_argument('--replay',type=Path,required=True);curriculum.add_argument('--replay-image-root',type=Path,required=True)
    curriculum.add_argument('--output',type=Path,required=True)
    curriculum.add_argument('--localization',type=int,default=4000);curriculum.add_argument('--search-stop',type=int,default=4000)
    curriculum.add_argument('--replay-count',type=int,default=1000);curriculum.add_argument('--wall-cap',type=int,default=900)
    cfixture=sub.add_parser('curriculum-fixture',help='用独立合成回放检查课程规模，非真实9K语料')
    cfixture.add_argument('--domain-release',type=Path,required=True);cfixture.add_argument('--image-root',type=Path,required=True)
    cfixture.add_argument('--output',type=Path,required=True)
    cfixture.add_argument('--localization',type=int,default=4000);cfixture.add_argument('--search-stop',type=int,default=4000)
    cfixture.add_argument('--replay-count',type=int,default=1000);cfixture.add_argument('--wall-cap',type=int,default=1800)
    caudit=sub.add_parser('audit-curriculum',help='已有processor的真实CPU课程展开和位置mask审计')
    caudit.add_argument('--input',type=Path,required=True);caudit.add_argument('--processor',type=Path,required=True)
    caudit.add_argument('--output',type=Path,required=True);caudit.add_argument('--max-records',type=int,default=16)
    infer = sub.add_parser("infer", help="显式对现有本地模型推理；不训练、不下载权重")
    infer.add_argument("--model", required=True)
    infer.add_argument("--adapter")
    infer.add_argument("--tasks", type=Path, required=True)
    infer.add_argument("--image-root", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--run-id", required=True)
    infer.add_argument("--device", default="cpu")
    infer.add_argument("--max-episodes", type=int, default=1)
    args = parser.parse_args(argv)
    exit_code=0
    if args.command == "doctor":
        result = {"dependencies": {m: bool(importlib.util.find_spec(m)) for m in ("PIL", "numpy", "torch", "transformers", "peft")},
                  "models_loaded": False, "training_enabled": False}
    elif args.command == "plan":
        from .config import training_plan
        result = training_plan(read_json(args.config))
    elif args.command == "demo":
        from .demo import run_demo
        result = run_demo(args.output)
    elif args.command == "gradient-probe":
        from .algorithms.objectives import gradient_probe
        result = gradient_probe()
        if args.output:
            write_json(args.output, result)
    elif args.command == "processor-preflight":
        from .agent.backend import cpu_processor_preflight
        result=cpu_processor_preflight(args.model,args.output)
        write_json(args.output/"processor_receipt.json",result)
    elif args.command=='model-preflight':
        from .resources import model_admission
        if args.device.startswith('cuda'):os.environ.setdefault('CUDA_DEVICE_ORDER','PCI_BUS_ID')
        result=model_admission(args.model,args.device)
        if args.output:write_json(args.output,result)
        if not result['admitted']:exit_code=2
    elif args.command == "metrics":
        from .evaluation.metrics import aggregate
        result = aggregate(read_jsonl(args.input))
        write_json(args.output, result)
    elif args.command == "compare":
        from .evaluation.bootstrap import paired_group_bootstrap
        result = paired_group_bootstrap(read_jsonl(args.left), read_jsonl(args.right))
        write_json(args.output, result)
    elif args.command == "select":
        from .data.selection import select
        result = select(read_jsonl(args.input), args.count, args.arm, args.seed)
        write_jsonl(args.output, result.pop("rows"))
        write_json(str(args.output) + ".manifest.json", result)
    elif args.command == "group":
        from .data.grouping import build_groups
        protected = set(read_json(args.protected)) if args.protected else set()
        result = build_groups(read_jsonl(args.images), read_jsonl(args.edges), protected)
        write_json(args.output, result)
    elif args.command == "split":
        from .data.grouping import split_and_sample
        result = split_and_sample(read_jsonl(args.input), read_json(args.counts), args.seed)
        for name, rows in result.pop("splits").items():
            write_jsonl(args.output / f"{name}.jsonl", rows)
        write_json(args.output / "split_manifest.json", result)
    elif args.command == "normalize":
        from .data.schema import normalize_native_qa
        rows = [normalize_native_qa(r, args.source, args.revision) for r in read_jsonl(args.input)]
        write_jsonl(args.output, rows)
        result = {"count": len(rows), "manifest_sha256": identity(rows), "native_gold_still_requires_verification": True}
    elif args.command=='prepare-packaging':
        from .data.release import prepare_release,write_release
        counts=read_json(args.counts) if args.counts else {'train':8000,'dev':2000,'test':2000}
        prepared=prepare_release(read_jsonl(args.records),read_jsonl(args.reviews),args.image_root,counts,scope=args.scope,seed=args.seed,
            protected_images=read_jsonl(args.protected_index) if args.protected_index else [],
            reviewed_edges=read_jsonl(args.edges) if args.edges else [])
        result=write_release(args.output,prepared)
        if result['status']=='partial_needs_review':exit_code=2
    elif args.command=='dataset-demo':
        from .data.demo import dataset_demo
        result=dataset_demo(args.output,args.qa_count)
        if result['status']!='fixture_engineering_passed':exit_code=2
    elif args.command=='build-curriculum':
        from .data.curriculum_build import build_curriculum
        result=build_curriculum(args.domain_release,args.image_root,read_jsonl(args.replay),args.replay_image_root,args.output,
               localization=args.localization,search_stop=args.search_stop,general_replay=args.replay_count,wall_cap_seconds=args.wall_cap)
        if not result['geometric_replay_passed']:exit_code=2
    elif args.command=='curriculum-fixture':
        from .data.curriculum_build import build_curriculum,verify_release
        from .data.curriculum_fixture import create_replay_fixture
        if not verify_release(args.domain_release)[-1]:parser.error('该入口只用于合成domain fixture，不混入真实QA')
        replay=create_replay_fixture(args.output/'replay_inputs',args.replay_count)
        result=build_curriculum(args.domain_release,args.image_root,replay,args.output/'replay_inputs',args.output/'curriculum',
               localization=args.localization,search_stop=args.search_stop,general_replay=args.replay_count,wall_cap_seconds=args.wall_cap)
        if not result['geometric_replay_passed']:exit_code=2
    elif args.command=='audit-curriculum':
        from .data.curriculum_native import audit_native_curriculum
        result=audit_native_curriculum(args.input,args.processor,args.output,max_records=args.max_records)
    elif args.command == "infer":
        from .resources import model_admission
        if args.device.startswith('cuda'):os.environ.setdefault('CUDA_DEVICE_ORDER','PCI_BUS_ID')
        from .agent.backend import LocalQwenBackend
        from .agent.rollout import run_episode
        from .agent.protocol import PROTOCOL_VERSION, SYSTEM_PROMPT
        from .config import Budgets
        from dataclasses import asdict
        from .data.schema import QARecord
        if not 1 <= args.max_episodes <= 2000:
            parser.error("max-episodes 需在1到2000之间")
        rows = read_jsonl(args.tasks)
        if len(rows) > args.max_episodes:
            parser.error("题数超出本次有限上限；请明确选择任务清单或调整 max-episodes")
        result=model_admission(args.model,args.device)
        write_json(args.output/'model_admission.json',result)
        if not result['admitted']:
            parser.error('实际资源不满足权重准入：'+','.join(result['reasons']))
        backend = LocalQwenBackend(args.model, args.adapter, args.device)
        result = [run_episode(QARecord(**row), backend, args.image_root, args.output / identity(row["sample_id"]), args.run_id,
                              identity([PROTOCOL_VERSION, SYSTEM_PROMPT, asdict(Budgets())])) for row in rows]
        write_jsonl(args.output / "predictions.jsonl", result)
        result = {"episodes": len(result), "model_identity": backend.identity, "optimizer_steps": 0}
    else:
        raise AssertionError("未处理子命令")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if exit_code:raise SystemExit(exit_code)
