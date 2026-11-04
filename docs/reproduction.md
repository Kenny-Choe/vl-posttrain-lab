# 使用说明

## 安装与CPU检查

在仓库根目录创建独立虚拟环境，安装`.[images,numerical]`。完整Torch损失测试可安装CPU版Torch；未装可选依赖时检查报告会明确列出跳过项。

```bash
python -m pip install -e ".[images,numerical]"
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install "setuptools>=68" wheel
vl-lab doctor
vl-lab plan --config configs/route.target.json
vl-lab gradient-probe --output outputs/gradient_probe.json
vl-lab demo --output outputs/synthetic_demo
python tools/verify.py
python tools/verify_installed.py --output outputs/install_check
```

`verify.py`仅发现公开清单中的测试和文档，不依赖私人报告、模型身份或实验成绩。检查报告写入`outputs/verification/`。`verify_installed.py`离线构建wheel、安装到独立target，从空工作目录核对导入和CLI；需预先装好构建依赖，已有输出目录不覆盖。

## 数据入口

```bash
vl-lab normalize --input native.jsonl --source SOURCE --revision REV --output outputs/normalized.jsonl
vl-lab group --images images.jsonl --edges confirmed_edges.jsonl --protected protected_ids.json --output outputs/groups.json
vl-lab split --input reviewed_grouped_qa.jsonl --counts split_counts.json --output outputs/split
vl-lab select --input scored_train.jsonl --count 20000 --arm visual_mixture --output outputs/selected.jsonl
```

这些输入需由使用者提供；分组结果先写回QA，再划分。原生test身份不能通过重命名project_split转为train。审阅到发布见[数据构建](data_release.md)，领域轨迹与回放见[课程构建](curriculum.md)。

## 已有本地模型

```bash
python -m pip install -e ".[inference]"
vl-lab processor-preflight --model /path/to/base --output outputs/processor_check
vl-lab model-preflight --model /path/to/base --device cuda:0 --output outputs/model_admission.json
vl-lab infer --model /path/to/base --adapter /path/to/adapter \
  --tasks private/tasks.jsonl --image-root private/images \
  --output outputs/run01 --run-id run01 --device cuda:0 --max-episodes 1
```

模型路径由使用者显式指定，权重从本地读取。预检拒绝时保留JSON并返回退出码2；准入是估算，不保证峰值。单个后端调用的硬超时需外部进程监督，异常采样续跑不宣称bit-exact。协议或模板变化时使用新run目录。

```bash
vl-lab metrics --input outputs/run01/predictions.jsonl --output outputs/run01/metrics.json
vl-lab compare --left outputs/left/predictions.jsonl --right outputs/right/predictions.jsonl --output outputs/paired.json
```

## 源码包

```bash
python tools/package_release.py
```

`configs/publication.json`决定公开文件，`.gitignore`阻止本地个人材料被普通Git添加。打包器独立执行同一范围约束，包内不含私人报告、原图、权重或依赖目录。运行产物留在outputs，源码ZIP与manifest留在dist。CI只运行CPU检查和安装／打包验证。
