# VL PostTrain Lab

面向多模态后训练与电商主动视觉 Agent 的数据、算法和评测工具。项目以 Qwen3-VL 为参考基座，研究如何让模型利用图片、按需裁剪观察、定位回答证据，并在有限工具与上下文预算内结束任务。

## 项目流程

项目包含两条相互衔接的方法路线，模型起点分别管理。

**一、多模态数据工程与后训练**

```text
公开多模态数据
  → 原生问答恢复与字段归一化
  → 图像去重、同源分组与划分隔离
  → 有图／无图 NLL 差值与视觉依赖选数
  → 通用 SFT → 推理 SFT
  → 从共同起点分别研究 DPO 与 GRPO
```

**二、电商包装主动视觉**

```text
Unitail-OCR / RPC 包装图
  → 字段问答、答案别名与紧证据框
  → 按商品家族及图像组划分
  → 局部定位 → 搜索、纠错与停止课程
  → 共同 SFT 起点
  → 同奖励 GRPO / SAPO 对照
  → 任务评分、观察干预、能力保留与成本分析
```

SAPO使用固定温度的平滑门控替代硬裁剪surrogate。对照共享数据、起点、参考模型、奖励和工具预算，将策略更新方式作为算法变量。阶段关系见[项目流程](docs/route.md)，公式和实现约定见[方法说明](docs/methods.md)。

## 代码仓各部分

| 位置 | 内容 |
|---|---|
| [`src/vl_posttrain/data/`](src/vl_posttrain/data/) | 数据适配、金标恢复、图像身份与分组、选数、偏好对、课程和监督mask |
| [`src/vl_posttrain/agent/`](src/vl_posttrain/agent/) | 真实像素crop、视图注册、局部到原图坐标、预算、轨迹及本地推理接口 |
| [`src/vl_posttrain/algorithms/`](src/vl_posttrain/algorithms/) | DPO、GRPO、SAPO目标函数、Torch损失、梯度诊断和冻结rollout合同 |
| [`src/vl_posttrain/evaluation/`](src/vl_posttrain/evaluation/) | 答案／证据评分、工具成本、重复裁剪、配对统计及基准评分组件 |
| [`src/vl_posttrain/config.py`](src/vl_posttrain/config.py) | 工具与token预算、模型分支和更新次数核算 |
| [`src/vl_posttrain/resources.py`](src/vl_posttrain/resources.py) | 模型加载前的内存、设备和权重元数据检查 |
| [`src/vl_posttrain/cli.py`](src/vl_posttrain/cli.py) | 命令行入口 |
| [`configs/`](configs/) | 流程、来源、评测和ZeRO-2参考配置，以及公开文件清单 |
| [`tests/`](tests/) | 数学、几何、数据隔离、预算、评分、CLI及发布边界测试 |
| [`tools/`](tools/) | 公开版检查、离线安装验证和源码打包 |
| [`.github/workflows/cpu.yml`](.github/workflows/cpu.yml) | Ubuntu／Windows CPU持续集成 |

训练部分提供参数参考、目标函数和执行合同；完整大模型训练需将组件接入实际训练器。`plan`用于核算参考配置，默认命令不会启动训练。

## 快速开始

Python 3.10+，建议使用独立虚拟环境。

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[images,numerical]"
vl-lab doctor
vl-lab plan --config configs/route.target.json
vl-lab gradient-probe
vl-lab demo --output outputs/synthetic_demo
```

`demo`使用自建测试图检查真实crop、视图几何和计数流程，是隔离工程示例。数据输入与已有模型接入见[使用说明](docs/reproduction.md)。

## 技术文档

| 文档 | 阅读目的 |
|---|---|
| [项目流程](docs/route.md) | 两条路线、共同起点、独立分支及阶段输入输出 |
| [方法说明](docs/methods.md) | 视觉依赖、坐标、奖励、SAPO／GRPO、mask与统计 |
| [使用说明](docs/reproduction.md) | CLI、CPU检查、已有模型推理和源码打包 |
| [数据构建](docs/data_release.md) | 审阅绑定、图片核验、家族隔离及标签分离 |
| [课程构建](docs/curriculum.md) | 定位、搜索停止、纠错上下文和通用回放 |
| [数据接口](docs/data_card.md) | 来源、字段、几何与标注约定 |
| [模型接入](docs/model_card.md) | 基座／adapter输入、资源检查和身份管理 |

## 检查与打包

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python tools/verify.py
python tools/verify_installed.py --output outputs/install_check
python tools/package_release.py
```

运行报告写入`outputs/`，源码包写入`dist/`。公开文件由[发布清单](configs/publication.json)限定，并与`.gitignore`配合管理。

## 许可

代码采用[MIT许可证](LICENSE)。数据、模型与第三方实现遵循各自条件；源码不随附原始图片或模型权重。
