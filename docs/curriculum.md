# 定位与搜索课程

课程接受已发布train QA与独立普通问答回放，输出可审核的教师候选和真实crop；生成候选后仍需检查目标语义和可读性。

| 阶段 | 训练内容 | 监督约定 |
|---|---|---|
| 局部定位 | 局部读图、问题目标、紧证据和view绑定 | 只监督助手正确动作及final |
| 搜索与停止 | 粗到细、错误恢复、早期view引用和直接回答 | 注入错误历史作为不监督上下文 |
| 普通回放 | GQA／TextVQA训练侧原生问答 | 维持一般视觉指令，保护评测划分 |

```bash
vl-lab build-curriculum --domain-release outputs/release --image-root private/images \
  --replay private/replay.jsonl --replay-image-root private/replay_images \
  --output outputs/curriculum
vl-lab audit-curriculum --input outputs/curriculum/trajectories.private.jsonl \
  --processor /path/to/local/processor --output outputs/mask_audit
```

实际输出文件名以命令manifest为准。每条轨迹保留QA、原图、crop、动作与mask身份。原生processor审计检查实际token和结束后缀，不以解码字符串拼接替代位置核对。

`curriculum-fixture`用于隔离工程检查，不能混入实际任务评测。训练器需按阶段读取候选并持续维护optimizer与scheduler；构建CLI本身不训练模型。
