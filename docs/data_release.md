# 数据构建与发布接口

`prepare-packaging`把候选QA、审阅记录和实际图片绑定后，执行保护集合、家族隔离与配额发布。

1. 按`QARecord`验证来源、原生split、答案、紧框和图片身份。
2. 审阅记录绑定标注内容hash；答案或框改变后旧审核失效。
3. 核验文件与规范像素指纹，近似图只召回待确认候选。
4. 冲突标签、未知保护关系与几何问题进入隔离队列。
5. 按确认同源组和商品家族整体划分，保护成员优先。
6. 达到配额才输出完整发布；不足保留partial状态，不复制样本补数。

```bash
vl-lab prepare-packaging --records private/qa.jsonl --reviews private/reviews.jsonl \
  --image-root private/images --counts private/counts.json \
  --protected-index private/protected.jsonl --edges private/confirmed_edges.jsonl \
  --output outputs/release
```

公共任务字段只含问题、图片身份和路径，答案／GT框与审阅记录留在私有输出。输出目录有内容身份保护，不覆盖不同输入的发布版本。`dataset-demo`可用自建图检查同一工程流程，生成数据明确标为fixture。

字段约定见[数据接口](data_card.md)，代码在`data/schema.py`、`grouping.py`和`release.py`。
