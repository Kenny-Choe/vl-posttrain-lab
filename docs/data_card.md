# 数据来源与字段

## 来源索引

[来源示例配置](../configs/sources.example.json)列出候选来源：LLaVA与Vision-R1-cold用于多模态数据研究，Unitail-OCR／RPC用于包装字段候选，GQA／TextVQA训练侧用于普通回放。来源图像、原生问题、文字区域和本项目派生QA分别登记。

商品整框不是字段紧框；原始来源不直接等于课程或工具轨迹。具体来源版本、可用字段和上游条件由数据使用者保存，图片与私有标注不随源码发布。

## QARecord

结构定义在[`data/schema.py`](../src/vl_posttrain/data/schema.py)。重要字段包括样本／图片ID、source/revision/source_split、question、answers、图片路径与SHA、自然尺寸、evidence_box_xyxy、image_group_id、product_family_id和审查状态。

坐标使用EXIF规范显示后的原图自然像素，xyxy右下exclusive。文字题框所问文字，属性题框所问属性区域。别名显式列出；不可读、单位缺失、问题不唯一或需不可见知识的样本隔离。

actor仅接收task_id、question、image_id、image_path和image_sha256。金答案、GT框、IoU、距离反馈及review字段不进入策略观察。按家族和同源图像组划分，源侧评测样本始终受保护。

公共源码中的合成图和合成QA是测试fixture，不表示上游数据或模型输出。数据构建流程见[数据发布](data_release.md)。
