# 模型接入与身份

参考模型为Qwen3-VL-8B-Instruct，用户可显式提供兼容的本地基座与adapter。仓库不附模型权重或特定个人检查点注册表。

`LocalQwenBackend`在本地加载权重，冻结参数用于推理。模型路径、adapter路径和device由调用者指定；加载前读取权重索引、分片存在性与容器实际余量。processor预检单独检查多视图与上下文，资源预估通过不等于实际生成峰值保证。

模型身份应记录基座revision、文件SHA、adapter、父模型、参考模型、模板和processor配置。更改metadata不能改变权重的实际rank、训练数据或父关系。

训练研究中，通用SFT后接推理SFT；DPO／GRPO独立起点；电商课程另从原始基座开始。电商GRPO／SAPO共用课程SFT及固定reference，只有policy更新，reference不进入optimizer。

训练参数与ZeRO-2配置是参考方案，算法模块提供目标函数和执行合同，完整训练需接入实际训练器。具体CLI见[使用说明](reproduction.md)，分支关系见[项目流程](route.md)。
