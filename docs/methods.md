# 方法与实现约定

## 数据身份与视觉依赖

`QARecord`将来源版本、原生QA、图片、图像组、家族、答案和原图紧框分开记录。actor只接收task_id、question、image_id、image_path和image_sha256；金标和评分留在私有侧。

文件SHA检查字节重复，规范RGB像素SHA检查解码等价，DCT pHash召回近似候选。只有确认的同源边进入分组；评测保护集合参与分组并优先隔离。家族关系高于图像，不由相似Logo自动认定。

视觉依赖定义为`NLL_without_image−NLL_with_image`。两分支重新模板化，预测同一目标token；t位置预测labels[t+1]。分层配额由最大余数法分配，混采先取高分，再从剩余候选不放回随机。它是代理指标，应控制任务构成、目标长度及训练预算。

## 真实视图与预算

每个view登记根图整数矩形和自然尺寸。局部0–1000坐标通过引用view映回根图：左上floor、右下ceil、右下exclusive。嵌套crop重新读取原图，不裁已缩小的模型输入，也不按金标修正预测。

工具观察包含真实图片。成功crop用实际processor grid检查视觉token；错误调用计次数。生成预算跨turn累计，上下文、工具、单图视觉与新增视觉分别记账。越界或截断不由后处理补成正常final。

## 奖励与评分

终局奖励为`R=V(A+0.5AI+0.05D)`：A是答案正确，I是最终原图框IoU，D是归一化中心距离反馈，V是合法且预算内结束。只在终局发一次奖励，主任务条件为答案、证据阈值与合法预算同时满足。

别名表显式冻结，单位不可模糊放分。crop窗口允许粗搜索，最终框按问题指定的文字或属性区域收紧。连续奖励与离散任务成功分别统计。

## GRPO、SAPO和参考正则

`rho=exp(logp_current−logp_old)`；GRPO任务损失为`−min(rho×adv,clip(rho,1−eps,1+eps)×adv)`。正优势超上界、负优势低下界才是有效裁剪。

SAPO用`f=4/tau×sigmoid(tau×(rho−1))`，任务损失`−adv×f`。正／负优势温度固定，导数为`−adv×rho×4p(1−p)`。ratio为1时与未裁剪策略梯度一致；软门控仍会饱和，不能替代奖励和数据设计。

两方法共用参考正则。k3计算`expm1(logp_ref−logp_cur)−(logp_ref−logp_cur)`；它是采样token估计，不能保证完整分布真KL有界。old对应采样行为策略，ref始终共同SFT，current参与更新。

动作token先做每轨迹均值再批均值。mask包括工具参数、回答和实际结束后缀，排除prompt、工具观察、视觉与padding。old/ref/优势detach，门控不可detach。padding在指数计算前屏蔽，防止NaN污染。

## 复用、恢复与统计

复用批在更新前冻结token、前缀、view、grid、mask、old/ref logps、奖励与优势。同批多轮更新不重新计算old，不重采样同分组。checkpoint需绑定policy、ref、optimizer、scheduler、RNG、逻辑游标和累计成本；中途恢复还需要原冻结rollout。

全部尝试进入任务分母，失败token进入每成功成本分子；无成功时成本未定义。重复crop结合根图IoU、面积比和有效分辨率。商品组配对bootstrap保留组内相关性，训练种子变化与题组抽样不确定性分别报告。

源码位置见[README目录表](../README.md)。项目采用已有DPO／GRPO／SAPO方法，提供可读参考实现与可微检查。
