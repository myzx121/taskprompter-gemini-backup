# Phase 1 梯度流与校准机制代码审计报告

审计时间：2026-10-29。审计对象为 `CVPR_continuous_20261028`、`CVPR_selective_20261028`、`CVPR_cotuning_20261028`、`CVPR_head_tuning_20261028`。本报告及备份均位于 `backup_gemini_20261029/phase1_audit`，没有写入正式实验目录。

## 结论

1. **没有发现优化器漏参数或 requires_grad 假死。** 四个队列的动态完整性检查均通过：`optimizer_no_duplicates=true`、断点恢复一致、冻结参考一致、缓存/在线推理检查通过。
2. **“小数点后完全一致”主要是实验设计造成的任务隔离，不是优化器漏加参数。** 训练脚本逐次只取一个任务的缓存特征（`run_continuous.py:140-148`），并调用 `forward_task`；continuous 的 depth policy 只改变 depth 分支。semseg/normals/edge 共享的冻结缓存和冻结 expert，因此它们不应因 depth policy 改变而发生浮点变化。该现象应标为“任务独立适配的预期不变性”，不能当作四任务共享梯度已经激活的证据。
3. **head_tuning 明确切断了 expert/共享特征的反向图。** `head_tuning.py:35-40` 在 expert 前向使用 `torch.no_grad()`，随后 `x.detach()`；这保证只训练最后预测头和 calibrator，但不可能产生共享 LoRA/FiLM 的四任务梯度。
4. **depth_gain=1.25 目前作用在每个任务的 expert 输出混合上，而不是四任务共享层。** `head_tuning.py:37` 的 `x=x+gain*(y-x)` 只在 `task=='depth'` 时执行，随后才进入该任务的 head。因此现有代码没有证据表明 depth_gain 直接污染 semseg 特征；语义-几何跷跷板更可能来自训练目标/输出校准和共享数据分布，而不是 depth_gain 乘在共享 backbone 上。
5. **calibration 位于任务输出之后。** `task_calibration.py:86-103` 对 raw prediction 做 depth affine、semseg bias/temperature、normal rotation；它没有直接修改共享特征。需要注意，`combined` 中 depth 的 `self.depth` 是固定 buffer，51 个可训练量主要来自 semseg bias 与 normals direction；因此不能把全部 RMSE 下降归因于“训练了 51 个参数”。

## 关键代码定位

| 位置 | 观察 | 判定 |
|---|---|---|
| `continuous.py:9-12` | depth_frozen 将 depth head 全部 `requires_grad=False` | 预期设计，不是漏参 |
| `run_continuous.py:121-125` | 参数来自 `network.parameters()`，优化器由 `optimizer_groups()` 构建 | 动态检查确认无重复、无遗漏 |
| `run_continuous.py:140-148` | 每一步只计算一个 task 的 loss | 造成跨任务指标可完全不变 |
| `head_tuning.py:35-40` | expert `no_grad` + head 输入 `detach` | 共享 expert 没有反向梯度 |
| `head_tuning.py:37` | depth gain 仅在 depth task 分支执行 | 不支持“直接污染四任务共享特征”的说法 |
| `head_tuning.py:39-40` | 任务 head 输出后才调用 calibrator | 校准是后置任务专属操作 |
| `task_calibration.py:86-103` | 校准作用于 raw prediction | 可以作为解耦后置校准的基础 |

## 已执行动态验证

以下现有审计输出均为 `all_passed=true`：

- continuous：8 个变体；参考冻结、源池隔离、精确 Adam 恢复、优化器无重复均通过。
- selective：10 个变体；同上全部通过。
- cotuning：10 个变体；同上全部通过。
- head_tuning：8 个变体；源 head 隔离、冻结 expert/BN、输入梯度阻断、断点兼容均通过。

这说明现有代码是“受控冻结适配器”实现，而不是已经实现了 Gemini 要求的“共享 LoRA/FiLM 四任务梯度协调”。

## Phase 1 修复建议（仅备份，不直接改正式工程）

- 保留现有冻结队列作为基线；不要把任务隔离不变性误报成 bug 修复收益。
- 新建独立实验分支，在 `forward_task` 中为候选 FiLM/LoRA 提供可导路径；只对新增模块解除冻结，原始 expert 与 backbone 继续冻结。
- 将校准器限制在每个任务 raw prediction 之后；depth affine、semseg temperature/bias、normal tangent 分支分别计数参数，并报告固定 buffer 与可训练参数。
- 为每个任务记录 `grad_norm` 和 optimizer 参数 ID，验证四任务确实触达共享轻量模块；不能只依赖最终指标判断梯度流。
- 在 Phase 2 前先完成零初始化 step=0 恒等性测试：四任务输出与未修改基座逐元素比较，误差必须为 0（或明确的浮点容差）。

## Phase 1 阶段门槛

Phase 1 已完成代码审计和动态验证，但 Gemini 文档中要求的“共享轻量模块四任务梯度协调”尚不存在于旧队列。因此下一步只能在本备份目录创建新架构并做 smoke/step=0 验证；在恒等性和参数计数通过前，不应启动 120/300 步正式比较。
