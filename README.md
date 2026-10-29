# TaskPrompter Gemini Backup Experiments

这是 TaskPrompter 项目针对 `gemini-code-1793264781769.md` 要求执行的独立备份实验记录。

## 直接阅读

- [Phase 1 梯度流与校准机制代码审计报告](backup_gemini_20261029/Phase1_梯度流与校准机制代码审计报告.md)
- [Phase 2 实验报告](backup_gemini_20261029/Phase2_实验报告.md)
- [Phase 2 零初始化验证结果](backup_gemini_20261029/phase2_results/phase2_invariants.json)

## 当前结论

- 旧实验中的指标锁死主要来自任务隔离和冻结缓存，不是优化器漏参数。
- 严格零初始化 FiLM/LoRA、后置任务校准和四任务梯度触达验证已经通过。
- 当前没有新的大幅性能提升结果，因此没有把小幅波动方案升级为正式候选。
- Exp-A 需要先恢复 `.pt.zst` 缓存为训练器可读的 `.pt` 文件，之后才能继续 NYUD 120/300 步验证。

所有内容均为备份实验，不写入正式实验目录。
