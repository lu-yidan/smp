# Recovery-Core 消融矩阵

所有主要消融必须保持机器人、网络、actor observation、PPO 预算、seed、reward 尺度和
平地环境一致，每次只改变一个因素。统一使用 4096 environments、20000 updates、
seed 3883 和每 250 updates 保存。

Full R2 和全部 R2 消融统一使用 372 维四帧 actor observation；恒为零的
`base_lin_vel` 已完全删除，critic 的 privileged velocity 保留。

## 核心消融

| ID | Reset | SMP reward | 有序恢复约束 | 目的 |
|---|---|---|---|---|
| A Full R2 | 60% GSI + 40% 程序化 | 有 | 有 | 主方法 |
| B GSI-only | 100% GSI | 有 | 有 | 检验程序化离分布倒地是否必要 |
| C Procedural-only | 100% 程序化四姿态 | 有 | 有 | 检验 GSI 是否改善自然性与阶段覆盖 |
| D No-SMP-reward | 与 A 相同 | 无 | 有 | 隔离 SMP reward 的贡献 |
| E No-ordered-route | 与 A 相同 | 有 | 无 | 检验阶段约束是否阻止弹射/捷径 |

A 与 B 当前并行运行。C、D、E 应在完成固定 checkpoint 健康门后依次启动，不能把压板、
地形或不同网络结构混进核心消融。

## 部署相关补充实验

以下是补充分析，不必都作为同等规模的主消融：

1. 1-frame 对比 4-frame，检验历史是否弥补不可观测速度；
2. actor 使用真实 `base_lin_vel` 的 privileged oracle，作为状态估计上界；
3. 关闭 domain randomization，量化摩擦、编码器偏置和质心随机化对 sim-to-real 的贡献；
4. F2S2 prior 对比 LAFAN-route V7 prior，量化恢复轨迹覆盖范围的贡献。

## 统一评测

每个模型都在不参与训练的冻结 reset 集上报告：

- 俯卧、仰卧、左侧卧、右侧卧成功率；
- GSI holdout 成功率；
- 有序阶段完成率；
- 恢复时间 median/P90；
- 最大关节速度、力矩和功率；
- 脚步位移、站立宽度和站立后再次跌倒率。

只有训练 reward 不足以选择 checkpoint。
