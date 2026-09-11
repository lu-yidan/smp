# R1–R4：从零训练的自然低位 reset 消融

以 B1 为对照：部署适配 93D 单帧 actor、10 秒 episode、原始 f2s2 prior、动态 GSI、原奖励、原 PPO、观测噪声和域随机化。四组不加载任何旧 PPO checkpoint。4096 环境，10000 次更新，每 500 次保存，seed 20260911。

分支 `codex/scratch-recovery-low-reset`；本地 `/home/luyd/workspace/smp-master-repro`，服务器 `/root/workplace/smp-master-repro`。

| 组 | GPU | reset | 新增自然 reset 的低 SMP 终止 |
|---|---:|---|---|
| B1（正在继续） | 1 | 100% 原始 GSI | 不适用，原规则 |
| R1 | 2 | 90% 原始 GSI + 10% 自然低位 | 原规则：5 步后单次低于阈值即结束 |
| R2 | 3 | 与 R1 完全相同 | 先宽限 0.5 秒，之后连续低分 0.5 秒才结束 |
| R3 | 4 | 与 R1 完全相同 | 先宽限 0.5 秒，之后连续低分 1.0 秒才结束 |
| R4 | 5 | 与 R1 完全相同 | 先宽限 0.5 秒，之后连续低分 1.5 秒才结束 |

B0 在 GPU0 继续。B2–B5 已按用户方向停止，保留最近 checkpoint，停止进度记录在服务器 `run_control/scratch93_termination_seed20260911_10000/stopped_B2_B5_for_reset_experiments.json`，不声称它们完成了 10000 次更新。

## reset 分布

每次 reset 独立抽样 10% 自然低位，使用私有 RNG。原始 GSI 仍对所有 reset 环境先执行相同抽样，再覆盖自然子集，避免额外消耗全局 GSI/网络初始化随机数。生成池仍为 4096 个窗口，每 2400 控制步刷新 1024。

新增自然子集内部：右侧 50%、左侧 20%、俯卧 20%、仰卧 10%。因此总 episode 抽样概率为右侧 5%、左侧 2%、俯卧 2%、仰卧 1%；这不是样本帧数比例，也不是环境步数比例。每个方向内先对原始片段组均分概率，再对组内帧均分，避免长片段主导。

自然 reset 使用零速度、重复当前状态的 SMP 历史；不是把静态姿态伪装成原始 GSI 动态窗口。姿态保持数据中的朝向，不添加新随机 yaw 扰动。actor 仍只有单帧 93D，无新增观测。

数据源为已审核接触的 `lafan_stratified_bank.npz` low 层：base height <0.3m，倒地阶段至首次站起阶段之间，部署模型下检查全部关节限位、有限值和初始接触，最深穿透约 1.999mm（上限2mm）。本次没有修复或筛掉原有 GSI 生成池的穿透，避免同时改动 baseline 的 GSI 分布。

按 `sha256('reset-v1:'+原始片段名) mod 5` 划分训练/留出，镜像对保持在同一集合：

| 集合 | 总数 | 仰卧 | 俯卧 | 左侧 | 右侧 | 原始片段组 |
|---|---:|---:|---:|---:|---:|---:|
| 训练 | 1714 | 1240 | 108 | 183 | 183 | 45 |
| 留出 | 645 | 346 | 226 | 37 | 36 | 12 |

训练 bank：`datasets/reset_banks/natural_low_right_v1.npz`，SHA256 `43943c6872a3816cb44ac04db201010cdb6bcaf02cc71c6d591c8b8e57f60f75`。
留出 bank：同目录 `natural_low_right_v1_heldout.npz`，SHA256 `468ed9d457a4222295e5cc4e65a687a4b3b2cf54641c7698ea35239e635a0d9b`。

这个留出是自然 reset 数据的片段留出，不保证对 prior 预训练未见。此前诊断用的均衡256集来自同一大候选池，可能与本次训练池重叠，因此今后不能把它当作本实验独立测试集。使用新的645留出集并按方向报告，另继续保留规整128姿态测试。

## R2 的时间逻辑

分数为 `s = exp(-6 * raw_error)`，阈值仍是 `s < 0.02`，等价于 raw_error > 0.652（约值）。raw_error 是扩散去噪误差，不是关节角度偏差、距离或直接概率，不能称为“稍微偏离一点”。

仅针对自然 reset：

1. 前25控制步（0.5秒）不触发低 SMP 终止，并清零连续低分计数。
2. 从第26步开始，每次低于0.02加1；达标则立即清零计数。
3. 连续25次低分（0.5秒）才终止，因此一直低分时最早在第50步附近（1秒）结束。
4. 新 episode 清零计数。原90% GSI reset 保留前5步宽限、单次低分终止。

沿用现有管理器时序：终止检查读取上一控制步计算的 raw SMP。R1/R2 只改变上述时间条件，没有改变阈值或奖励。R2 联合检验启动宽限与持续超标计时；若有效，后续才能拆分两者各自贡献。

R3/R4 是用户要求增加的持续时间对照：启动宽限仍为 0.5 秒，只将连续低分阈值改为 1.0/1.5 秒。如果始终低分，最早分别在第75/100步（1.5/2.0秒）结束。中途一次达标即清零连续计时，不累计断续低分。这轮不是把启动宽限改成1/1.5秒；R2/R3/R4之间仅持续低分时间不同。R1/R2原运行不重启。

站起终止仍是头高≥1.2m、base速度<0.5m/s连续0.5秒（作为截断）；10秒超时作为截断；数值不稳定保护始终生效。宽限不关闭这些条件。

## prior 选择

当前保留 f2s2，以控制变量。当前实现共用一个 prior 做 GSI 生成与 SMP 奖励，直接换 V6/V7 会同时改变两个因素。

V6 来自更广的六条 LAFAN 序列；覆盖更广不保证生成状态物理有效或 PPO 更易学习。V7 checkpoint 配置显示从 V6 初始化，加入审核过的路线数据继续训练200 epoch；审核路线集只有6条独立路线及其镜像。它不是一个全新、覆盖全面的 GSI 生成器。后续可以单独测试 GSI 生成器，但必须与 SMP 评分 prior 解耦、正确使用各自 normalizer，并审计生成状态接触及方向分布。

## 验证与日志

`scripts/check_natural_low_ablation.py` 检查非目标配置一致、噪声和DR保留、宽限边界、持续计时、恢复后清零、原GSI规则、采样权重和镜像片段留出。
正式训练前每组128环境、4次更新试跑，检查自然 reset 零速度、低高度、GSI定期刷新、PPO更新和保存；正式训练不加载试跑权重。

W&B：

- R1：https://wandb.ai/tabletennis/smp/runs/scratch93-natural-low-r1-seed20260911
- R2：https://wandb.ai/tabletennis/smp/runs/scratch93-natural-low-r2-seed20260911
- R3：https://wandb.ai/tabletennis/smp/runs/scratch93-natural-low-r3-seed20260911
- R4：https://wandb.ai/tabletennis/smp/runs/scratch93-natural-low-r4-seed20260911

新增 `Natural/*` 日志包括各方向 episode 抽样比例、自然 reset 实际步数占比，以及自然/GSI 各自低 SMP 和站起终止率；这些训练终止率仍不是固定倒地评测成功率。

正式 checkpoint：`logs/rsl_rl/scratch93_natural_low/R{1,2}_seed20260911_10000/`。
启动记录：`run_control/natural_low_seed20260911_10000/launches.json`。

R3/R4 对应 `R3_seed20260911_10000/` 和 `R4_seed20260911_10000/`，启动记录为 `run_control/natural_low_longer_seed20260911_10000/launches.json`；批量启动入口增加 `--longer-holds` 仅启动GPU4/5上的两组。新增两组试跑为128环境、5次更新，逻辑检查覆盖第50/75/100步边界以及分数恢复后计数清零。
