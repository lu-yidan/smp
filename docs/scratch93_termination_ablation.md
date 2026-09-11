# 从零训练 recovery：B0–B5 终止条件消融

第一轮目标：在部署适配的 93D 单帧 actor 下，分开检验时长、低 SMP 终止、站起终止。六组均从随机初始化开始，不加载 V33 或任何已训练 PPO actor/critic。冻结的 f2s2 SMP prior 是所有组共有的运动先验；这里的 from scratch 指 PPO 策略从零训练。

分支：`codex/scratch-recovery-termination-ablation`。
本地：`/home/luyd/workspace/smp-master-repro`。
服务器：`ssh -p 17535 root@192.168.11.91`，目录 `/root/workplace/smp-master-repro`。

## 第一轮配置

| 组 | GPU | 最长 episode | 站起终止 | 低 SMP 终止 |
|---|---:|---:|---|---|
| B0 | 0 | 5 秒 | 保留 | 保留 |
| B1 | 1 | 10 秒 | 保留 | 保留 |
| B2 | 2 | 10 秒 | 取消 | 保留 |
| B3 | 3 | 10 秒 | 保留 | 取消 |
| B4 | 4 | 10 秒 | 取消 | 取消 |
| B5 | 5 | 80% episode 为 5 秒；20% 为 20 秒 | 仅短任务保留 | 仅短任务保留 |

B5 在每次 reset 独立抽样任务类型，使用独立随机数生成器，避免抽样直接消耗 GSI/策略初始化的随机数流。模式在该 episode 内不变，不加入 actor 或 critic 观测。80/20 是 episode 抽样概率，**不是训练步数占比**。长任务可能贡献大部分训练步数；W&B 同时记录 episode 抽样占比和实际步数占比。

所有组加入相同 `unstable_sim_state` 数值保护，触发阈值与上一轮持续恢复组一致。这也是 B0 相比旧部署 5 秒组的一个显式差别，不能宣称逐项完全复现旧组。它不依据姿态、SMP 或是否倒地结束；触发率单独记录。B0 使用新 seed 20260911，旧成功组 seed 为 20260910。

原始站起终止：头高 ≥1.2m、base 速度 <0.5m/s 连续 25 控制步（0.5 秒），作为截断处理并保留 value bootstrap。原始低 SMP 终止：`exp(-6 * raw_err) < 0.02`，前 5 步宽限，作为真正终止。超时作为截断，数值不稳定作为真正终止。

## 六组共同固定

- 4096 并行环境，10000 次 PPO 外层更新，rollout 24 步，每 500 次保存。每组总计 983,040,000 个环境步。第一轮每组一个 seed；baseline 与入选配置后续至少 3 个 seed。
- 部署 XML、PD/动作裁剪/扭矩限制及入场混合与旧部署 93D 组一致。2ms 物理步、decimation 10、50Hz 控制、solver 100 / line search 50。
- actor 单帧 93D，去掉 base linear velocity；critic 保留原始特权 960D。尚不测试 actor 4/10 帧历史。
- 冻结 `pretrained_getup_f2s2.pt`，SHA256 `9439d9d9f58940f5472015a57da27c72e2305aafbd29cfd9be44f93da675cc59`。
- 原始 GSI：4096 个生成窗口，启动 batch 1024，每 2400 控制步 FIFO 刷新 1024；保留原始生成状态速度和 10 帧 SMP 历史。没有加入静态倒地/LAFAN reset。
- 原始任务奖励乘 SMP，未增加站稳奖励、动作平滑惩罚或翻滚惩罚。
- PPO 初始 lr=1e-3 adaptive，gamma=0.99，lambda=0.95，固定动作探索 std=0.30，actor/critic 隐层 512/256/128，ELU，观测归一化。
- actor 观测噪声开启：角速度 ±0.2、重力方向 ±0.05、关节位置 ±0.01、关节速度 ±1.5。critic 不加噪声。
- 原始足部摩擦 0.3–1.2，编码器偏置 ±0.015，躯干 COM 随机化，1–3 秒间隔随机推扰；显式接触对同步足部摩擦。没有新加执行器增益、延迟或惯量随机化。

## 日志与核验

入口 `scripts/train_termination_ablation.py`；批量启动 `scripts/launch_termination_ablation.py`。
`scripts/check_termination_ablation.py` 检查六组非消融配置一致、截断/终止标志、B5 80/20 抽样及局部 reset、私有 RNG 不改变全局 RNG、250/1000 步超时边界、站起和低 SMP 的模式屏蔽。

小规模 preflight 每组 16 环境、2 次更新，另强制验证一次 GSI 刷新；这些检查不是学习效果评估，也不会加载到正式训练。正式训练另起目录、重新随机初始化。每组保存 `random_initial.pt`、`launch.json`、完整 env/agent YAML，支持检查实际初始权重和源代码版本。

W&B 项目 `tabletennis/smp`，group `scratch93_termination_seed20260911`：

| 组 | W&B |
|---|---|
| B0 | https://wandb.ai/tabletennis/smp/runs/scratch93-termination-b0-seed20260911 |
| B1 | https://wandb.ai/tabletennis/smp/runs/scratch93-termination-b1-seed20260911 |
| B2 | https://wandb.ai/tabletennis/smp/runs/scratch93-termination-b2-seed20260911 |
| B3 | https://wandb.ai/tabletennis/smp/runs/scratch93-termination-b3-seed20260911 |
| B4 | https://wandb.ai/tabletennis/smp/runs/scratch93-termination-b4-seed20260911 |
| B5 | https://wandb.ai/tabletennis/smp/runs/scratch93-termination-b5-seed20260911 |

训练日志除了 PPO/reward，还记录每类终止、头部高度、原始 SMP、GSI 刷新位置。新增低位占比与“头高 <0.85m 且角速度模长 >2rad/s”占比，排除本步刚 reset 的环境；它仅是低位运动诊断，**不是经过验证的打滚识别或成功率**。B5 记录长任务 episode 抽样占比、步数占比、长/短任务结束率。

服务器运行记录：`run_control/scratch93_termination_seed20260911_10000/launches.json`，各组 `.log` 同目录。
checkpoint：`logs/rsl_rl/scratch93_termination/B{0..5}_seed20260911_10000/`。

## 后续评测与下一轮

所有组使用相同固定 128 个倒地初态（四方向各 32）和 20 秒持续评测，只保留超时/数值保护，不按训练时的站起/低 SMP 结束。报告各方向成功率、起身时间、连续 10 秒双脚稳定站立、再次跌倒和站立运动质量。既报告名义环境，也要单独做观测噪声及域随机化评测，不能用名义成功率代替部署鲁棒性。

公平选模：各组使用相同 checkpoint 评估节点；若按验证集选 checkpoint，最终比较另外的固定测试集，避免只展示挑出的成功视频。最终配置至少补齐 3 个训练 seed。本次启动的是第一轮 6 组单 seed，不是已经完成多 seed 结论。

第二轮只改变 reset：原始 GSI、90% GSI+10% 四方向静态倒地、90% GSI+10% LAFAN get-up 低位切片，仍独立从零训练。
第三轮只改变 actor 历史：1/4/10 帧，直接拼接为 93/372/930D；部署端必须使用一致历史缓冲。第二、三轮尚不启动，待第一轮结果决定基底配置。
