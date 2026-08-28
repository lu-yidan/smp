# Recovery-Core R1：收敛后的 SMP 平地恢复主线

## 定位

Recovery-Core 是论文和真机恢复的独立主线，不包含压板和复杂地形。它回答一个单一问题：

> 在只有可部署本体观测的条件下，SMP motion prior 与 GSI 能否让 G1 从任意倒地姿态
> 安全、自然地恢复并继续稳定站立？

Task：`Smp-Recovery-Core-R1-G1`

## 固定范围

- 纯平地；
- 物理场景中不存在压板实体、压板传感器、压板 reward 或压板 termination；
- 不使用 LAFAN 里程碑 reset；
- 60% GSI，40% 程序化俯卧、仰卧、左侧卧和右侧卧；
- 同一个 recovery policy 负责倒地恢复和站立保持，不切换 balance policy；
- episode 成功后不提前终止。

Actor 接口保持可部署：

- 4 帧历史，384 维；
- `base_lin_vel` 固定为零；
- IMU 角速度和 projected gravity；
- 关节位置、关节速度与上一时刻动作；
- 无接触真值、reset 类型、GSI 标签、地形类型或仿真线速度。

## SMP 设置

V3.8.7.2--V3.8.7.4 为了帮助随机 actor 探索，曾把 `task_smp_product`
权重降至 0.05。Recovery-Core 将其恢复到 0.50，使 SMP 成为实质性目标，同时保留
`smp_floor=0.03`，避免策略落在 prior 外时完全失去恢复梯度。

LAFAN 里程碑只属于 bootstrap 诊断，不属于 Recovery-Core 最终方法。

## Bootstrap 审计与正式起点

V3.8.7.4 的 `model_2997`、`model_3250` 和最终 `model_3496` 已在
512 个纯程序化环境、四种姿态、500 steps 的统一协议下评测。所有 checkpoint 的
有序阶段完成率均为 0%；`model_3250` 和 `model_3496` 的四姿态正式成功率也均为
0%。因此 V3.8.7.4 被记录为失败的 bootstrap 诊断，不能作为 Recovery-Core 起点。

正式起点改为真机已验证的 V3.3 balanced `model_95000.pt`：

- source SHA-256:
  `38063879c144bd29af8e792bb7547b0e6c99e4043ba9b4c3c08219dab16ef81a`；
- 使用 `scripts/adapt_checkpoint_actor_history.py` 将 actor 从 96 维单帧扩展为
  384 维四帧，critic 保持 960 维；
- adapter first-layer equivalence 最大绝对误差为 `9.537e-06`；
- optimizer 重置，checkpoint iteration 重置为 0；
- adapted checkpoint SHA-256:
  `7f774491893a3a95ab02e2cb712f5ffc1d5e70342450958d86d972096aef69d5`。

当前正式训练：W&B `tabletennis/smp/2jlopob8`，4096 environments，
seed 3881，GPU 7，500 updates，每 250 updates 保存。

适配后 seed 的冻结基线（每姿态 512 environments、500 steps）：

| 姿态 | 正式成功率 | 有序阶段成功率 | 最大关节速度均值 | 最大功率均值 |
|---|---:|---:|---:|---:|
| 俯卧 | 92.97% | 88.67% | 16.32 rad/s | 383.7 W |
| 仰卧 | 99.41% | 84.96% | 11.99 rad/s | 383.4 W |
| 左侧卧 | 99.22% | 94.53% | 15.18 rad/s | 304.2 W |
| 右侧卧 | 98.24% | 88.09% | 15.08 rad/s | 455.8 W |

Recovery-Core checkpoint 必须保持上述成功率，同时显著降低关节速度、功率和脚步位移；
仅降低动作幅度但丢失恢复能力的模型不予接受。

## Reset 路由

每个环境只选择一种恢复初始化：

- 60%：由 `pretrained_getup_lafan_route_v7.pt` 的 GSI pool 生成；
- 40%：程序化倒地，四种姿态等概率。

Recovery-Core 没有压板随机变量，因此不会出现“先定位压板、再覆盖机器人姿态”的顺序错误。

## 实验结构

后续实验固定为四条线，不再继续叠加 V3.8.x 子版本：

1. Recovery-Core：本文主恢复策略；
2. Recovery-Ablation：移除 SMP reward 或 GSI 的对照；
3. Plate-Escape：仅倒地 cohort 带压板；
4. Terrain-Recovery：台阶、斜坡、粗糙地面，无压板。

只有前三条独立门槛通过后，才建立使用互斥 cohort 的 Unified 模型。

## 评测

统一对俯卧、仰卧、左侧卧和右侧卧报告：

- 正式恢复成功率和有序阶段完成率；
- 恢复时间中位数与 P90；
- 最大关节速度、力矩和功率；
- 脚步位移与腿部外展；
- 手、膝支撑比例；
- 站立后再次跌倒率。

Bootstrap checkpoint 必须在纯程序化测试环境中评测，禁止 LAFAN milestone 覆盖测试姿态。

## R1 健康门与 R2 from-scratch 长训

R1 的 500 updates 不是收敛训练，而是进入数千轮训练前的健康门。冻结评测显示：

| checkpoint | 四姿态平均正式成功率 | 四姿态平均有序阶段成功率 | 结论 |
|---|---:|---:|---|
| adapted model 0 | 97.46% | 89.06% | 真机 V3.3 起点，动作较剧烈 |
| R1 model 250 | 99.90% | 76.12% | 保留恢复能力且明显降低速度和功率 |
| R1 model 499 | 99.90% | 13.82% | 学会绕过坐/蹲阶段的站立捷径 |

健康门发现原有稳定站立 reward 可以绕过有序阶段。因此 Recovery-Core R2 将
`scratch_stable_stand` 改为只有完成有序恢复阶段后才生效，并惩罚“已经稳定站立但
尚未完成有序阶段”的状态。

正式 R2 主实验不加载 V3.3、R1 或其他 PPO checkpoint，actor/critic 与 observation
normalizer 均从随机初始化开始；SMP prior 与 GSI generator 仍保留，因为它们是方法的
组成部分而不是 PPO warm start。训练使用 4096 environments、初始学习率 `3e-4`、
20000 updates，并每 250 updates 保存一次。500/1000 updates 仅为中途诊断，
不能作为最终结论；最终 checkpoint 必须通过固定四姿态成功率、有序阶段、关节速度、
功率和脚步位移评测。

实际长训于 2026-08-28 启动：

- W&B：`tabletennis/smp/u2hmbkmr`；
- server run：`2026-08-28_23-59-17_recovery_core_r2_ordered_scratch_20k_4096_seed3883`；
- GPU 7，seed 3883，4096 environments，20000 updates；
- 启动时未设置 resume、run path 或 checkpoint，确认为 PPO from scratch。

较早启动的 5000-update run `hn29fn14` 已主动终止，由上述 20k run 替代。
