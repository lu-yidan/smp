# V3.8.6 分阶段统一策略 policy scratch

## 实验问题

V3.8 的朴素 from-scratch control（W&B `tabletennis/smp/1yw7b3os`）将随机
actor/critic 直接放入完整统一任务，训练 1200 updates 后在 8 kg 压板和
flat/stairs L0/L1 的冻结成功率仍均为 0%。V3.8.6 检验失败是否来自 curriculum，
而不是“统一单 actor 本身不可学习”。

本实验只对 policy scratch：actor/critic 随机初始化，继续使用冻结的 V7 LAFAN
route motion prior。重训 prior 属于单独 ablation。

## S1 配置

任务：`Smp-Getup-Plate-Terrain-V386-Scratch-S1-Deploy-G1`。

- actor：4 x 96 = 384 维部署本体观测；
- critic：10 x 96 = 960 维；
- actor 的 base linear velocity 恒为零；
- 无 reset label、plate state、contact truth、terrain label 或 height map；
- terrain：flat/slope/stairs/rough = 75%/8%/12%/5%，全为 L0；
- pose：沿用能力平衡的 prone/supine/left/right 采样；
- plate probability by procedural pose：35%/35%/25%/25%；
- plate 仅在 flat/stairs center 出现，并沿用 V3.3 的轻质量、露边课程；
- symmetry augmentation 开启，mirror loss coefficient 0.02；
- PPO learning rate 3e-4，checkpoint 每 100 updates。

因此从 S1 开始就同时出现压板与四类地形，最终仍是一个 actor；“分阶段”只改变
训练分布和难度，不拆成 plate policy 与 terrain policy。

## 启动命令

```bash
CUDA_VISIBLE_DEVICES=7 uv run scripts/train.py \
  Smp-Getup-Plate-Terrain-V386-Scratch-S1-Deploy-G1 \
  --env.scene.num-envs 16384 \
  --agent.max-iterations 600 \
  --agent.run-name v386_scratch_s1_16384_health_gate
```

命令不包含 `--agent.resume` 或 checkpoint 参数，所以 actor/critic 确实随机
初始化。600 updates 是健康门，不是最终训练预算。

实际运行：W&B `tabletennis/smp/0erhlfpq`，seed 386。16384 environments 每个
update 产生 16384 x 24 = 393216 条真实 transition；因此 model_100 已约为
4096-environment 训练的 model_400 采样量。checkpoint 仍在 100/300/599 做冻结
评测，但横向比较同时报告 environment transitions，不能只比较 update 编号。

## S1 晋级门槛

冻结 model_100/300/599，至少满足：

1. flat 无压板四姿态平均稳定恢复率明显非零，并持续上升；
2. 4--8 kg 课程外冻结压板出现非零 escape-and-stable-stand；
3. 左右侧均非零，不能用 aggregate success 掩盖单侧 0%；
4. L0 slope/stairs/rough 至少出现恢复信号；
5. penetration < 20 mm，invalid rate、joint speed、power 没有系统爆炸；
6. 训练无 NaN、unstable dynamics 或 broadphase overflow。

若 model_300 仍在所有冻结 case 为 0%，先停训检查 reward gating/episode length，
不盲目延长到数千 updates。通过后才从所选 S1 checkpoint resume 到 S2。

## 后续阶段

- S2：提高 plate cohort 和完整质量/覆盖课程，主训 flat/stairs center，同时保留
  clean、slope 和 rough replay floor；
- S3：加入 stairs edge、L1 slope/stairs/rough，并保留 S1/S2 cohort；
- 最终与 warm-start V3.8.2、V3.8.5 canonical adapter、朴素 scratch 使用相同
  frozen evaluation protocol 对比。

## Smoke test

本地 RTX 4090、64 environments、1 update 已通过：actor 384、critic 960，
随机策略 mean reward -0.36，symmetry loss 0.0189，无 NaN 或 solver overflow。

## 16384-environment S1 结果（拒绝）

W&B：`tabletennis/smp/0erhlfpq`。600 updates 在 GPU7 用时约 1 小时 51 分，
约 11.2 秒/update、3.5 万 steps/s；总 rollout 约 2.36 亿 transitions。保存了
model_0/100/200/300/400/500/599。

训练末期 mean reward 为 0.55，symmetry loss 约 0.0001，但关键训练指标为：

- `stable_stand = 0`；
- `recovery_stage_complete = 0`；
- `Curriculum/terrain_levels/stand_success = 0`；
- `escape_completion = 0`；
- `recovery_stage` 仅 0.006。

正 reward 主要来自 recovery initiation、prone support route 和很小的
task/SMP product，是“低位支撑但不起身”的局部最优，不能解释为 recovery 成功。

冻结评测：

| checkpoint | flat 无压板四姿态均值 | 8 kg 压板 | invalid | penetration max | power mean |
|---|---:|---:|---:|---:|---:|
| model_100 | 0% | 0% | 62.96% | 24.01 mm | 82.95 W |
| model_300 | 0% | 0% | 22.22% | 26.51 mm | 93.79 W |
| model_599 | 0% | 0% | 17.78% | 21.55 mm | 122.76 W |

model_599 在更接近早期课程的 6 kg、longitudinal/lateral offset
0.18/0.22 m 条件下仍为 0%。它的 hand-support median 达 602 steps、invalid
降到 3.70%，但 escape separation median 只有 0.324 m，仍无完整脱困或站立。

结论：S1 已完成但不晋级，不保存部署副本。16384 environments 增加了每次
rollout 的覆盖，却没有增加每 update 的 PPO optimizer step 数；当前 2 epochs x
4 minibatches 只有 8 次梯度更新/update，巨大 batch 不能代替足够的策略优化。

下一轮先增加 S0：flat、无压板、同一 384 维 deploy actor 和冻结 V7 prior，
只在四姿态 stable stand 明显非零后逐步加入轻压板与 L0 terrain。同时重新配平
num mini-batches/learning epochs，使 16K rollout 不再形成过大的单个 minibatch。
