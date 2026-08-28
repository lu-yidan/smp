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
CUDA_VISIBLE_DEVICES=4 uv run scripts/train.py \
  Smp-Getup-Plate-Terrain-V386-Scratch-S1-Deploy-G1 \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 600 \
  --agent.run-name v386_scratch_s1_health_gate
```

命令不包含 `--agent.resume` 或 checkpoint 参数，所以 actor/critic 确实随机
初始化。600 updates 是健康门，不是最终训练预算。

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
