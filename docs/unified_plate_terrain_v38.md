# V3.8：可部署的压板 + 复杂地形统一恢复

## 目标

V3.8 用同一个 actor 同时解决：

1. 无压板的平地、斜坡、台阶（含边缘）和粗糙地面恢复；
2. 仰卧或俯卧时，在平地或台阶中心被 0.90 m × 0.64 m 压板约束后，先脱困再站立；
3. 真机不可获得的状态不进入 actor。

第一版不在斜坡、粗糙地面或台阶边缘放压板。这些组合会让水平、仅竖直滑动的压板直接撞地，难以区分策略失败和场景建模错误。无压板 episode 仍覆盖全部复杂地形。

## 可部署观测

actor 的单帧顺序保持原来的 96 维：

- base linear velocity：3 维，恒为 0；
- IMU angular velocity：3 维；
- projected gravity：3 维；
- joint position：29 维；
- joint velocity：29 维；
- previous action：29 维。

每项分别保存 4 帧，按 oldest → newest 展平后再拼接，actor 总维度为 384。压板位姿、压板质量、地形类型、台阶等级、reset cohort、接触真值和仿真 base linear velocity均不进入 actor。critic 保持 10 帧 960 维。

真机端不能简单把完整 96 维帧重复四次后拼接；必须使用同样的 term-wise 布局：

```text
base_lin_vel[4,3], base_ang_vel[4,3], projected_gravity[4,3],
joint_pos[4,29], joint_vel[4,29], previous_action[4,29]
```

## reset 分布

训练时保持 V3.7 四类姿势及台阶位置 cohort：

- prone / supine / left-side / right-side；
- stair center / near-edge / straddle / lower-tread；
- 平地 / 斜坡 / 台阶 / 粗糙地面。

压板概率为 0.90，但还需同时满足：

- reset 是 prone 或 supine；
- terrain 是 flat 或 stairs；
- stair cohort 是 center（平地 cohort 也为 0）。

因此压板 episode 约占总训练分布的 20%–25%，其余样本持续回放复杂地形恢复，降低灾难性遗忘。

事件顺序必须是：

```text
mixed fall reset
→ stair edge location sampling
→ local terrain grounding
→ plate placement (不再二次平地 grounding)
→ recovery-stage reset
```

## 两组配对训练

### A. 快速验证：V3.6.3 warm start

先将 V3.6.3 Deploy 的 96 维 actor 迁移到 384 维。旧权重只连接每个 term 的最新一帧，旧帧连接权重初始化为 0；normalizer 统计复制到四帧。这样迁移前后的第一层输出仅有浮点求和误差，随后策略可以学习利用历史。

```bash
uv run scripts/adapt_checkpoint_actor_history.py \
  logs/rsl_rl/smp_getup_terrain_v363_deploy_g1/baseline_v363_deploy_strict_102049/model_102049.pt \
  logs/rsl_rl/smp_getup_plate_terrain_v38_deploy_g1/v363_seed_history4/model_00000.pt \
  --force
```

训练：

```bash
uv run scripts/train.py Smp-Getup-Plate-Terrain-V38-Deploy-G1 \
  --env.scene.num-envs 4096 \
  --agent.resume True \
  --agent.load-run v363_seed_history4 \
  --agent.load-checkpoint model_00000.pt \
  --agent.max-iterations 2000 \
  --agent.run-name v38_unified_warmstart_102049
```

### B. 论文对照：from scratch

完全相同的 task、384 维 actor、reward、reset 和 seed 范围，不加载任何 RL checkpoint：

```bash
uv run scripts/train.py Smp-Getup-Plate-Terrain-V38-Deploy-G1 \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 4000 \
  --agent.run-name v38_unified_from_scratch
```

“from scratch”指 RL actor/critic 从随机初始化开始；SMP 的 recovery prior 仍使用同一冻结 prior。若连 prior 也重新训练，那是另一项独立 ablation，不能与本实验混在一起。

## 快速健康门

warm-start 前 100–200 iterations 先检查：

- 无 NaN、solver overflow 或 broadphase overflow；
- escape invalid contact / invalid setup 没有系统性升高；
- plate episode 比例符合约 20%–25%；
- 平地与 L0 terrain recovery 不发生明显崩塌；
- escape completion 开始上升；
- max joint speed、torque 和 power 不显著超过 V3.6.3 / V3.4。

只有 warm-start 环境通过物理健康门，才让 from-scratch 长时间运行。若联合训练互相干扰，先调整 sampling 或分阶段 curriculum，不改变 actor 观测与最终统一 task。

## 播放

复杂地形、无压板：

```bash
uv run scripts/play.py Smp-Getup-Plate-Terrain-V38-Deploy-G1 \
  --checkpoint-file <checkpoint> \
  --terrain-type stairs --terrain-level 1 \
  --terrain-reset-pose prone --terrain-edge-cohort near-edge \
  --escape-obstacle False --auto-disturbances False \
  --num-envs 1 --viewer native --no-terminations True
```

台阶中心、俯卧、压板：

```bash
uv run scripts/play.py Smp-Getup-Plate-Terrain-V38-Deploy-G1 \
  --checkpoint-file <checkpoint> \
  --terrain-type stairs --terrain-level 1 \
  --terrain-reset-pose prone --terrain-edge-cohort center \
  --escape-obstacle True --auto-disturbances False \
  --num-envs 1 --viewer native --no-terminations True
```

play 默认关闭压板，必须用 `--escape-obstacle True` 显式启用，避免复杂地形测试被压板混入。

## 评估矩阵

统一 checkpoint 要分别报告：

- plate-free：4 terrains × L0/L1 × 4 poses；
- stair edge：center / near-edge / straddle / lower-tread；
- plate：flat/stairs-center × prone/supine × 4/8/12 kg；
- safety：peak penetration、peak contact force、joint speed、torque、power、stand stance width；
- ablation：warm-start 与 from-scratch，同样预算、同样 seeds。

最终选择 checkpoint 不能只看总 reward，必须同时满足无压板恢复保留、压板脱困成功和物理有效性三条门槛。

## 2026-08-26 启动记录

- 分支：`codex/unified-recovery-v38`
- 实现提交：`de45c23`
- 服务器工作区：`/mnt/workspace/user/luyidan/smp-v38-unified`
- terrain-seed warm-start W&B：`tabletennis/smp/ysl4x4xm`
- plate-seed warm-start W&B：`tabletennis/smp/uejfq9qh`
- from-scratch W&B：`tabletennis/smp/1yw7b3os`
- 三组均为 4096 environments、每 100 iterations 保存。

terrain-seed 初始物理健康门（约 iteration 22）：无 unstable dynamics；
压板峰值穿透约 1 mm，峰值接触力约 128 N；terrain curriculum
success 约 0.78。此时 escape completion 约 0.8%，仅表示旧地形 seed
尚未学会新压板任务，不作为最终成功率。

冻结的 `model_100.pt` 审计表明该路线没有快速学到压板：

- flat/stairs、L0/L1、prone/supine 的 8 个无压板 case 平均成功率 58.8%；
- 8 kg 全覆盖压板下 escape-and-stand 为 0；
- invalid plate episode 为 21%。

训练内 terrain success 随后下降且 escape 仍接近 0，因此在保存
`model_200.pt` 后提前停止，不将它作为候选部署模型。两个 checkpoint
保留在服务器 `baselines/G1_Recovery_Below_Block/v38/audit/`。

V3.4 plate seed 迁移到完全相同的 384 维 deploy actor 后，未微调即得到：

- 8 kg 压板 escape-and-stand 60.95%，valid episode 中为 80%；
- flat L0/L1 的 prone/supine 为 100%；
- stairs L0 为 3.1%–12.5%，stairs L1 为 0；
- 压板 invalid episode 23.3%，关节速度和功率仍明显偏高。

这说明两个 seed 能力互补。后续主 warm-start 改为 plate seed，在统一
任务中补复杂地形和安全性；from-scratch 保留为同任务对照。
