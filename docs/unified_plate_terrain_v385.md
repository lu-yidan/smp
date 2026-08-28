# V3.8.5 部署侧姿态规范化与分阶段 scratch 计划

## 为什么没有继续普通微调

V3.8.3 增加右侧卧压板 replay 后，model_12/model_24 的右侧稳定脱困率仍为
0%。V3.8.4 又加入 RSL-RL sagittal data augmentation 和 mirror loss；训练
symmetry loss 从约 3.48 降到 1.32，但 model_12/model_24 的右侧仍为 0%，
总体分别只有 32.5% 和 31.0%。短程软约束没有改变策略的行为分支，因此不替换
V3.8.2 H4 model_25。

冻结诊断已经证明同一个 V3.8.2 actor 在显式镜像后能完成右侧脱困。V3.8.5
把这个诊断改成部署安全的确定性状态机，而不是读取 reset type。

## 部署适配器

评测开关：`--canonicalize-right-side`。

适配器只读取 actor 已有的 4 帧 projected gravity：

1. 跌倒恢复开始后的前 8 个 control steps 为分类窗口；
2. 若任一帧 body-frame gravity-y >= 0.65，则判定为右侧卧；
3. 将 observation 映射到左侧规范坐标系，单次 actor inference 后把 action
   映射回右侧；
4. 本次恢复全程锁存，避免翻滚经过 prone/supine 时切换动作 convention；
5. 非右侧姿态不改变 observation 或 action。

它不使用 base linear velocity、reset label、plate state、contact truth、terrain
label 或 height map，也不增加网络前向次数。真机版需要由 recovery supervisor
在“由站立进入跌倒”时打开分类窗口，并在重新稳定站立后清除锁存。

## 三个冻结 seed 的 8 kg 压板结果

512 environments、750 steps、四姿态均匀目标采样。active 数量受严格物理接触
与 setup validity 筛选影响。

| seed | active | overall | prone | supine | left | right | force max | penetration max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260814 | 200 | 45.50% | 48.98% | 51.02% | 36.67% | 47.62% | 10.82 kN | 18.22 mm |
| 20260815 | 192 | 38.54% | 37.25% | 50.98% | 27.91% | 36.17% | 8.12 kN | 19.98 mm |
| 20260816 | 199 | 33.17% | 30.77% | 51.85% | 22.58% | 27.27% | 2.73 kN | 14.48 mm |

分类数分别为 42、47、44，与每个 seed 的 right-side active 数完全一致，没有把
后续翻滚的 prone/supine/left 样本误切换。右侧从原策略的 0% 变成
27.27%--47.62%，说明规范化方向正确；但 seed 方差、峰值接触力和接近 20 mm
的最坏穿透仍要求更多安全审计。因此它是快速部署候选，不是最终硬件安全结论。

评测示例：

```bash
uv run scripts/evaluate_escape_checkpoint.py \
  --checkpoint logs/rsl_rl/smp_getup_plate_terrain_v382_h4_deploy_g1/\
2026-08-27_19-20-32_v382_h4_paired_from_v38_model100/model_25.pt \
  --task Smp-Getup-Plate-Terrain-V382-H4-Deploy-G1 \
  --num-envs 512 --steps 750 --seed 20260814 --device cuda:0 \
  --plate-mass-kg 8 --reset-pose all --canonicalize-right-side
```

保存的 checkpoint：

- server：`baselines/G1_Recovery_Below_Block/v385/checkpoints/`
  `v385_canonical_v382_h4_model25.pt`；
- local：`/home/d080/workspace/G1_Recovery_Below_Block/checkpoints/v385/`
  `v385_canonical_v382_h4_model25.pt`；
- SHA-256：`bc17f39dfc6ce797fabd051b6a66183182cf89c94c4a942dd1398b5abaccae8b`。

## from-scratch 的正确对照

朴素 policy scratch 已经训练过：W&B `tabletennis/smp/1yw7b3os`，
`model_1199.pt`。它随机初始化 actor/critic、保留冻结 V7 motion prior，但直接
进入统一压板和复杂地形任务；8 kg 压板及 flat/stairs L0/L1 冻结成功率均为
0%。因此不重复相同训练。

下一轮采用同一个 384 维 actor、同一个冻结 prior、最终仍为单一统一策略的三阶段
curriculum：

1. S1：flat 为主、少量 slope/stairs/rough L0，加无压板和轻/偏置压板，学习
   安全起身、手部支撑和初步横向脱困；
2. S2：flat/stairs-center、完整质量与覆盖课程，并保留 S1 replay floor；
3. S3：stairs edge、slope、rough 和 L1，同时保留 clean/center cohort。

每阶段使用独立的 frozen plate/terrain/safety gate。只有 S1 同时出现无压板恢复和
非零压板脱困，才投入长时间 S2；保存间隔为 100--250 updates，避免磁盘浪费。
