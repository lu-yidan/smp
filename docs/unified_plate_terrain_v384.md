# V3.8.4 左右对称的统一压板与复杂地形恢复

## 诊断结论

V3.8.3 将右侧卧压板训练占比从约 4.5% 提高到 12.26%，但 model_12 和
model_24 的右侧稳定恢复率仍为 0%，并且 model_24 已出现整体退化，因此按
早停规则停止普通 oversampling。

冻结 V3.8.2 H4 model_25 的配对审计：

| case | left side | right side |
|---|---:|---:|
| flat、无压板 | 100% | 100% |
| 8 kg 压板、原策略 | 24.76% | 0% |
| 8 kg 压板、镜像策略 | - | 37.14% |

左右压板的 initial covered geom median 都是 17，invalid rate 接近；但原策略
左侧 hand-support median 是 238 steps，右侧只有 7 steps。将右侧本体观测
镜像成左侧、再把动作镜像回右侧后，右侧 hand-support median 增至 292
steps，39/105 个环境稳定站立。max penetration 从 12.86 mm 降至 8.14 mm，
joint-speed p95 从 16.05 降至 14.03 rad/s。

因此右侧 0% 的主要原因是策略非对称，而不是 reset、压板尺寸或不可恢复的
接触几何。

## 方法

任务：`Smp-Getup-Plate-Terrain-V384-Symmetric-Deploy-G1`。

环境、采样和观测完全复用已审计的 V3.8.2 H4：

- actor：4 x 96 = 384 维部署安全本体观测；
- critic：10 x 96 = 960 维；
- 压板仅出现在 flat 和 stairs-center；
- actor 无真实 base linear velocity、reset type、plate state、contact truth、
  terrain label 或 height map。

训练使用 RSL-RL 原生 symmetry extension：

1. 每个 PPO mini-batch 追加 sagittal-mirrored observation/action；
2. mirror loss 强制 `pi(M(o))` 接近 `M(pi(o))`；
3. polar vector、axial angular velocity、29 个关节左右置换和 roll/yaw
   符号都按 G1 坐标定义变换；
4. actor 与 critic 的 4/10 帧历史使用同一变换。

训练完成后仍导出一个普通 actor；真机无需 mirror wrapper、reset label 或
第二次网络前向。

## 第一轮短训

- seed：V3.8.2 H4 model_25，optimizer 清空；
- 4096 env，seed 384；
- 25 updates，save interval 12；
- learning rate 5e-7；
- data augmentation：enabled；
- mirror loss coefficient：0.05（初始未加权 symmetry loss 约 3.48）。

先冻结评测 model_12/model_24 的四姿态压板。如果 right-side 明显变为非零且
prone/supine/left 没有明显退化，再做 flat/stairs L0/L1 保持评测。

## from-scratch 路线

V3.8.4 验证成功后，再启动 policy scratch：随机 actor/critic、保留现有
motion prior、保持同一 384 维部署观测与 symmetry 配置。训练按 flat 基础
恢复、flat/stairs-center 压板、edge/slope/rough L1 三阶段推进，并保留前一
阶段 replay floor。这样能公平回答“性能来自 warm start 还是方法本身”，同时
避免把 prior 重训混入第一轮变量。
