# V3.8.2 历史长度与能力保持对照实验

## 决策目的

V3.8.1 model_200 的四姿态压板成功率从 V3.8 seed 的 33.50% 降到
19.00%，而地形均值只增加 1.36 个百分点。根因首先是采样稀释，不足以
证明 4-frame 观测有问题。因此 V3.8.2 分两步控制变量：

1. 用按姿态独立的压板概率保留旧能力；
2. 从同一个 V3.8 model_100 分别训练 4-frame 与 10-frame，仅比较历史长度。

任务：

- `Smp-Getup-Plate-Terrain-V382-H4-Deploy-G1`
- `Smp-Getup-Plate-Terrain-V382-H10-Deploy-G1`

## 观测

单帧仍是 96 维可部署本体感知：zero base linear velocity、IMU angular
velocity、projected gravity、joint position、joint velocity、previous action。
不加入地形标签、压板状态、reset type、接触真值或真实 base linear velocity。

- H4 actor：384 维，约覆盖最近 60--80 ms；
- H10 actor：960 维，约覆盖最近 180--200 ms；
- critic 两组都保持原有 10-frame 训练特权历史。

H10 checkpoint 将 H4 每个 observation term 的四帧权重右对齐复制到最新
四帧，新增六帧权重置零。因此适配后初始策略与 H4 数值等价，差异来自后续
训练学到的时间信息，而不是随机重置 actor 第一层。

## 采样修复

pose weights 仍为 prone/supine/left/right = 2.0/1.5/1.0/1.0。压板条件
概率改为：

| reset pose | conditional plate probability |
|---|---:|
| prone | 0.90 |
| supine | 0.90 |
| left side | 0.65 |
| right side | 0.65 |

预计原有 prone/supine 压板 cohort 约 24%，接近 V3.8 的约 26%；新增
side plate cohort 约 10%，总压板 episode 约 34%。压板仍只在 flat 和
stairs-center 生效。

4096-env、seed 382 的实际 reset 审计：

- 总压板 1292/4096（31.54%）；
- prone/supine 压板 919/4096（22.44%），接近旧能力保留目标；
- left/right side 压板 373/4096（9.11%）；
- flat 压板 824，stairs-center 压板 468；
- slope、rough 和非中心 terrain cohort 均为 0 个压板。

因此采样修复达到了“保留旧 cohort 并加入约 9% 侧卧难例”的实际目标。

## 训练与早停

两组均：

- 从同一个 V3.8 `model_100.pt` 开始；
- 4096 env；
- 相同 seed；
- learning rate 1e-6；
- 只训练 50 updates；
- 每 25 updates 保存。

为保证优化器状态也完全配对，H4 和 H10 都先通过同一适配器生成 seed；H4
保持 4 帧、仅清空旧 optimizer state，H10 右对齐扩展至 10 帧：

    uv run scripts/adapt_checkpoint_actor_history.py \
      /path/to/v38_model_100.pt /path/to/v38_model_100_h4_reseed.pt \
      --history-length 4 --learning-rate 1e-6

    uv run scripts/adapt_checkpoint_actor_history.py \
      /path/to/v38_model_100.pt /path/to/v38_model_100_h10.pt \
      --history-length 10 --learning-rate 1e-6

两份 seed 的 iteration 都重置为 0，actor 初始输出均与 V3.8 checkpoint 等价。

## 冻结评测

对 seed、H4-25/H4-50、H10-25/H10-50 使用完全相同的 reset seed 和条件：

- 8 kg plate，prone/supine/left/right 分别报告；
- flat/stairs L0/L1，prone/supine/side 分别报告；
- rough/slope 作为保持能力审计；
- penetration、peak contact force、joint speed、torque、power；
- stand foot spacing 和 foot speed。

只有当 H10 在压板成功率或困难地形上有可重复提升，且安全指标不恶化，才
保留 10-frame。否则继续使用更简单、部署延迟更小的 4-frame。
