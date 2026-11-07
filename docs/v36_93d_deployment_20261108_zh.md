# V36 → 93D 部署适配

**当前方案已改为台阶专项续训。** 下文“原环境保持什么”和“初始匹配训练（已停止）”记录第一版匹配实验；该任务已按用户的新要求停止。正在使用的地形、reset、位移约束和评测见文末“台阶专项版本”。两者不能混用。

## 方法和来源

沿用用户确认的 V34 gate6000 流程：去掉 actor 的三个基座线速度输入，折叠零速度对应的第一层偏置，再继续 PPO。这是投影热启动与强化学习适配，不是 teacher–student 动作蒸馏。

- 分支：`codex/v36-93d-deploy`，基于 `origin/codex/terrain-recovery-v36`。
- 原策略：W&B `tabletennis/smp/de2kit7e/model_102000.pt`。
- 原策略 SHA256：`2ebfa32eea590a40ebaec4bad237180c95d334fbe507ea7544e055ede70a2b55`。
- 选择依据：历史 V3.6.1 文档明确保留 V36@102000；相较 V36@105999，它的历史峰值功率和足滑较低。该历史结论不能替代本次新评测。
- 投影 SHA256：`71253d8706ad92bbc29bf784d9425cd91621ff3bd493c8382823027e764c28ac`。
- Actor：93D 单帧（角速度、重力方向、29关节相对位置、29关节速度、上一步29维动作），29维动作输出。
- Critic：960D，原 critic 及其归一化统计完全继承；优化器、迭代计数和环境步数从零开始。
- 网络：512/256/128 ELU；actor 观测噪声及固定动作探索标准差 0.30 保留。

投影按实际 RSL-RL 的 `(x-mean)/(std+0.01)` 运算折叠偏置。旧 V34 投影脚本省略了 epsilon；本次修正这一数值细节，并通过真实 `MLPModel` 前向验证。512组输入的最终动作最大绝对差为 `3.05176e-5`，比较的是原96D策略**线速度设零时**的输出，不代表与有真实线速度输入的策略性能等价。

## 原环境保持什么

除 actor 去除线速度外，环境配置逐项对比不变。机器人使用原 V36 训练模型，尚未迁移成部署 XML；此举用于先隔离观测适配的影响。

| 项目 | 设置 |
|---|---|
| 地形 | 平地30%、坡30%、台阶25%、碎地15% |
| 4096环境实际配额 | 1229 / 1228 / 1024 / 615 |
| 初始化 | 全部程序化接地低位；伏卧35%、仰卧25%、左右侧卧各20%；比例为随机采样概率 |
| LAFAN/GSI | 使用V7运动先验；GSI生成和刷新保留，但最终reset全部由程序化姿态覆盖并重建历史 |
| 物理/控制周期 | 5ms / 20ms |
| episode | 最长20秒；保留原站起终止、越界和仿真不稳定终止；无低SMP终止 |
| 奖励 | 原V36分阶段恢复、SMP、平面漂移、接触足滑和站姿宽度等；不新增reward |
| 随机化 | 保留原足摩擦、躯干COM、编码器偏差和观测噪声；原V36无自动push，也无后期新增的分部件质量/Kp/Kd/命令延迟随机化 |

**历史限制**：V36课程门限在带探索噪声的训练中没有晋级，原run实际停在level0。本次先保留这一匹配配置，不以修正课程同时改变任务。更高台阶的能力需要单独测，不能宣称已训练过全部5/10/15/20cm台阶。后继V3.6.1修正课程后曾退化，不直接套用。

## 初始匹配训练（已停止）

- 服务器：`gpu4090-bj`，即 `root@192.168.11.91:17535`。
- 工作区：`/root/workplace/smp-v36-93d`，复用该机原有Python环境，`PYTHONPATH`指向独立工作区。
- 正式GPU：4；配对评测GPU：1（遇占用则另选空闲卡）。
- 4096环境，每更新24步，12,000次更新，每1,000保存。
- 初始LR `1e-4`、adaptive KL、gamma .99，保留V34适配流程的PPO配置；seed `20261108`。
- W&B：https://wandb.ai/tabletennis/smp/runs/a6xbab18 ，已实际进入PPO更新。
- 日志：`run_control/train_20261108.log`；进程元数据：`run_control/train_20261108.process.json`。
- 输出：`logs/V36_93D_from102000_12k_20261108`。
- 64和4096环境各2次PPO预检通过，正式训练重新加载未训练的投影权重，不继承预检。

完成后脚本导出含归一化的93D ONNX，并核对PyTorch/ONNX输出。此时仅获得部署接口候选，仍需RoboMimic_Deploy的sim2sim验证；当前未修改部署默认策略。

## 配对评估

`scripts/evaluate_v36_93d.py` 对原96D策略、零速度投影93D策略、后续训练checkpoint使用同一地形/姿态/seed。报告平地、斜坡、台阶、碎地，分别按伏卧/仰卧/左右侧卧统计，不以训练reward挑选策略。

保留历史连续站立0.5秒指标，同时单独计算SR3：20秒内开始站立、维持3秒，观察23秒。站立条件为相对地形原点的头高≥1.10m、直立度≥.85、base速度<.5m/s、角速度<1rad/s。台阶下移后，该相对原点的高度要求仍保留，因此另看地形下降与视频，不能将其等同于所有落脚高度下的恢复。

还记录二次摔倒、越界、足滑、漂移、关节速度/力矩/功率。力矩与功率当前为控制步采样，不能称为捕获所有物理子步瞬时峰值。初步原策略检查：5cm台阶仰卧32/32达到SR3（小样本，仅验证运行链路）。

```bash
cd /root/workplace/smp-v36-93d
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl PYTHONPATH=src /root/workplace/smp/.venv/bin/python scripts/evaluate_v36_93d.py \
  --checkpoint artifacts/v36_source/model_102000.pt \
  --task Smp-Getup-Terrain-V36-G1 --levels 0 1 --num-envs 64 \
  --output logs/eval_source_64.jsonl
# 换93D权重时，task改为Smp-Getup-Terrain-V36-93D-G1。
```

## 代码验证

`PYTHONPATH=src python -m unittest discover -s tests -p test_v36_projection.py`：验证真实actor前向等价性、critic逐张量不变，以及train/play配置仅移除actor线速度。

原代码的 `scripts/play.py` 支持 `--terrain-type stairs --terrain-level 0 --terrain-reset-pose supine`，新task为 `Smp-Getup-Terrain-V36-93D-G1`。正式训练尚未完成时，不能把投影权重或预检权重当成训练好的V36部署版。

## 台阶专项版本

目标是在台阶上、台阶底部与边缘恢复，并控制相对初始位置的移动；不能把滚到远处平地后站起来当成局部恢复成功。这里的“台阶下”指台阶底部附近，不包含楼梯下方的封闭顶棚空间。

任务名：`Smp-Getup-Stairs-V36-93D-G1`。仍从 V36@102000 的原始93D投影权重启动，不继承前一短跑或预检的更新。

| 项目 | 当前设置 |
|---|---|
| 环境 | 4096；平地10%（410个）、台阶90%（3686个），无坡面/碎地 |
| 台阶形状 | 四周阶梯式平台；踏面宽30/40cm，各45%环境；地形块8×8m |
| 单级高度 | 5/10/15/20cm，从第一轮就全部覆盖 |
| 高度课程 | 台阶环境内的采样概率由55/25/15/5%，在前2000次PPO更新线性变为20/25/30/25%；每次reset重新抽取，不以站起成功率作为晋级门槛 |
| 姿态 | 伏卧、仰卧、左侧卧、右侧卧各25%；随机yaw、0.18rad朝向扰动 |
| 台阶位置 | 上部平台15%、踏面/立面附近40%、底部接近区域30%、转角15%；每区域内随机位置和朝向 |
| 关节状态 | 默认姿态附近±0.20rad生成2048个候选通过筛查后构成采样库；关节限位内，MuJoCo自碰撞穿透不超过1mm；采样后不再添加未经检查的关节扰动 |
| 接地 | 按机器人碰撞体和实际台阶box几何确定竖直位置，预留6mm几何间隙；允许身体跨不同高度的踏面 |
| episode | 20秒；取消站起提前结束，保留超时、仿真不稳定和地形越界结束；无低SMP终止 |
| 相对reset位移 | 0.40m以内不增加这项成本，超出后的平方成本权重−0.30；距reset超过1.50m提前结束，记为局部越界 |
| 站立保持 | 满足站立条件1秒后记录站立锚点；距该锚点0.15m以外的平方成本权重−0.50 |
| 其他奖励 | 保留原V36的SMP、分阶段恢复、动作/关节速度/力矩/功率和接触足滑等项；高度改为相对当前局部支撑面 |
| PPO | 初始LR 3e−5，4096环境、每更新24步、12000更新，每1000保存；actor噪声、探索std0.30保留 |
| 随机化 | 保留V36的足摩擦、躯干COM、编码器偏差；本版未新增分部件质量/Kp/Kd/延迟/push，不能称已完成真机动力学鲁棒化 |

位置类别是采样区域，不保证整个身体只落在同一个踏面上。底部与边缘样本可以跨接多级台阶，这正是需要学习的接触变化。关节库只过滤无效自碰撞，不代表已经验证了每个初始状态都可恢复。

### 验证和评测

- `scripts/audit_stair_resets.py` 检查早期/后期各256个reset，共512个。地形接触穿透未超1mm，自碰撞最深约0.49mm，几何接地间隙最小约6mm；部分环境reset不会改动其他环境的状态、原点和位移锚点。
- 4096环境、2次PPO更新预检通过；保存的actor/critic所有张量均有限。初始四姿态数量为1023/1001/1030/1042，四位置区域数量635/1633/1195/633。
- [经过检查的初始化示例](../evidence/v36_93d_20261108/stairs/reset_examples.png)：15/20cm台阶，四姿态的踏面/边缘和底部样本；这是reset截图，不是训练后恢复成功截图。
- 新任务的 `stair_sr3` 等训练日志使用episode末值，避免对“已成功”标记做时间平均；它仍是带探索和训练随机化的指标，不代替独立评测。
- 独立评测：20秒内开始连续站立，保持3秒，观察23秒。站立条件为局部支撑面以上头高≥1.10m、直立度≥0.85、base速度<0.5m/s、角速度<1rad/s。另报全程距reset最大位移≤0.75m的局部SR3、二次摔倒、足滑、位移P95和关节负载；1.5m越界后停止计入后续动作。
- 按台阶高度、姿态、位置分别报成功数/样本数，不用一个汇总平均掩盖高台阶或侧卧失败。负载仍为控制步采样；不是全部物理子步瞬时峰值。

```bash
cd /root/workplace/smp-v36-93d
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl PYTHONPATH=src /root/workplace/smp/.venv/bin/python scripts/evaluate_v36_93d.py \
  --task Smp-Getup-Stairs-V36-93D-G1 \
  --checkpoint artifacts/v36_projected/model_102000_zero_velocity_93d.pt \
  --terrain-types stairs --levels 0 1 2 3 \
  --reset-modes prone supine left_side right_side --num-envs 128 \
  --output logs/eval_stairs_multisite_projected_128.jsonl
```

此命令保留新任务的多位置接地，所有评测环境均为台阶，30/40cm踏面各半。之前的 `Smp-Getup-Terrain-V36-93D-G1` 中心reset评测不能证明这些新位置的能力。训练成功后还需部署模型下sim2sim与负载检查，不能仅凭93D接口就直接判为真机安全。

### 新任务的未续训基线检查

93D原始投影权重，15cm台阶、30/40cm踏面各半、每种姿态32个随机位置，seed=20260818：

| 姿态 | SR3 | 全程位移≤0.75m的局部SR3 | 单回合最大力矩P95 | 单回合最大关节功率P95 |
|---|---:|---:|---:|---:|
| 伏卧 | 10/32 | 5/32 | 103.7Nm | 439.7W |
| 左侧卧 | 11/32 | 6/32 | 115.5Nm | 545.0W |

这只是评测链路与困难样本检查，不是最终性能结论。样本小，负载为控制步采样；位移超过1.5m后该样本被标记越界并隔离，位移统计上限截断于1.5m。结果记录在 `evidence/v36_93d_20261108/stairs/eval_multisite_stairs_projected_smoke.jsonl`。

### 当前运行记录

- 服务器 `gpu4090-bj`（root@192.168.11.91:17535），GPU4，工作区 `/root/workplace/smp-v36-93d`。
- 正式输出 `logs/V36_93D_stairs_multipose_12k_20261108_offline`。
- 进程记录 `run_control/stairs_train_20261108_offline.process.json`，日志 `run_control/stairs_train_20261108_offline.log`。
- W&B创建run两次返回HTTP403，均发生在第一次PPO更新之前；保留失败日志，没有继承这些失败尝试的权重。正式训练使用 `WANDB_MODE=offline`，不代表目前W&B网页可见，网络恢复后可用 `wandb sync` 同步对应offline目录。
- 原匹配任务 `a6xbab18` 已停止；其他任务未停止。
- 离线run ID `j3q4gsdv`，目录 `wandb/offline-run-20261108_004147-j3q4gsdv`，已进入PPO更新。同步命令：`/root/workplace/smp/.venv/bin/wandb sync wandb/offline-run-20261108_004147-j3q4gsdv`（在该工作区执行，需网络恢复）。
