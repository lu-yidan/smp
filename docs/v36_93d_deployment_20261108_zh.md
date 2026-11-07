# V36 → 93D 部署适配

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

## 本次训练

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
