# V3.8.7.4 GSI + LAFAN 里程碑恢复课程

## 目标

V3.8.7.1--V3.8.7.3 证明：仅靠随机初始化的策略和全程倒地 reset，虽然能够学到抬高躯干，
但很难探索出完整的“倒地→跪/坐→蹲→稳定站立”长时序过程。V3.8.7.4 恢复原始 SMP
使用的 Generative State Initialization（GSI），并增加人工审核过的 LAFAN 关键阶段 reset，
用于快速验证可部署观测下是否能够学会完整恢复。

这不是加入部署特权信息。GSI、轨迹阶段标签和 reset 类型只在训练初始化及日志中使用，
不会进入 actor observation。

## 任务与观测

- Task: `Smp-Getup-Plate-Terrain-V3874-GSI-Milestone-Deploy-G1`
- Experiment: `smp_getup_plate_terrain_v3874_gsi_milestone_deploy_g1`
- Actor: 4 帧历史，每帧 96 维，共 384 维。
- `base_lin_vel` 固定为零；其余为可部署的机身角速度、projected gravity、关节位置、
  关节速度和上一时刻动作。
- 不使用接触真值、地形类别、高度图、压板状态、reset 标签或仿真线速度。
- 站立后不终止，也不切换 balance policy；仍由同一个 recovery policy 保持站立。

## Reset 分布

每次 reset 先执行 GSI，然后由后续事件覆盖部分环境：

- 约 40% GSI：从 `pretrained_getup_lafan_route_v7.pt` 生成的状态窗口池采样；
- 约 40% 程序化倒地：俯卧、仰卧、左侧卧、右侧卧；
- 约 20% LAFAN 里程碑：从已审核的轨迹窗口直接初始化。

里程碑内部采样权重：

- 跪姿/半跪姿 45%；
- 蹲姿 35%；
- 站立 20%。

数据来源：

- `datasets/csv/getup_lafan_prone_routes_v7/manifest.json`
- `datasets/npz/getup_lafan_prone_routes_v7/*.npz`
- `datasets/pretrain_ckpt/pretrained_getup_lafan_route_v7.pt`

当前审核库共有 278 个跪姿窗口、406 个蹲姿窗口和 220 个站立窗口；每个窗口为
10 帧、59 维、50 Hz。完整窗口用于填充 SMP 历史，最后一帧用于写入机器人物理状态。

## 奖励修正

- 使用双边膝关节角度区间约束，避免一直保持深蹲，也避免直膝抬躯干的捷径；
- 将阶段占用奖励降至 0.20，避免停在某一阶段刷分；
- 新增一次性阶段转换奖励，鼓励依次到达跪/坐、蹲、站立；
- 稳定站立权重提高至 4.00；
- 保留关节速度、功率、力矩、动作平滑和竖直速度约束。

## 快速验证计划

1. 先从 V3.8.7.3 `model_2997.pt` 续训，4096 环境、单 seed、每 250 update 保存。
2. 固定 checkpoint 分别评测俯卧、仰卧、左右侧卧，记录恢复成功率、阶段完成率、
   恢复时间、最大关节速度、最大功率和脚步位移。
3. 若续训能稳定完成阶段转换，再用同一课程从随机 actor 开始训练，作为论文中
   “GSI/里程碑是否只是微调技巧”的对照。
4. 平地门槛通过后才逐步加入压板和复杂地形，避免同时改变探索和接触动力学。

## 通过门槛

- 四种倒地姿态均出现非零并持续提升的正式恢复成功率；
- 跪/坐→蹲→站三次阶段转换均能发生，不能仅抬高躯干；
- 不依赖真实 base linear velocity；
- 相比 V3.8.7.3 不增加危险的最大关节速度、功率和明显小碎步；
- 同一策略在站起后能继续稳定站立，不依赖策略切换。
