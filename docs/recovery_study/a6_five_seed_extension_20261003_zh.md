# A6_no_alpha 同配置：扩展到五个配对训练 seed

2026-10-03。作者希望检验额外训练 seed 下完整方法与组件消融的表现。本次扩展在查看前三个 seed 的部分结果后决定，不能追溯描述为最初就登记了五个 seed。新增数量固定为两个，不以 Ours 排名第一作为继续/停止训练条件。

## 实验设计

原 seed：20261021、20261022、20261023；新增：20261024、20261025。每个 seed 均跑以下四组，共新增八个模型，总计二十个模型。

|组名|唯一改动|
|---|---|
|C20_full|A6_no_alpha完整配置，α=1|
|C20_no_geometry|G、dense clearance、separation三项权重置零；保留完成奖励与板接触力代价|
|C20_no_Q|path_quiet_feet权重置零|
|C20_no_L|path_joint_stall权重置零|

所有组从同一 R2@9000 actor/观测归一化开始，重新初始化 critic 和优化器；不是从已部署 A6_no_alpha@9999 接着微调。相同 seed 的四组保持相同初始化与训练设置。完整配置的构建结果通过测试与 AC3_a100 配置逐字段相等；这里只借用该配置定义，不把 AC3 checkpoint 冒充历史部署 checkpoint。

4096环境、24步/update、20000次更新、每500次保存、10秒训练episode。场景为平地50%＋上下滑动板25%＋自由板25%；板下全低位，平地L/M/H为40/40/20，合计约70/20/10；低位四方向均衡，自然/程序化75/25。保留推力、观测噪声、分部件质量/惯量、Kp/Kd、0–10ms命令延迟等既有随机化。保留SMP奖励，不因低SMP或站起终止。不改L奖励、不引入其他地形或新板尺寸。

## 评价与报告

延续原方案：固定model_19999.pt为最终点，model_10000.pt为中期点；5000/15000仅为小规模监测。正式点评测使用相同2048初态（平地1024、上下板512、自由板512），标称及upper130各一套。主指标SR10仍是在20秒内脱困并连续稳定站立10秒，不改成新30秒协议。

同时报告SR1、脱困率、位移、近直立接触脚移动、HLS95及力矩/速度/功率；按板类型分列成功率。HLS95是高负载低速时长，不直接等同真正堵转。最终统计五个训练seed各自指标及mean±SD，不挑选Ours最好的seed；保留原来所有失败和低分运行。历史部署策略单列参考，不替代某个seed的完整组。验证初态仍为复用的validation bank，不能称独立测试集。

## 运行位置

- 分支：`codex/a6-alpha1-confirmatory-20k`。
- 服务器：`dsw1-clone`。
- 工作目录：`/mnt/workspace/user/luyidan/smp-a6-paper`。
- 新队列：`scripts/queue_a6_confirmatory_extension.py`。
- 新训练目录：`logs/rsl_rl/a6_confirmatory/extension_20261003_20k`。
- 训练仅占空闲GPU1、2；GPU7沿用现有文件锁串行评测。原五个正在运行的任务继续。
- W&B：project `tabletennis/smp`，group `a6_alpha1_confirmatory_extension_20261003`。
- 新队列复用已通过的四组4096环境32步预检；配置差异测试扩展覆盖五个seed。队列启动后产生registration.json、queue_status.json，实际启动信息另存证据。
- 队列只可启动一次，已有输出目录会拒绝覆盖。每次派发前检查目标GPU内存，避免占用其他任务正在使用的卡。

## 已启动核对

新增seed20261024完整组与去几何组已在GPU1/2进入PPO训练，另外六组排队。已核对实际actor、critic、初始qpos及reset计数相同，仅三项几何权重不同；初始化与完整五seed配置差异测试通过。启动证据见 `evidence/a6_five_seed_extension_20261003/launch_audit.json`。

- C20_full: https://wandb.ai/tabletennis/smp/runs/28cfa2o7
- C20_no_geometry: https://wandb.ai/tabletennis/smp/runs/r64qpk48
