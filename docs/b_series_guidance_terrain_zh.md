# B2–B7：初始化、局部动作引导与地形对照

分支 `codex/b-series-guidance-terrain`，工作树 `/home/luyd/workspace/smp-b-series`，服务器 `/root/workplace/smp-b-series`。

## 实验矩阵

|组|GPU|Actor 初始化|场景配额|局部路径进度奖励|
|---|---:|---|---|---|
|B2_FT|0|FT12k|平地50% / 竖直压板50%|无|
|B3_L4|1|早期 L4 model_9999|同上|无|
|B4_FT_guidance|2|FT12k|同上|有|
|B5_L4_guidance|5|早期 L4 model_9999|同上|有|
|B6_FT_terrain|6|FT12k|平地、竖直压板、自由板、台阶/斜面各25%|有|
|B7_L4_terrain|7|早期 L4 model_9999|同上|有|

T_P5/T_FT12k/T_L4 已停止，保留 checkpoint 和日志；服务器停止审计为 `/root/workplace/smp-p5-terrain/outputs/stopped_for_b_series_20260918.json`。GPU3/4 的 B0/B1 原任务不修改，保持原10000更新预算，不自动追加。

六组从相应 Actor（含探索标准差及观测归一化）开始；critic、optimizer 重新初始化，各组使用相同的新 critic。全部仍是 finetune。SMP reference 统一冻结为 FT12k 保存的 mean/count，避免把奖励归一化变化混入初始化对照。新六组 seed 为原 B1 seed + 101，B2 因而是 B1 的不同 seed 重复，并非复制已有曲线。

新增10000 PPO更新，4096环境，保存和独立评估每500更新；每1000更新及最终生成视频。Actor93D，critic960D，f2s2 prior、ws6、训练episode10秒。站起后继续，无低SMP提前终止，保留数值异常及无效压板接触终止。沿用 B1 的任务/SMP/板奖励，不加 P7 过渡奖励，不加失败回放。

## Reset 与动力学

全部低位，各场景中仰卧/伏卧/左侧/右侧各25%，自然75%＋程序化25%。这是环境配额，不等价于实际 reset 次数或训练步数的比例。自然来源为经接触筛选、加姿态/位置变化的 LAFAN low bank，不是原始 LAFAN 无筛选采样。复用 `multiterrain_bank/train.npz` 和独立 validation bank。

四场景组台阶/斜面内部：台阶内部204、边缘308、跨台阶204，斜面内部104、边缘204环境，合计1024；非仅台阶中心。自由板可平移、旋转和抬起。压板质量课程4–6kg→4–12kg；新阶段课程计数归零。

所有六组保留 B1 动力学随机化：双腿、上身（含手臂）、骨盆/腰三个质量组，质量和惯量使用相同系数；六个电机分组的实际 Kp/Kd 同系数缩放；左右对称。系数前2000更新由±10%扩大至±20%，约25%episode保持新增动力学标称值。全关节位置命令延迟按物理步采样0/2/4/6/8/10ms，每episode固定。力矩上限不改变。这不是任意惯量独立扰动，也不是电机转速-力矩模型。

原有 push、摩擦、COM、编码器与 Actor 观测噪声继续保留。推扰不是新增变量，因此不会仅给某个组开启。

## 局部动作引导的精确定义

BeyondMimic `fallAndGetUp2_subject2_mj.npz` 中 3–10秒实际是站到躺下；本实验参考 **12.5–15.5秒** 的起身片段。FK核对与导出审计见 `datasets/guidance/audit.json`。原始动作有手臂/腿部自穿透与地面穿透，因此不直接用作全身 tracking 或 reset。

只提取两侧髋/膝/踝 pitch、三个腰关节、骨盆重力方向（12维特征）。不跟踪手臂、根部绝对位置，不向Actor添加参考信息。仅平地仰卧、初始姿态接近片段且未接近片段末端的episode启用。其他方向、压板和台阶不直接得到此项引导。

每步只允许最近路径索引保持或向前一帧；归一化姿态误差门槛0.25，初始化门槛0.75。势函数为归一化路径进度加0.2倍姿态相似度，只奖励超过本episode历史最佳势函数的正增量。没有按时间追赶参考，也不因反复摆腿或原地保持重复获得进度奖励。128环境冻结策略校准发现权重0.2的实际信号过弱，正式组改为权重5，单episode积分额外收益理论上不超过6。该设计是待检验的弱引导，不是运动速度约束，不保证消除甩腿；必须对比 B2/B4、B3/B5 的实际轨迹、关节速度和功率。

## 评测与解释

所有组独立评测同样四场景、四方向，20秒，不推扰/无Actor噪声；压板统一6kg。评测标称动力学以及上身质量/惯量×1.3压力域。报告有效脱困、连续站稳1秒/10秒、无效接触、每轨迹峰值的P95（力矩、速度、功率、头部接触力）。站稳率不等于安全，也不自动衡量甩腿次数；同步查看视频/trace。

B2/B3比较初始化；B4/B2、B5/B3比较局部引导；B6/B4、B7/B5比较混合地形。最后两对同时改变平地/原压板配额，结论应称“混合场景训练配置的效果”，不能拆解成自由板或台阶的单独因果作用。仅一个新seed，不能据小幅成功率差异宣称显著优势。

## 正式运行记录（2026-09-18）

服务器目录 `logs/rsl_rl/b_series/formal_20260918_b_v1`。六组均已进入PPO更新并连接W&B；运行代码提交 `d8bbb750cd7ec1032c722c37d40e44b68d2adeef`，服务器 `outputs/source_revision.json` 逐文件SHA核对通过。

全部通过4096环境4更新预检、实际动力学/延迟审计、局部reset隔离检查和128环境20秒checkpoint重载四场景评测。预检成绩不是正式训练成绩。校准及验证摘要见 `b_series_preflight_20260918.json`。权重5时，冻结策略的引导积分/全环境任务积分约0.13%–0.46%，仅平地仰卧受益；后续需要观察其是否足以改变起身路径。直接回放参考特征可完整推进路径，重复回放无额外进度收益。

W&B：
- [B2_FT](https://wandb.ai/tabletennis/smp/runs/formal_20260918_b_v1-b2_ft)
- [B3_L4](https://wandb.ai/tabletennis/smp/runs/formal_20260918_b_v1-b3_l4)
- [B4_FT_guidance](https://wandb.ai/tabletennis/smp/runs/formal_20260918_b_v1-b4_ft_guidance)
- [B5_L4_guidance](https://wandb.ai/tabletennis/smp/runs/formal_20260918_b_v1-b5_l4_guidance)
- [B6_FT_terrain](https://wandb.ai/tabletennis/smp/runs/formal_20260918_b_v1-b6_ft_terrain)
- [B7_L4_terrain](https://wandb.ai/tabletennis/smp/runs/formal_20260918_b_v1-b7_l4_terrain)
