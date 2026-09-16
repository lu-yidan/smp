# L4 → V33 reward FT → 受限场景 recovery：训练与验证流程

本流程是当前证据支持的研究方案，不声称已找到最优超参数。核心原则是固定可部署接口，先掌握恢复能力，再改善动作质量，再学习受限路径；每次迁移保留可回退的 checkpoint，用分姿态成功率和物理指标选择候选，不能只看 epoch 或训练 reward。

## 已有证据与模型血缘

|阶段|起点与目标|实际协议|可据此作出的结论|
|---|---|---|---|
|A：L4 from scratch|随机 actor/critic，学平地 recovery|4096 env，10k 更新；late40%/middle40%/low20%，low=自然15%+程序化5%，四方向均分；早期 L4 model9999|原始 L4 仿真起身能力强，但动作很快，不是安全完成品|
|B：V33 reward transfer|L4 actor+actor观测归一化迁移，替换为 V33 风格的阶段任务与成本|新 critic、新 optimizer，LR1e-4；安全成本500更新渐入；10秒episode，取消站起终止；低位不因SMP终止|保留 FT12k：已在真机仰卧起身，但伏卧停滞、坐蹲过渡振荡仍存在|
|C：本轮受限场景微调|四组从同一个 FT12k actor 出发|4096 env/组，+10k PPO更新，每500保存；20秒episode，50%压板只在E2/E3；critic/optimizer重新初始化|这是准备验证的假设，正式结果尚未产生|

关键身份：
- 早期 L4 `model_9999.pt` SHA256：`7e98c06aa72f4ced7ab1f5bcfc67c011096fb5b3b4cdc453e43f7aac97409813`。
- FT12k `model_12000.pt` SHA256：`8f05543b644b0a1d11246e85778f99220460ef2a744417ea940296eed768c768`。
- 部署 ONNX SHA256：`51aed4a4c591b3bd3fbb0a692221ec9d674fe5942141d195e65bbfe1bfd560e7`。
- FT12k 的“12000”是继承的绝对编号：L4 完成9999后从10000开始微调，不代表额外微调了12000次。
- 当前 FT 使用 f2s2、SMP ws6；历史压板 V33 使用 V7 prior、ws4。本流程迁移的是奖励/任务机制，不是完整复现旧 V33。
- A→B：L4缺失SMP参考统计，曾使用重建参考，不能称作精确恢复。B→C：直接读取FT12k保存的 mean/count，不再重建，也不重新估计以人为提高SMP得分。
- 早期L4的站起终止与B/C不相同。比较历史结果时必须注明训练协议；E0是本轮共同20秒协议下的续训对照，不是完全不变的历史FT训练。

## 为什么当前用 FT12k，不直接继续 L4 或最新 FT

FT12k已有真机轨迹和可重现的部署输出。日志五段逐帧重算，目标关节角与该ONNX完全一致。三次仰卧起身，两个短伏卧尝试停滞；其中三次左膝PD估算力矩达139Nm，过渡阶段振荡。bridge没有传tau_est，因此这些不是实测电机力矩或功率。

历史后续FT曾出现伏卧验证退化，因此“多训练”不是默认改进。当前先固定FT12k研究场景和奖励因素，找到有效方案后，再用L4初始化做复现，并最终补全from-scratch路线与总预算对照。

## 本轮四组，均从 FT12k 初始化

|组|4096环境的分配|奖励|所回答的问题|
|---|---|---|---|
|E0_flat|全部旧L4 reset分布|现有V33风格平地奖励|共同20秒训练协议下，仅继续训练会怎样？|
|E1_prone|2048旧分布平地+2048接触有效伏卧支撑起点；没有压板|同E0|更合适、更多的伏卧起点是否就足够？|
|E2_plate|2048旧分布平地+2048同E1伏卧起点上放板|平地奖励+压板超力代价；无正向脱困奖励，无受困门控|物理压板加通用接触成本能否帮助探索？|
|E3_guided|与E2完全相同|增加V33脱困奖励与受困门控|专门的脱困机制提供多少收益？|

E2与E1不仅物理环境不同，还多了仅在接触板时生效的超力成本，不能解释为严格的“仅物理板”对照；E2与E3隔离正向脱困奖励+任务门控这个组合，尚未拆分该组合内部各项。

实际分布口径：
- E0：late1638、middle1642，low四方向各204，其中每方向程序化51。
- E1/E2/E3：基础半组late819、middle821，low四方向各102，其中每方向程序化25；额外2048个固定伏卧支撑环境。
- 因而后三组整体约20%late/20%middle/60%low。50%是固定环境配额，也是控制步分配；不是每次reset抽样概率，更不是旧20%低位里的50%。episode计数比例会因提前终止而变化，不能混用。
- E1/E2/E3初始化的完整机器人qpos必须逐元素一致；有板与无板只改变场景，不能偷偷改身体起点。

共同训练合同：93D actor（无真实基座线速度）、960D critic、部署对齐机器人/PD/力矩上限、2ms物理步/20ms控制步、f2s2 prior/ws6、actor观测噪声和checkpoint探索分布、原有push/摩擦/COM/编码器随机化均保留。所有组使用相同seed；这是首轮筛选，不是多seed统计结论。当前critic也未额外加入板几何观测。

20秒episode，站起不结束。旧基础半组的中/后段仍按原SMP阈值0.02、前5步宽限结束；全部低位及新增伏卧组没有低SMP结束。仿真不稳定结束；压板穿透>20mm、接触力>1500N、加载超时/无效加载会标记并结束，单独统计。20mm/1500N是仿真有效性保护阈值，不是硬件安全标准。

FT12k现有安全成本已训练，C阶段保持满权重，不重新从零渐入让策略暂时失去约束。奖励任务/episode分布变化，因此四组统一新critic/optimizer，保留actor及其归一化和std，初始LR1e-4、自适应KL。参数载入逐tensor验证。

## 压板机制与迁移修正

E3在等待压板/受困阶段将`task×SMP`乘0.05，脱困后恢复1。SMP仍计算；独立脱困奖励不再乘SMP：
- 全身几何进展 +0.45：需要手部地面接触、头高≤0.9m，奖励覆盖深度的历史最好改善和清空后的距离改善，避免来回蹭刷进展。
- 清空比例/间隙 +0.08。
- 脱困后持续 +0.60（不是一次性bonus；后续可单独消融持续奖励，检查是否影响动作速度）。
- 中心分离进展 +0.01，只是弱方向提示。
- E2/E3共同：板接触超300N平方代价 −0.03。

成功要求所有机器人碰撞体离开板投影范围、至少25mm平面间隙、无板接触持续15控制步，并满足至少5步手支撑、累计至少4cm支撑分离进展。`escaped`只是脱困，不等于恢复完成；最终还要连续10秒严格稳定站立。

板尺寸0.90×0.64×0.07m，固定水平位置/朝向、仅有被动竖直滑动自由度。前100000控制步（约4167次PPO更新，每次24步）从较偏置、4–6kg扩展至接近居中、4–12kg。质量与惯量一致缩放。课程采用明确的时间表以保证E2/E3难度可比，不按各组成功率独立调整以混淆消融。

### 验证中发现的迁移问题

旧“crawl-ready”姿态不能直接搬到当前模型。直接照搬时肩臂相互穿入接近70mm；旧root pitch符号在当前姿态判据下也与我们的伏卧标签不一致。该原型未用于正式训练。

新的`plate_prone_v1`由当前部署机器人生成，训练2048、验证256，固定独立随机种子：
- 与现有四方向bank一致，伏卧projected gravity x约+1。
- 修正肩部roll方向，检查关节限位、地面和自碰撞。
- 最低碰撞面距地4mm，至少一只手碰撞面距地≤40mm，头高≤0.4m；允许微小落地建立接触，不宣称reset瞬间双手已经承重。
- CPU逐姿态检查深穿透；随后GPU环境再次审计，不能只凭生成器通过。
- E1/E2/E3复用同一个bank；验证集不用于训练。该bank是程序化支撑姿态，不是LAFAN，也不是直接复制真机日志姿态。
- 生成器：`scripts/build_plate_prone_bank.py`；哈希在bank的`manifest.json`及每个运行的`launch.json`保存。

## 训练前必须通过的验证

1. 源checkpoint SHA、actor逐tensor一致，critic为新初始化；93D/960D维度及噪声/push配置检查。
2. reset无速度残留，SMP历史从最终物理姿态重新填充；部分reset不污染其他环境。
3. 四组各4096环境、4次PPO更新，reward/obs有限，完整checkpoint保存→独立评测链路通过。
4. E1/E2/E3机器人初始qpos完全一致，E2/E3实际有板比例恰好50%；初始接触深度符合筛选标准。
5. 对原FT12k做固定20秒评测和场景视频，确认板的位置、伏卧朝向、接触及脱困判定。短检查只能证明工程链路有效，不证明学会了。
6. 正式launcher读取验证报告并核对代码哈希；未通过不启动formal。

## 评测与选模：先可恢复，再稳定，再降低负载

每500更新保存并评测，1000更新及最终录制20秒视频。固定验证条件：
- 平地四方向：现有procedural validation，按方向报告连续1秒与10秒严格站立成功率。
- 额外伏卧：新bank held-out姿态、无板，直接测试是否迁移到平地。
- 容易压板：固定偏置分布、4–6kg。
- 困难压板：居中、8kg；与随训练变化的课程分开。

四个评测均使用确定性actor、无额外观测噪声/push/DR，作为可比较的nominal检查；这不代表训练关闭随机化，也不等于鲁棒性验证。候选另做有DR/延迟/扰动评测，最终还要未见尺寸/方向/负载测试。训练集、validation和最终test职责分开，不反复用test选checkpoint。

负载在2ms物理子步统计：每episode全关节最大力矩、速度、机械功率与头部接触力，再跨episode取P95。失败episode与成功episode分别检查；不能把“趴着不动导致低功率”当作更安全。仿真机械功率也不能与当前真机缺失的tau_est直接比较。

选择采用约束/Pareto思路，避免把一切压成总分：
1. 无明显穿模利用、无异常碰撞、无统计缺失。
2. 四方向及有板成功率满足预先确定的门槛，不能用平均值掩盖伏卧全失败。
3. 同样成功率下比较稳定保持、支撑转换、速度/功率/持续力矩和自然程度。
4. 连续多个保存点表现稳定，再进入deployment MuJoCo验证；不要仅挑一次尖峰成绩。
5. 若无板伏卧提高但仰卧退化，先修正任务/数据平衡；若仅有板成功，分析是否依赖板支撑；若E2/E3都停滞，回看接触与reward有效梯度，不直接增加总epoch或减弱全部安全成本。

后续“更智能”的课程必须作为独立实验：仅在固定验证连续通过后升级间隙/覆盖/质量，退化时保留旧难度回放。但首轮不让各组各自改分布，否则无法判断奖励改动的效果。

## 真机与论文边界

桥接日志需补tau_est有效性、活动FSM、模型哈希、状态tick/包龄及控制时间戳，推扰同步录像。实测力矩采集补全后再分析真实负载。坐蹲阶段的5–6Hz振荡优先核查接触和延迟，不能仅凭频谱认定PD共振；当前不通过降低真机PD增益来掩盖问题。

受限恢复论文的重点是受阻时选择低位脱困路径，然后恢复站立；不是简单堆叠更多地形。先用压板有/无、引导有/无对照验证机制，再加入台阶。站立抗扰动作为恢复完成后的检查，小扰动希望原地保持，大扰动允许必要跨步。

最终论文要补同总预算from-scratch受限场景、平地直接迁移、普通微调、新方法，多seed和未见场景。当前四组是机制筛选，尚不能替代最终论文实验。完整报告A+B+C预算，不能把微调更新量当作总训练成本。

## 复现入口

在独立工作区`/root/workplace/smp-escape-ft`运行，分支`codex/ft12k-plate-recovery`；不改部署默认模型，不自动上真机。

```bash
export PYTHONPATH=src:scripts:.
# 本地生成后同步bank，或者在服务器相同源码下生成；核对manifest哈希。
.venv/bin/python scripts/build_plate_prone_bank.py

.venv/bin/python scripts/launch_plate_transfer.py --mode preflight --gpus 0 1 2 3 \
  --checkpoint /root/workplace/smp-master-repro/logs/rsl_rl/v33_reward_transfer/formal_20260916_083821/FT/model_12000.pt

.venv/bin/python scripts/verify_plate_preflight.py --launches run_control/plate_transfer/PREFLIGHT_ID/launches.json

.venv/bin/python scripts/launch_plate_transfer.py --mode formal --gpus 0 1 2 3 --updates 10000 \
  --checkpoint /root/workplace/smp-master-repro/logs/rsl_rl/v33_reward_transfer/formal_20260916_083821/FT/model_12000.pt
```

运行信息见`run_control/plate_transfer/*/launches.json`，配置见每组`params/`、`launch.json`，评测见`validation/`。W&B项目`tabletennis/smp`，group `ft12k-plate-transfer`。

A/B历史复现入口见同目录`v33_reward_transfer.md`及`scripts/train_fixed_low_ablation.py`、`scripts/train_v33_reward_transfer.py`。本轮没有重跑A/B，也没有把不完整的历史SMP统计伪装成精确续训。

## 2026-09-16 验证与启动记录

- 代码：`8107aca`，独立分支 `codex/ft12k-plate-recovery`。
- 预检：`preflight_20260916_180116`，四组各4096环境、4次PPO更新，全部完成checkpoint及四条件评测；E1/E2/E3机器人初始qpos逐元素一致；全部验证初始碰撞深度为0。
- 正式：`formal_20260916_180706`，E0/E1/E2/E3分别GPU0/1/2/3，源checkpoint重新载入FT12k，未使用预检训练后的actor。旧GPU4/5运行未停止。
- 验证报告：`outputs/plate_preflight/plate_preflight_verified.json`（本地）；服务器为`outputs/plate_preflight_verified.json`。
- 视频：本地`outputs/baseline_verified/`，服务器同名目录。这里是训练前FT12k，不是新训练成果。

|原FT12k固定验证条件|n|连续稳定10秒|有效脱困|
|---|---:|---:|---:|
|flat|64|95.31%|0.00%|
|prone|64|79.69%|0.00%|
|plate_easy|64|0.00%|1.56%|
|plate_hard|64|0.00%|0.00%|

无板条件的“有效脱困”不适用，其0不代表起身失败。平地64例为仰卧/伏卧/左/右各16；对应稳定率100%/81.25%/100%/100%。新支撑伏卧无板为79.69%；两种压板均未达到稳定10秒。该结果支持继续学习的必要性，不证明任何消融组已优于原模型。初始无穿透不等于整个rollout无穿透；压板评测无效比例约29.7%，需在训练中跟踪降低。

W&B：
- [E0_flat](https://wandb.ai/tabletennis/smp/runs/plate-ft12k-e0_flat-20260916_180706)
- [E1_prone](https://wandb.ai/tabletennis/smp/runs/plate-ft12k-e1_prone-20260916_180706)
- [E2_plate](https://wandb.ai/tabletennis/smp/runs/plate-ft12k-e2_plate-20260916_180706)
- [E3_guided](https://wandb.ai/tabletennis/smp/runs/plate-ft12k-e3_guided-20260916_180706)
