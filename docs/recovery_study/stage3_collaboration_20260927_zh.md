# 第三阶段受限恢复：同学接入与对照协议（2026-09-27）

**状态更新（2026-09-27）：我方第三阶段实现已接入，7组均已通过4096环境、24更新预检，正式启动状态见运行记录。** 新入口为 `scripts/train_stage3_recovery.py`，读取 `configs/recovery_study/stage3_comparison_plan_v1.json`；木条/梯架GPU拓扑、独立训练bank、软位置代价已实现。旧M入口保持不变。外部方法适配器仍待同学提供训练接口，收敛项的效果仍未验证。实际启动状态与PID见 `stage3_training_20260927_zh.md` 和正式run manifest。

## 1. 同学从哪里开始

- 仓库：`git@github.com:lu-yidan/smp.git`
- 分支：`codex/recovery-study`。本文编写时可用训练/评测代码基线为 **3713aec**；后续交接提交只添加本文和计划JSON，不改历史训练。
- 本地：`/home/luyd/workspace/smp-a6-egress`；新训练工作区 `/root/workplace/smp-stage3`，旧M工作区 `/root/workplace/smp-recovery-study` 不覆盖。
- 论文：`git@github.com:lu-yidan/G1_Recovery_Below_Block.git`，分支 **main**。论文基准方法仍为L4→R2→A6；本计划与历史方法分开。

```bash
git clone --branch codex/recovery-study git@github.com:lu-yidan/smp.git
cd smp
# 阅读，不会启动训练：
cat docs/recovery_study/stage3_collaboration_20260927_zh.md
cat configs/recovery_study/stage3_comparison_plan_v1.json
```

|入口|用途|
|---|---|
|`scripts/train_stage3_recovery.py::build_config`|第三阶段7组的配置与验证入口；JSON选择场景/消融；不直接加载HoST/FIRM|
|`src/smp/recovery/stage3_task.py` / `stage3_geometry.py`|新分层配额、9个物体槽、分件梯架、软锚点|
|`scripts/recovery_study/build_stage3_bank.py`|独立训练4176/验证1392初态；不载入clutter留出案例|
|`scripts/train_mixed_recovery.py::build_config`|M原训练环境、reward装配、actor-only继承与验证入口；只接受M0/M1/M2，不能直接加载HoST/FIRM|
|`src/smp/recovery/mixed_task.py::quotas/reset/update/task`|分层reset、逐物体遮挡、奖励关闭、任务调制；当前比例仍为旧M|
|`src/smp/recovery/mixed_geometry.py`|旧10场景统一GPU拓扑与几何范围；当前SHARES不等于本计划|
|`scripts/recovery_study/build_clutter_benchmark.py`|木条/梯架/C形固定空间/空箱的CPU评测场景，不等于GPU训练实现|
|`src/smp/rl/tasks/getup/v33_reward_transfer.py`|R2式任务分项、执行器/动作成本|
|`configs/recovery_study/smp_reward_reference_v1.json`|固定SMP归一化参考，独立于actor checkpoint|
|`scripts/recovery_study/evaluate_clutter.py`|独立CPU场景评测接入起点|
|`docs/recovery_study/clutter_benchmark_v2_zh.md`|开发/留出状态及接触验收边界|

**不要复制旧launch脚本直接启动**：它会运行旧M分布与A6/R2混合初始化。新脚本为 `scripts/recovery_study/launch_stage3_20260927.sh`，要求每组预检完成与初始R2评测完成；固定代码commit、配置SHA、bank SHA。

## 2. 初始化：哪些统一R2，哪些不应该

我们的所有新消融统一初始化自 **FT_R2@9000**，不是A6/M2，不是历史scratch R2：

- 服务器文件：`/root/workplace/smp-ft-speed/logs/rsl_rl/ft_prone_speed/formal_20260918_ft_speed_v1/FT_R2/model_9000.pt`
- SHA256：`8b4889f80c6b7cc675f6e9b1e98f2d4a1886a15e372f86070f63c329ba9ca5d1`
- 继承actor及obs归一化；重置critic、optimizer和iteration。固定动作分布std=0.3，不把actor输入噪声和探索噪声混为一谈。
- “第三阶段不用SMP”仍继承SMP训练过的R2，必须写成 **no SMP during adaptation**，不能称从未使用运动先验。

HoST/AMP/FIRM用同学各自已能recovery的原生checkpoint。不能加载R2权重后称作HoST/FIRM对比。先登记机器型号/关节数、obs及history、action及PD、是否有高层命令、归一化、checkpoint SHA、预训练数据和预算；需要可训练checkpoint与完整算法状态，只有ONNX不等于能直接finetune。

每种外部方法先在本协议的平地四方向复核，并记录逐方向能力。移植造成退化要修复接口后再比较；如果原模型确实只会部分姿态，如实分层报告，不从测试集删掉这些方向。

## 3. 场景、姿态和来源分配

平地降为 **20%**，80%用于上方约束；本轮没有台阶/斜坡/固定箱支撑地形。

|场景|三场景对照|扩展分布|初态低/中/近站立|
|---|---:|---:|---|
|平地|20%|20%|40/40/20|
|竖直导向板|40%|15%|100/0/0|
|自由板|40%|20%|100/0/0|
|固定顶板|0|15%|100/0/0|
|交叉木条（2–4件）|0|10%|100/0/0|
|梯架＋板|0|10%|100/0/0|
|多自由板（先2件）|0|10%|100/0/0|

两种分布总体均为 **低88%、中8%、近站立4%**。低位四方向各25%，每个方向自然LAFAN/程序化75/25；中与近站立全部自然。总体为自然低位66%、程序化低位22%、自然中8%、自然近站立4%。程序化22%是降低平地比例后的算术结果，不是“程序化占低位22%”。初始方向按pelvis重力分量标注，不按相机左右。

按scene×stage×direction×source建立环境配额，使用最大余数法分配4096环境，输出精确计数；此为交互份额而非每次reset次数严格相等。不同组使用相同冻结初态及采样规则。无GSI刷新、无在线失败回放，本轮不同时引入新的采样机制。

板下初态必须接触有效、确实受限并留有候选出口；所有qvel归零，SMP/history按静态初态填充（外部时序策略需验证这种history填充是否符合其接口）。自然片段与镜像按组分割。旧clutter留出480例不进入新训练；新增训练形状须重建独立训练bank，若将开发失败用于设计则记录开发用途。

尺寸/质量建议继承现有M合理范围：单板0.60–1.20×0.45–0.90×0.03–0.08m、2–12kg；固定顶板底面0.55–0.80m、长宽1.0–1.7×0.75–1.3m。固定物体不做“可推动质量”随机化。木条与梯架用分件碰撞保留缝隙、真实刚体组合和按几何计算的惯量。它们的精确范围在实物测量/生成器验收后冻结，不能拿整架实心AABB替代横档空隙，也不能照搬独立评测案例作为训练bank。

## 4. 基础reward合同：先保持简单

我们的R2续训采用原M1式模块（它继承A6权重，但已是多物体适配），不在这一轮同时改上升速度、SMP权重或所有动作成本：

`r = alpha * S * T_R2 + R_escape + R_motion_cost + R_convergence`

权重按reward manager乘0.02s前登记，**不能再手动乘一次dt**。SMP ws=6，f2s2冻结prior与固定统计；no-SMP组令S=1，T和其余系数不变。去掉S会改变task整体尺度，结果同时包含运动约束与尺度效应；若据此主张纯“先验信息”效果，需追加冻结均值匹配的尺度敏感性对照，不能只凭这一组定论。

### 4.1 我们的共同恢复任务和成本

|任务分项|权重|
|---|---:|
|阶段姿态|0.22|
|头部上升速度 / 高度|0.18 / 0.10|
|直立度|0.15|
|双足 / base安静|0.08 / 0.07|
|角速度 / 关节速度 / action平滑|0.07 / 0.06 / 0.07|

阶段速度目标仍0.10/0.15/0.20/0m/s，near-target逐渐降到0；不增加一倍速度同时做本轮对照。R2式全部共同成本保持：action rate −0.0015、action acceleration −0.0012、joint acceleration −5e−8、torque −1e−6、joint overspeed −0.02、joint overpower −2e−6、head overspeed −1、sustained effort −0.05、soft joint limits −0.1。继承R2后这些项从续训开始即全量，不重新开启课程。

### 4.2 上方约束与收敛共同项

|项|权重|作用|
|---|---:|---|
|手撑低位几何进展G|+0.45|有效受阻物体，历史最佳覆盖/净空改善，手支撑比例，head≤0.9m|
|稠密净空|+0.08|解除该物体阻挡后饱和|
|全局完成|+0.60|所有有效上方物体持续解除阻挡|
|root到物体距离进展|+0.01|仅当前仍受该物体阻挡时；历史最佳不重置|
|过大接触力|−0.03|与SMP独立|
|近直立足部速度Q|−0.03|平地/障碍共同项|
|单关节持续高负载低速停滞L|−0.20|平地/障碍共同项|

受阻alpha=0.05，解除后alpha=1。旧M的全局解除是逐物体blocked连续15步均false；**这不等价于完全无接触**，可接受有用支撑，但梯架残留套挂可能误被判定完成。统一geometry/closure后再做对照；不让不同方法使用不同完成函数。独立CPU上移探针用实际碰撞分件评测，不能将它冒称训练reward已经具备的能力。

定义外部方法成对对比中的“dense guidance OFF”：G、dense clearance、separation权重均0，alpha恒1；仍保留相同完成+0.60、force−0.03、Q−0.03、L−0.20与各自原生执行质量成本。因此标签为**无稠密脱困引导**，不是没有任何脱困奖励。ON恢复上表三项与alpha门控。两者使用同一场景与reset；避免只对ON组加入随机化或给更多交互。

### 4.3 收敛模块：待验证，默认外部主对比关闭

本轮把候选收敛模块缩小为**脱困后的局部软位置代价**，不一次叠加新的yaw、default pose、速度和力矩项。已有Q/L和R2安静项两组都保留。

- 障碍episode首次全局解除确认后，锁存base XY为`p_anchor`；不使用reset原点。
- 平地在首次连续0.3s满足head≥0.85m、直立度≥0.70、双足各载荷>20N后锁存anchor。
- 门控沿用高度/直立程度：`g=clip((head−0.85)/0.30,0,1)*clip((upright−0.70)/0.23,0,1)`。
- `cost=clip(max(norm(p_xy−p_anchor)−0.25,0)/0.25,0,1)^2`，初始候选权重 **−0.05**，再乘g和当前解除状态；0.25m是允许重心调整的软范围，不是安全硬边界。
- 重新受阻时暂停此项，但不反复更新anchor；只在新episode清除锁存，避免锚点随外移漂移。代价封顶，允许确需进一步逃逸时作取舍。
- 不在actor中加入世界位置或锚点。93D无位置信息，不能保证精确停在锚点；这是训练行为约束，需要用尾部位移、成功率和负载共同验证。
- **未证明有效**。ES2/ES3给出局部改善，也有自由板能力/功率/upper130偏移的trade-off。本轮仅检验一个新项。若无改善则删除，不作为论文已成立的贡献。

## 5. 建议实验矩阵

所有我们自己的组同一R2父代、相同PPO配置与种子；ID带S3前缀以免与旧C/R系列混淆。

|ID|场景|SMP（仅本次适配）|稠密引导|新软锚点|问题|
|---|---|---|---|---|---|
|S3-C0|三场景20/40/40|有|有|无|同预算续训控制|
|S3-C1|三场景20/40/40|有|有|有|简单场景中收敛项效果|
|S3-C2|扩展|有|有|无|主配方，场景扩展收益|
|S3-C3|扩展|有|有|有|收敛项在复杂场景是否仍有效|
|S3-G0|扩展|有|无|无|稠密脱困引导的增量|
|S3-P0|扩展|无（S=1）|有|无|第三阶段SMP的增量|
|S3-PG0|扩展|无（S=1）|无|无|补齐SMP×引导2×2，资源允许时做|

最小优先集为C2/G0/P0，加C3回答当前漂移；C0/C1用于隔离场景和锚点交互。单seed先筛查；正式主结论至少3独立训练seed，不能以一条seed多跑rollout代替。初始actor固定为R2，故结论只关于从该父代的适配，不是整个训练流程的总成本最优。

外部方法每种登记三种状态：Z（原生已训练策略直接评测）、G−（在扩展场景适配，无稠密引导）、G+（同预算适配，加稠密引导）。Z不新增训练且不加奖励；G−/G+共享完成/负载/Q/L等附加项，新锚点均关闭。S3-C2与S3-G0对应我们的G+/G−。

|方法|初始化与必须保留的内容|接入边界|
|---|---|---|
|SMP ours|R2 actor及obs norm；固定SMP|新入口已实现；实际运行状态见训练记录|
|no SMP during adaptation|同一个R2；仅本阶段S=1|不是从头无prior，不等于HoST|
|AMP|同学AMP恢复actor、normalizer、discriminator及算法所需状态|保留AMP判别训练、原生奖励合成、history；不能只把S换成判别分数就声称原生AMP复现|
|HoST|同学HoST恢复policy、多critic与动作约束实现|接入上方奖励并明确分入哪个critic；不把整套原生负成本乘0.05；零样本评测需无外力辅助；适配时也不以向上拉力帮助爬出顶板|
|FIRM|先确认是完整diffusion+online adapter，还是RL技能生成teacher|完整FIRM并非普通PPO actor；需要其原生可训练接口，或调整技能生成/再蒸馏，记录额外数据和算力。仅RL teacher应写FIRM-derived teacher，不称完整FIRM|

所有外部方法保留合理原生结构/历史，不为强行93D而削弱某一个；披露可观测信息。部署actor不能额外读真实障碍geometry。若依赖depth/目标命令等，另列信息条件，不能混称纯本体感知对照。

**外部方法reward接入接口仍是概念合同，不是通用适配API**：拆开 `native_positive_recovery_task`、`native_regularization/prior` 与共享障碍项；ON组只门控正向恢复任务，负成本不能随alpha缩小。SMP继续S×T，AMP保留其原生style/task组合，HoST保持原生多critic分组。记录各方法奖励尺度和预算；跨方法是方法级对照，只有组内G−/G+才隔离引导因子。若要只比较AMP与SMP本身，需额外统一网络/数据/奖励组合，不能用本表直接证明prior优劣。

## 6. 预算、随机化和评测

我们的共同配置（来自当前M `build_config` 实际构造并核对）：4096环境；24控制步/更新；10000更新；每500保存并验证；10s训练episode；50Hz控制/500Hz物理；lr=1e−4、adaptive KL=0.01、gamma=0.99、GAE lambda=0.95、5 PPO epochs、4 minibatches、clip=0.2、固定探索std=0.3。保留actor观测噪声。总追加交互预算为 **983040000 control transitions**；这不是10000个episode。

新的PPO对照统一此预算和超参数。外部原生算法可保留其优化配置，但匹配环境交互/场景暴露并报告差异、调参预算、墙钟/GPU时长；FIRM的再蒸馏不伪装成相同PPO更新。统一训练随机化：push、摩擦、COM、encoder bias、obs noise；部件mass/inertia与六组Kp/Kd从±10%至±20%（2000更新），25%新增项nominal；命令延迟0/2/4/6/8/10ms。需检查实际仿真参数，而不只看配置开关。

10s超时；保留数值异常与异常穿透/力保护；不因站起或低SMP分数终止。共同20s外部评测，四方向均衡，训练/验证/最终测试分离。名义和扰动测试独立汇报，部署控制链复核完成后才比较真机。

主要成功率：20s内解除实际阻挡并连续安静站立10s（所有trial作分母），同时给出clear rate、1s hold、最长hold、time-to-recovery、重摔/介入/保护。按场景和方向报告，场景宏平均作为选点标准之一；训练份额不用于加权掩盖难场景。独立接触/上移探针用于复核，不把“站起来但梯架挂在肩上”算完全脱困。

位移分受困、脱困→站稳、站稳后三段；以脱困点为参考，报告3s固定窗口路程/最大偏移，窗口不足标删失，不记0。负载在2ms子步记录，trial内取峰值再跨trial取P95，失败也计入；另报连续/累计高负载（明确阈值和定义）。

目前**没有本计划下HoST/AMP/FIRM的成功率**。论文准备表全部写Pending，不能填原论文异任务成功率或把待实现写0%。

## 7. 同学交付清单与首个里程碑

1. 给出repo/commit/checkpoint SHA及原生平地四姿态结果，注明只有ONNX还是可续训状态。
2. 接统一机器人/控制接口与冻结评测；导入前后相同obs动作一致性、关节顺序/归一化/history/PD检查。
3. 新训练bank与GPU多物体拓扑验收（CPU/GPU一致、分件缝隙、初态接触、局部reset隔离）；旧缓存/BVH故障回归保留。
4. 每种方法配对G−/G+，各自原生checkpoint相同，先短预检再正式预算；把实际reward日志分项输出。
5. 保存launch/config/SHA/逐trial结果及未剪辑视频，不按不同测试场景分别挑checkpoint。

## 官方资料核对

- HoST官方：https://github.com/InternRobotics/HoST ，https://taohuang13.github.io/humanoid-standingup.github.io/ 。多critic、辅助课程、动作缩放是其方法内容；任务奖励PPO不等于HoST。
- FIRM官方：https://firm2025.github.io/ ，https://arxiv.org/abs/2511.07407 。技能生成与diffusion memory/adapter两阶段决定续训接口；2026-09-27从主页点击公开Code链接返回404，不能据此认定同学没有私有实现。
- 本地AMP历史静态审计见论文仓库 `docs/pipeline_ablation_plan_20260923_zh.md`，与实际同学checkpoint仍需对应。
