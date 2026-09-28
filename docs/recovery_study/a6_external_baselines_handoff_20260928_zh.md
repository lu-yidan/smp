# HoST / AMP / FIRM：历史 A6 第三阶段对比实验交接

日期：2026-09-28。对象：已拥有可训练 recovery 基线的同学。

**本轮目标：从各方法自己的平地 recovery checkpoint 出发，迁移历史 A6 的场景、reset、共同任务与成本，配对检验稠密脱困引导是否有效。**

本文件规定的是待执行协议，不代表外部方法已经接入、已经开始训练或已有成功率。

## 1. 使用哪个分支和版本

文档维护分支为 `codex/recovery-study`，但迁移历史 A6 时必须固定代码提交：

```text
1b6d7e61ddbcd2caf9ea17b02160dd1cf0d119d2
```

A6 原实验分支名是 `codex/r2-v33-path-ablation`。上述运行提交已包含在远端 `codex/recovery-study` 历史中。请新建独立工作目录：

```bash
git clone --branch codex/recovery-study https://github.com/lu-yidan/smp.git smp-a6-baseline
cd smp-a6-baseline
git switch -c baseline/a6-adaptation 1b6d7e61ddbcd2caf9ea17b02160dd1cf0d119d2
git rev-parse HEAD
```

本文是后来增加的交接说明，不在该历史提交内；切换前保存本文或保留在线链接。

**不要复制最新分支的同名训练文件来声称复现 A6。**后续实验已经修改部分脚本、场景及奖励逻辑。`stage3_collaboration_20260927_zh.md` 是后续 S3 方案，不能直接当作本轮 A6 配置。

| 条件 | 本轮历史 A6 协议 | 后续 S3 协议 |
|---|---|---|
| 平地比例 | 50% | 20% |
| 训练障碍 | 单导向板、单自由板 | 三场景变比例，或扩展固定顶板、木条、梯架、双板 |
| 脱困后的 separation 进展 | 历史 A6 仍可能支付 | 已修改为受阻时支付 |
| 位置锚点 | 无 | 部分组测试新增软锚点 |
| reset bank | 历史 A6 bank | 新建 S3 bank |

不能把两套配置混用后统一标成 A6。

## 2. 从哪里找实现

以下代码链接全部固定到历史提交，不随分支更新。

| 入口 | 内容 |
|---|---|
| [scripts/train_v33_path_ablation.py](https://github.com/lu-yidan/smp/blob/1b6d7e61ddbcd2caf9ea17b02160dd1cf0d119d2/scripts/train_v33_path_ablation.py) | 总入口；重点阅读 `build_config(..., arm='A6')` 及它调用的 `train_d_series.build_config`；包含预检、配置导出、历史验证 |
| [v33_path_ablation.py](https://github.com/lu-yidan/smp/blob/1b6d7e61ddbcd2caf9ea17b02160dd1cf0d119d2/src/smp/rl/tasks/getup/v33_path_ablation.py) | G、dense clearance、Q、L；手支撑、历史最佳几何进展、关节历史与 reset |
| [r2_ablation.py](https://github.com/lu-yidan/smp/blob/1b6d7e61ddbcd2caf9ea17b02160dd1cf0d119d2/src/smp/rl/tasks/getup/r2_ablation.py) | 平地低/中/近站立分配、自然状态采样、关节负载计时与子步统计 |
| [multiterrain.py](https://github.com/lu-yidan/smp/blob/1b6d7e61ddbcd2caf9ea17b02160dd1cf0d119d2/src/smp/rl/tasks/getup/multiterrain.py) | 板子放置/质量、完成和无效判据、alpha 门控、completion/separation/force |
| [multiterrain_geometry.py](https://github.com/lu-yidan/smp/blob/1b6d7e61ddbcd2caf9ea17b02160dd1cf0d119d2/src/smp/rl/tasks/getup/multiterrain_geometry.py) | 场景实体及50/25/25配额；需一起追踪导向板实体的构造 |
| [v33_reward_transfer.py](https://github.com/lu-yidan/smp/blob/1b6d7e61ddbcd2caf9ea17b02160dd1cf0d119d2/src/smp/rl/tasks/getup/v33_reward_transfer.py) | R2式起身、安静运动、动作/执行器代价与阶段目标 |
| [balanced_dynamics.py](https://github.com/lu-yidan/smp/blob/1b6d7e61ddbcd2caf9ea17b02160dd1cf0d119d2/src/smp/rl/tasks/getup/balanced_dynamics.py) | 分部件质量/惯量、Kp/Kd、命令延迟及 reset |

历史实际运行证据：

- [A6 env.yaml](https://github.com/lu-yidan/G1_Recovery_Below_Block/blob/b942d04/evidence/lineage/a6_env.yaml)：展开后的奖励、事件、终止、仿真配置。
- [A6 launch.json](https://github.com/lu-yidan/G1_Recovery_Below_Block/blob/b942d04/evidence/lineage/a6_launch.json)：初态数量、bank SHA、初始化信息。
- [论文完整奖励表](https://github.com/lu-yidan/G1_Recovery_Below_Block/blob/b942d04/sections/reward_definitions_table.tex)：27项公式和三阶段权重，便于阅读；程序实现和 frozen YAML 是数值核对依据。

只拷贝 reward 函数不够。必须同时移植传感器、历史缓存、更新频率、reset 清空以及完成/无效判据，否则同名奖励可能计算出不同结果。

## 3. 初始化和方法身份

| 方法 | 适配初始策略 | 必须明确的接口 |
|---|---|---|
| SMP | FT_R2@9000；历史 A6 的 actor 父代 | actor及obs normalizer继承，fresh critic/optimizer；固定 SMP 统计 |
| AMP | 同学自己的 AMP recovery checkpoint | actor、normalizer、discriminator、动作历史及算法所需训练状态 |
| HoST | 同学自己的 HoST recovery checkpoint | 原生训练结构、critic/奖励分组、动作约束和辅助课程 |
| FIRM | 同学自己的可训练 FIRM 系统 | 明确完整方法还是技能生成 RL teacher；续训/再蒸馏流程与额外预算 |

外部方法不能加载 R2 或 A6 actor 后仍称作原生 HoST/AMP/FIRM 对比。只有 ONNX 推理文件不等于已有可续训接口。仅使用 FIRM 的 RL teacher 时应标为 FIRM-derived teacher。

本仓库训练器不是通用外部方法适配器；不能把 HoST/FIRM checkpoint 直接传给原脚本 `--checkpoint`。原脚本要求本仓库 runner 的 checkpoint 结构，还依赖 SMP reference。

先记录各自平地四姿态能力。迁移关节顺序、归一化、PD或history造成的退化应先排查；原模型确实缺失某个方向能力时，如实报告。

## 4. 场景和 reset：与历史 A6 对齐

| 场景 | 环境份额 | 内部低位 L | 内部中间 M | 内部近站立 H |
|---|---:|---:|---:|---:|
| 平地 | 50% | 40% | 40% | 20% |
| 竖直导向板 | 25% | 100% | 0 | 0 |
| 自由刚体板 | 25% | 100% | 0 | 0 |

- 导向板仅能上下移动，不能平移/倾斜；不是零自由度固定顶板。自由板可平移、旋转。
- 每类场景的低位四方向均衡：仰卧、伏卧、左侧、右侧，各约25%。
- 各低位方向内部：约75%自然 LAFAN、25%程序化；中间和近站立全部来自自然状态。
- 全体约为低70%、中20%、近站立10%；历史4096环境实际为低2868、中820、近站立408，自然3380、程序化716。场景数量为2048/1024/1024。
- 这是环境配额，不保证不同失败时长下每类 reset 次数或实际采集步数也恰好同比例。
- 初始速度归零；静态初态填充历史。外部时序策略需核验与自身history接口的兼容性。
- 无 GSI 定期刷新，无在线失败回放，不加入台阶、斜坡、木条或固定顶板。

板尺寸为0.90×0.64×0.07 m；历史 A6 不随机尺寸。板质量范围由4–6 kg逐渐扩大到4–12 kg，历史名义验证使用6 kg。初始底面按覆盖机器人碰撞几何最高点加2 mm放置，不能改成统一世界高度。具体导向范围、接触和惯量实现按冻结代码迁移。

### 必须单独交付的数据

| 资产 | 历史代码路径 | 已归档 SHA256 |
|---|---|---|
| 低位/板下训练bank | `outputs/multiterrain_bank/train.npz` | `287eae8e8840c1b3281e7010a84182b824fefacd34439c4f993e9f130af1027a` |
| 自然课程训练bank | `datasets/reset_banks/natural_curriculum_v1/train.npz` | `e9f94540520d7927dee01150a0f0bae39ad2aad5ae008b1c7c71f521650e44b9` |
| 程序化低位训练bank | `datasets/reset_banks/procedural_low_v1/train.npz` | `e289bed8d1c93e87fbdcf969fc5fc741f2d8500a0908ed145d2a4e02e7fe116d` |

另需交付机器人XML/mesh、验证bank及其SHA和生成来源。克隆Git不保证这些大文件存在；首次联调前检查并补齐，不用其他bank静默替代。历史验证数据不能自动当成新的独立最终测试集；自然片段及其镜像需按组隔离。

## 5. 共同任务与代价

SMP侧的历史目标为：

```text
r = alpha * S * (T_rise + T_quiet) - C + R_escape
```

`S` 是 SMP 动作先验奖励。`C` 包含下面全部运动/执行器代价，含 Q/L；不存在重复扣分。

### 5.1 正向恢复任务

| 项 | 权重 |
|---|---:|
| 阶段姿态 | 0.22 |
| 头部上升速度 | 0.18 |
| 头部高度 | 0.10 |
| 直立程度 | 0.15 |
| 双足安静 | 0.08 |
| base安静 | 0.07 |
| 身体角速度安静 | 0.07 |
| 关节速度安静 | 0.06 |
| 动作平滑 | 0.07 |

低位/过渡/上升/站立的名义上升速度目标为0.10/0.15/0.20/0 m/s，接近目标高度时衰减。复制完整phase状态机、滞回、膝关节条件、脚支撑条件和独立超速成本，不能只复制上述速度数字。

### 5.2 运动与执行器代价

以下按奖励中的有符号权重列出；实现总公式的 `C` 时取相应正代价，最后统一减去。

| 项 | 权重 |
|---|---:|
| action rate | −0.0015 |
| action acceleration | −0.0012 |
| joint acceleration | −5e−8 |
| torque | −1e−6 |
| joint overspeed | −0.02 |
| joint overpower | −2e−6 |
| head overspeed | −1 |
| sustained effort | −0.05 |
| soft joint limits | −0.10 |
| Q：近直立脚部运动 | −0.03 |
| L：单关节持续高负载低速停滞 | −0.20 |

适配开始即启用全量成本，不重新开启第二阶段的500更新成本渐入。Q/L也用于平地。

### 5.3 障碍交互奖励

| 项 | 权重 | 必须保留的含义 |
|---|---:|---|
| G：有手部支撑的低位几何进展 | +0.45 | 历史最佳重叠深度/净空改善；手接触比例；头高≤0.90 m；曾接触、未脱困、未无效 |
| dense clearance | +0.08 | 清空覆盖碰撞体比例及最小间隙 |
| completion | +0.60 | 当前满足脱困确认时逐步支付，不是一次性bonus |
| separation progress | +0.01 | root到板的水平距离超过历史最佳；历史A6在脱困后仍可能支付 |
| excess plate force | −0.03 | 超过300 N的板接触力软代价，独立于先验 |

`alpha=0.05` 用于尚未确认脱困的板下episode，确认后/平地为1。它只缩放正向恢复任务；不能缩小负成本。G的水平几何计算与completion的几何判据不同，不能合并成一个近似布尔量。

历史completion需要曾接触、几何净空达标且连续15个控制步无板接触；重新受阻/接触会撤销当前确认。迁移时保留具体判据，不把“碰撞体高于板”的完成豁免加进G。

所有权重均为统一控制步长0.02 s缩放之前的值。原reward manager已乘dt；迁移时确认只缩放一次。

## 6. “使用相同reward”与保留原方法的边界

共同物理任务、成本和障碍项按上述定义迁移；各方法自己的运动先验与学习机制保留。AMP的判别器得分不直接冒充SMP的S；HoST的奖励/critic分组需在其训练器里接入；FIRM按其实际可训练流程适配。

每位同学交付一张reward映射表：`原生项 → 保留/替换/删除 → 共享项 → 权重 → dt → 所属critic或优化目标`。与共享成本重复的原生任务/成本应显式处理，不能静默叠加两份。因原生算法不可拆分而无法统一的部分，记录为方法差异。

因此这是一项“外部恢复方法对共同脱困引导的适配”实验；不是只比较SMP与AMP先验本身的严格单因素实验。跨方法保留原生信息/优化结构并披露差异；每个方法内部的G−/G+才用于隔离稠密引导的作用。

## 7. 每种方法的最小对照矩阵

| 组 | 初始化 | 新增适配训练 | 稠密引导 |
|---|---|---|---|
| Native | 自己的平地recovery checkpoint | 无；直接统一评测 | 无新增训练奖励 |
| G− | 同一个原生checkpoint | 相同A6场景/reset/预算 | 关闭G、dense clearance、separation；alpha恒1 |
| G+ | 同一个原生checkpoint | 相同A6场景/reset/预算 | 启用完整历史A6设置 |

G−/G+都保留completion、force、Q/L、相同共同任务和成本，以及各方法约定的原生机制。G−叫“无稠密脱困引导”，不是“没有任何脱困奖励”。本轮两组都不添加位置锚点、yaw或default pose新奖励。

我方SMP也需以R2为共同初始策略做匹配G−/G+；历史A6可以作参考，但不能用其挑选后的单个最佳checkpoint替代全部匹配种子。新增“脱困后关闭separation”应另立单因素组，不混入G+。

正式结果建议至少3个独立训练seed；先用一个配对seed完成接口和短程预检，再扩展重复。不要以同一checkpoint增加rollout数量代替训练seed。

## 8. 预算、随机化与终止

- 共同追加交互预算：4096环境×24控制步/更新×10000更新，即983,040,000个环境控制步；每500更新保存。
- 控制50 Hz，物理500 Hz，10个物理子步/控制步。匹配物理时间和控制动作含义；不能只匹配epoch标签。
- 训练episode 10 s；站起后继续，不按低SMP分数终止；保留数值异常及历史无效板保护。具体触发条件从frozen YAML和代码复制。
- 保留push、摩擦、COM、编码器偏置和actor观测噪声；继承分部件mass/inertia、六组Kp/Kd、0–10 ms命令延迟。
- 新增动力学项范围前2000更新从±10%扩大至±20%，约25%episode对此部分取nominal。这不是关闭全部随机化。
- 外部算法可保留必要的优化设置，但配对组应一致；记录额外蒸馏/离线数据、调参、墙钟及GPU预算。不同原生预训练成本单独报告。
- 统一机器人、关节/动作映射、PD与接触参数。SMP actor为93D；外部方法不必强制删去其必要history，但必须披露额外信息。actor不能额外读取真实障碍geometry后仍称同信息对照。

## 9. 迁移验收：正式长训练前完成

1. **接口**：相同原生策略在迁移前后的平地obs、action、关节顺序、normalizer、history、PD逐项核对；实际跑四方向。
2. **reset**：导出scene×L/M/H×direction×source计数；检查qvel为零、bank SHA、板位置和初始接触。部分reset不能修改其他环境。
3. **奖励**：相同状态/轨迹下逐项比较迁移前后的值及加权积分，不只比较总reward；检查米/弧度/牛顿/力矩单位和dt。
4. **时间状态**：G历史最佳值、接触历史、完成计数、关节高负载计时、0.5 s活动窗口在reset时正确清空；L和负载统计按物理子步采集。
5. **反例**：竖直抬板不产生水平G进展；无手支撑或头高超过0.9 m不获G；其他关节运动不掩盖某关节卡滞；G−开关只改变约定项。
6. **动力学**：检查实际mass/inertia、Kp/Kd、delay和push生效，而不只展示配置开关。
7. **短训练**：先完成跨过至少一次完整10 s episode的训练和验证，再从原始checkpoint重启正式训练；不继承预检优化后的权重。

## 10. 统一评测与交付

评测20 s，按场景及四方向分别报告；名义和随机化/压力测试分开。训练、验证选点、最终测试分离；不得按每个测试场景分别挑checkpoint。

| 指标 | 口径 |
|---|---|
| Clear rate | 解除实际阻挡的比例；平地不混入该分母 |
| SR1 / SR10 | 板场景需脱困且连续满足稳定站立1 s / 10 s；平地只检查稳定站立 |
| 起身时间、最长hold、重摔 | 失败和未完成均保留，不只统计成功视频 |
| 脱困后位移 | 从脱困位置起算，报告固定窗口位移/路程；窗口不足单独标记，不填0 |
| 力矩/关节速度/机械功率peak P95 | 物理子步记录；逐trial取峰值，再跨trial取P95，失败trial也计入 |
| 持续高负载、介入/保护 | 给出阈值与时长，真机介入/保护作为事件保留 |

冻结同一稳定站立检查器，至少涵盖高度、躯干直立、身体/关节/脚速度及双足支撑。报告逐场景结果与场景宏平均，不能用训练份额加权掩盖困难场景。视频用于核验动作与判据，不代替成功率。

每位同学交付：

- 方法名称与范围、repo/commit、checkpoint SHA、机器人和obs/action/PD说明。
- 原生平地四方向评测，及迁移一致性检查。
- 最终展开的env/agent配置、reward映射表、bank及模型SHA、seed与launch记录。
- G−/G+配对训练曲线、逐项reward日志、实际随机化抽样记录。
- 每trial结果表、选点规则、未剪辑视频、负载子步日志或可重算汇总。
- 对未完成/不兼容项明确标记；不把待测填成0%成功率。

**首个里程碑：先交付可训练checkpoint说明、平地迁移验证和A6三场景reset截图/计数；通过后再启动长训练。**
