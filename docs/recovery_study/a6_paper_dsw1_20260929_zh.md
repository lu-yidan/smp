# A6 主线：DSW1 配对奖励消融

## 目的与边界

论文主线采用历史 A6。这里回答的是：**相同起点、场景和 reset 下，脱困引导、任务门控、手部支撑和动作约束分别起了什么作用？** 不把后续 S3 扩展场景的结果混入本表，也不把正在迁移的 AMP / HoST / FIRM 对比替代为奖励消融。

- 分支：`codex/a6-paper-ablations`。
- 历史基线：`1b6d7e61ddbcd2caf9ea17b02160dd1cf0d119d2`。
- 新入口：`scripts/train_a6_paper_ablation.py`；旧 A6 源码保持原样。
- 服务器：`ssh dsw1-clone`。
- 工作目录：`/mnt/workspace/user/luyidan/smp-a6-paper`。
- 本次是一个配对种子 `20261013` 的第一轮；不能据此声称多种子统计显著性。根据结果再补关键组的独立训练种子。

## 七组实验

| GPU | 组名 | 相对 A6_full 的唯一干预 | 要回答的问题 |
|---|---|---|---|
| 0 | A6_full | 无 | 本机重训的配对基准 |
| 1 | A6_no_bundle | 三项几何奖励置零，同时 α=1 | 整套脱困机制是否有用 |
| 2 | A6_no_geometry | 三项几何奖励置零，保留 α | 保留任务门控时，几何信号是否有用 |
| 3 | A6_no_alpha | 仅 α=1 | 保留几何信号时，任务门控是否有用 |
| 4 | A6_no_hand | 仅把 G 中的手部支撑乘子设为 1 | 是否需要引导有支撑的低位移动 |
| 5 | A6_no_Q | 仅 Q 的权重置零 | 临近直立时额外足速惩罚的作用 |
| 6 | A6_no_L | 仅 L 的权重置零 | 持续高负载、低运动量惩罚的作用 |
| 7 | 评测与录像 | 不训练 | 串行完成统一评测 |

“三项几何奖励”明确指 `plate_geometry_progress`（G，权重 .45）、`plate_clearance`（稠密清空，.08）、`plate_separation`（分离，.01）。**仍保留** `plate_completion`（.6）和 `plate_force`（−.03），所以 no_geometry 不能称为“完全没有脱困奖励”。

α 在板下尚未脱困时为 .05，否则为 1；它乘在 SMP × 起身/安静任务的组合上。no_hand 保留完整 Q/L；虽然内部复用历史 A3 的去手部乘子分支，**本组不是历史 A3**。

前四组形成“几何奖励 × α 门控”的 2×2 对照。后三组分别检查完整机制中的一个组成部分；其他奖励及权重不动。不在这批中加入新场景、扩大随机化或修改脱困后分离奖励，以免混淆归因。

## 共同训练设置

- 全部从同一 `R2_9000.pt` 的 actor 和 actor 观测归一化开始；critic、优化器和学习计数重新初始化。
- actor 93D，critic 960D；SMP 使用历史 f2s2 prior、ws=6。
- 每组 4096 环境、10,000 PPO 更新；每 500 更新保存。
- 保留历史 A6 的 10 s 训练 episode；站起后继续，没有低 SMP 提前终止。
- 场景：平地 50%、上下滑动压板 25%、自由刚体板 25%。这里“上下滑动压板”不是完全固定的悬空桌面。
- 初始进度：低位约 70%、中段约 20%、后段约 10%。低位四方向均匀，继续使用历史自然片段与程序化状态的混合，未改成全部低位。
- 4096 环境实测：场景 `[2048,1024,1024]`；低/中/后 `[2868,820,408]`；自然/程序化 `[3380,716]`。这些是 reset 分配计数，不是 episode 实际步数占比，也不是纯 LAFAN。
- 保留 push、actor 观测噪声、机器人分部件质量/惯量、分组电机 Kp/Kd 及命令延迟随机化。机器人质量和增益范围由 ±10% 在前 2000 更新扩展到 ±20%；约 25% 为标称，随机化样本延迟取 0、2、4、6、8、10 ms。Kp/Kd 使用同组缩放因子。
- 板的参数与历史 reset 原样继承；这次不额外扩大板尺寸或质量分布。

SMP 奖励归一化使用已导出的 `configs/a6_paper/smp_reward_reference_v1.json`。其来源为历史 FT/model_12000.pt 的 `infos.scratch_tradeoffs.mean/count`；这只是冻结的奖励统计量，**不是额外的 actor 微调阶段**。本机没有原始完整 FT 参考 checkpoint，使用校验过的 JSON 数值恢复。其来源与 SHA 在 JSON 中记录。

## 启动检查与当前状态

2026-09-29 启动记录：环境配置与 CUDA 已配置好；七组已启动 **4096 环境 × 200 更新预检**。保存证据时七组均通过配置、实际动力学、局部 reset 与奖励干预检查，但尚未完成 200 更新。正式 10k 是否已启动，应以控制器日志为准。

控制器 `scripts/launch_a6_paper_dsw1.sh` 会依次：

1. 检查完整 A6 配置与历史版本相同、每个消融仅修改指定字段。
2. 跑完全部七组 200 更新预检，要求无失败且都写出完成标记。
3. 核对七组相同的 actor/critic/初始 qpos、数据 SHA、代码 SHA 及 reset 计数。
4. 在 GPU7 评测原始 R2 的标称动力学与 upper-mass ×1.3。
5. 全部通过后，各组**重新从原始 R2 开始**正式 10k；不用预检得到的权重。

历史配置保留名为 `stood_up` 的日志键，但函数恒为 false。新检查验证其实际行为，不把键名误当成站起终止。

- 控制器日志：`outputs/preflight_a6_paper/controller_v1.log`。
- 预检：`outputs/preflight_a6_paper/formal_size_v1/<arm>`。
- 正式结果：`logs/rsl_rl/a6_paper/formal_20260929_seed20261013/<arm>`。
- W&B：`tabletennis/smp`，group `a6_paper_20260929_seed20261013`；预检仅写本地 TensorBoard，正式训练才上传 W&B。
- 配对启动证据：[evidence/a6_paper_dsw1_20260929](evidence/a6_paper_dsw1_20260929)。它是预检启动证据，不是最终实验结果。

```bash
ssh dsw1-clone
cd /mnt/workspace/user/luyidan/smp-a6-paper
tail -n 20 outputs/preflight_a6_paper/controller_v1.log
nvidia-smi
```

## 统一评测与论文使用

每次保存，在 GPU7 上用文件锁串行评测两套动力学：标称，以及上身质量/惯量 ×1.3。每套 192 环境、20 s，平地/滑动板/自由板各 96/48/48，每类四方向均匀；评测全部低位，禁用训练时随机 push/观测噪声和随机动力学。评测种子固定，使用 validation bank。正式训练每 2000 更新及最后 checkpoint 录制四姿态拼图。

记录：脱困比例、连续站稳 1 s/10 s、再次跌倒、终止、脱困/站稳时间；接近直立时换脚次数与足部滑移；力矩、关节速度和机械功率的峰值 P95；高负载低速累计时长与连续高负载最长时长。力矩与功率在物理子步采样，使用各 trial 的峰值再跨 trial 取 P95；不是把全部时刻混在一起取 P95。失败 trial 不从负载统计中删除，自动 reset 后不算成第二次成功。

脱困后 3 s 位移只统计有完整窗口的 trial，并同时给出有效样本数。平地的 clear 从起始时刻成立，所以该项包括起身过程，**不能称为站立后漂移**。换脚与滑移累积量也受处于直立区间的时长影响，需要与成功率共同解释。

这 192 个固定 trial 用于训练过程验证和选 checkpoint，不能充当最终独立测试。论文最终需要冻结选模规则、单独的 held-out reset/障碍采样和多种子复现。不能把本入口指标直接与历史旧评测的数字合并，应把历史 A6 和外部方法放入同一测试协议重评。

GPU7 已完成原始 R2 的三场景渲染检查，位置为 `outputs/a6_paper_R2_rendercheck_v2`。视频上 A6_full 表示环境配置，actor 仍是 R2；**不是新 A6 已训练完成的视频**。

## 运行环境

8 × RTX PRO 5000 72GB；Python 3.12；Torch 2.10.0+cu128；MuJoCo 3.8.1；mujoco_warp 3.8.0.2；Warp 1.12.1；mjlab 1.3.0；rsl-rl-lib 5.2.0。运行环境为 `/root/workspace/smp-a6-env`，仓库 `.venv` 指向它。

服务器已有 NVIDIA EGL 动态库但缺 vendor 注册；补齐 `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` 后，最小 MuJoCo 离屏渲染与完整 R2 录像均通过。没有重启服务器或修改训练物理参数。

另一服务器的 S3 C2/G0 继续独立运行。它们属于扩展场景研究，不替代这里的 A6 配对消融。
