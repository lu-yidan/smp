# R2第三阶段消融启动记录（2026-09-27）

**已启动并核验7组均进入正式PPO更新，W&B在线。** 核验时间 2026-09-27T04:55:14.459540+00:00；代码 `10df7e1`。当前为首个配对种子筛选，不是多种子论文最终结果。没有启动AMP/HoST/FIRM。

运行记录：[run_manifest.json](evidence/stage3_20260927/run_manifest.json)。PID与核验时进度是当时快照，不代表未来状态。

## 运行矩阵

所有组继承FT_R2@9000 actor/obs normalizer，fresh critic/optimizer/iteration；不会继承短程预检训练后的权重。

|GPU|实验|场景|本阶段SMP|稠密脱困引导|新增软位置代价|
|---|---|---|---|---|---|
|0|[S3-C2](https://wandb.ai/tabletennis/smp/runs/mvj3vd4w)|扩展|有|有|无|
|1|[S3-C3](https://wandb.ai/tabletennis/smp/runs/cjqk60kl)|扩展|有|有|有|
|2|[S3-G0](https://wandb.ai/tabletennis/smp/runs/wowlmdmw)|扩展|有|无|无|
|3|[S3-P0](https://wandb.ai/tabletennis/smp/runs/8n716o2v)|扩展|无|有|无|
|4|[S3-C0](https://wandb.ai/tabletennis/smp/runs/7tfgxs6h)|三场景|有|有|无|
|5|[S3-C1](https://wandb.ai/tabletennis/smp/runs/pnjfpyh2)|三场景|有|有|有|
|6|[S3-PG0](https://wandb.ai/tabletennis/smp/runs/lkgv77nk)|扩展|无|无|无|
|7|串行验证 / 初始R2评测|统一七场景|—|—|—|

4096环境×24控制步×10000更新，seed=20260927；每500保存并验证，每2000及最终保存四场景视频。训练10s，验证rollout20s；不站起终止、不低SMP终止，保留数值异常与异常接触保护。部件mass/inertia、Kp/Kd、0–10ms延迟、摩擦、COM、push和actor观测噪声保留。探索std固定0.3。

扩展份额：平地20%、导向板15%、自由板20%、固定顶板15%、木条10%、梯架+板10%、双自由板10%。三场景份额20/40/40。平地低/中/近站立40/40/20，其余全部低位。低位方向均匀，各方向LAFAN/程序化75/25；最大余数法整数化，精确配额在每组launch.json。

自然低位按原LAFAN bank的stage与pelvis方向标注，包含自然过渡姿态，并不全是完全躺平；程序化部分补规范化躺地。新bank进行了100ms接触支撑沉降，按方向重新核验并归零qvel。模型随机化期间启动板位置在几何上受限，活动板由物理重力/接触开始运动。

## 位置、文件与复现

- 本地工作树：`/home/luyd/workspace/smp-a6-egress`，分支 `codex/recovery-study`。
- 新服务器工作区：`/root/workplace/smp-stage3`，不覆盖旧M结果。
- 正式输出：`logs/rsl_rl/stage3_recovery/formal_20260927_v1/<arm>`。
- W&B project：`smp`；group：`stage3_R2_20260927_v1`。
- 新入口：`scripts/train_stage3_recovery.py`；配置：`configs/recovery_study/stage3_comparison_plan_v1.json`。
- 启动器：`scripts/recovery_study/launch_stage3_20260927.sh`。要求所有预检完成；重复启动会拒绝覆盖已存在结果。
- 新bank：`outputs/stage3_bank_v1/train.npz`（4176）、`validation.npz`（1392）；manifest记录源bank/几何/生成器SHA。
- 初始R2：`outputs/initial_stage3_R2_final`（汇总）、`outputs/initial_stage3_R2`（视频）；验证是开发数据，不是论文heldout成功率。
- 预检：`outputs/preflight_stage3/<arm>_4096`，24 PPO更新（11.52s控制时间，跨过10s超时边界）；正式训练重新加载原R2。

## 实际验收与日志解释

预检检查actor与normalizer精确继承、critic不继承；actor93D、critic960D；所有初态CPU/GPU geometry中心/旋转一致、穿透不超过2.1mm、qvel=0、上方约束真实存在；部分reset不改变其他环境状态/尺寸/质量/位置锚点。奖励实际输出与SMP/引导开关组合做数值核验。7组均完成24更新预检；最深初态接触约−1.905mm，CPU/GPU位置误差≤1.91e−6m，旋转误差≤2.58e−7。4096环境显存约12.7GB。扩展场景5组的初始actor、critic、qpos、bank哈希完全一致；三场景C0/C1也配对一致。

预检修复了固定顶板单geom的MuJoCo same-frame快捷路径：固定体也使用与geom不同的显式惯性坐标，防止reset改变geom位置后CPU仍保留编译位置。此问题在正式训练前通过bank筛查发现并修复。

`Recovery/<scene>/smp`在无SMP组仍是诊断分数（计算但不进入奖励），`Stage3/smp_applied`为0。实际任务奖励为T或S×T，是否受阻调制由dense_guidance决定。G−仅关闭稠密进展与受阻时正任务门控，保留完成/接触/脚部安静/持续堵转成本。锚点仅C1/C3有负权重，其余组也记录候选raw值，便于比较。

`RecoveryStage/*`记录阶段占比；`Stage3Reward/*`是未加权原项；`Episode_Reward/*`是训练框架记录的加权episode项。`Validation/*/stable_10s`才是统一初态的连续安静站立10s指标，不能把在线stable比例当成功率。验证同时记录清障、再跌倒、失败/保护、逐方向表现、2ms子步力矩/速度/功率峰值P95。失败计入负载指标。

新增 `post_clear_path_mean`、`post_clear_max_offset_p95` 使用首次清障之后完整3s窗口，样本数为 `post_clear_3s_n`；窗口不完整的样本不当作0。另有首次连续站稳1s后完整3s窗口的 `post_stand_*`，用于区分起身位移和站立漂移；平地的post_clear从初始无障碍时刻开始，不能解释为站立后的位移。三场景组也在全部七场景验证，测量未训练场景泛化。训练解除判据为逐物体分件包围几何，验证同模型指标仍是开发代理；梯架挂住等情况需独立CPU接触/上移探针与视频复核，不能把该开发指标直接当最终论文成功率。

## 后续

先比较C2/G0/P0/PG0的引导×先验效果，再比较C2/C3、C0/C1的位移与成功率trade-off。不得仅因某组reward较高就选点。完成后再做独立seed与冻结heldout，不改本批比例、速度目标或奖励权重。
