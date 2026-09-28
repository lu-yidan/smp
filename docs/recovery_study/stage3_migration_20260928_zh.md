# S3 迁移到替代服务器（2026-09-28）

**当前阶段：C2/G0 正在进行 200 更新预检，尚未开始 10k 正式训练。** 两组已进入实际 PPO 更新，reset、随机化和继承检查通过。启动器会等待两组预检完成并通过配对检查，再做初始 R2 评测，最后重新加载原始 R2 开始正式训练；任一步失败都会阻止正式启动。这个状态不等于预检完成或训练完成。

## 服务器与资源

- SSH：`ssh -p 28036 root@10.177.36.91`；认证信息不写入仓库。
- 工作区：`/root/workplace/smp-stage3`；独立 `.venv`，不修改其他任务环境。
- 分支：`codex/recovery-study`。训练代码来自干净提交 `60d7e42`，与 `10df7e1` 的训练实现无差异；未带入本地未提交改动。
- 新主机可见 6 张 RTX 4090，单卡报告约 48 GiB 显存。GPU0/1/3/4/5 曾分别通过 CUDA 运算检查；GPU2 原有任务、GPU4 随后出现其他负载，均未停止或占用。

|GPU|本次安排|
|---|---|
|0|S3-C2：保留 SMP、几何奖励和受阻任务调制|
|1|S3-G0：保留 SMP，移除稠密几何奖励并令 α 恒为 1|
|2、4|已有其他任务，保留|
|3|暂不使用|
|5|R2 评测、渲染验收及后续串行 checkpoint 验证|

其余五组 S3 尚未恢复，也没有新增“几何奖励/α 分拆”消融。先确认核心配对的运行可靠性与首个正式验证结果，再恢复其余组。

## 迁移一致性

旧服务器已拒绝 SSH 连接，使用本地备份恢复：

- R2@9000 SHA256：`8b4889f80c6b7cc675f6e9b1e98f2d4a1886a15e372f86070f63c329ba9ca5d1`。
- 训练 bank：4176 条，`1f6c2214926477f795c280f81337f2c915f4268c8f7a35cfeec9bbfa41c3d3a9`。
- 验证 bank：1392 条，`55add90e3dbb81956509cb8cbfa0ab40adac60c5abc985ad6b6ea4d6fb366f21`。
- 训练入口与 `src/smp` 的联合代码哈希：`1fcbe926ecf83b4365d3ee5235538c1b97a3169edc494fa1f9a9725d162d71f9`，与旧服务器最终初始 R2 评测记录一致。
- 新预检两组的 actor、fresh critic、初始 qpos、bank、代码哈希相同。不是从旧故障运行的 `model_0.pt` 续训。

环境按本地可用评测环境固定为 Python 3.12、PyTorch 2.10.0、MuJoCo 3.8.1、MuJoCo Warp 3.8.0.2、Warp 1.12.1、mjlab 1.3.0、rsl-rl-lib 5.2.0。mjlab 与 MuJoCo Warp 分别从缓存的 `731fa27`、`88b55fc` 源码构建 wheel，并逐文件确认其 Python 源码与本地安装版本一致。旧服务器完整依赖快照未取回，因此不声称全部运行库逐位相同；新主机需要重新完成基线和动力学验收。

新机器已完成 224 次 R2 渲染/评测验收，导向板与自由板视频成功导出。新旧 actor、critic、初始 qpos、bank 和代码哈希一致；计划 JSON 唯一字段差异是记录状态由 `IMPLEMENTED_PENDING_PREFLIGHT` 改为 `IMPLEMENTED_PREFLIGHT_PASSED`，奖励与采样参数没有变化。但跨主机 rollout 结果并非逐位相同，不能把新旧结果混合当作同一运行环境的配对试验：

注意：验收视频叠字 `S3-C2` 指环境配置，视频策略仍为原始 R2，不是完成适应训练的 C2。

|连续站立10秒（每场景32次）|旧主机 R2|新主机 R2|
|---|---:|---:|
|平地|32|32|
|导向压板|0|0|
|自由板|10|12|
|固定顶板|0|1|
|交叉木条|6|6|
|梯架＋板|7|5|
|双自由板|8|7|

这些差异的具体来源尚未确定，不能仅凭计数归因为某个驱动或求解器版本。后续消融统一使用新主机环境和基线。上述为原始 R2 的开发验证，既不是 S3 训练成果，也不是论文最终测试。见[新机 R2 汇总](evidence/stage3_migration_20260928/initial_R2_rendercheck/summary.json)。

随机种子仍为 20260927；4096 环境、24 步/更新、10k 更新、每 500 更新保存与验证。训练 episode 10 秒，验证 rollout 20 秒。奖励、场景/reset 比例和质量/惯量、Kp/Kd、命令延迟、摩擦、COM、push、actor 观测噪声保持原计划。

## 运行与查看

[启动器](../../scripts/recovery_study/launch_stage3_migration_20260928.sh) 保留失败日志，不覆盖已有目录。预检训练产生的权重不会用于正式初始化。

```bash
ssh -p 28036 root@10.177.36.91
cd /root/workplace/smp-stage3
# 当前阶段、预检失败原因或正式启动信息：
tail -n 20 outputs/preflight_stage3_migration_20260928/controller.log
# 预检实际 PPO 进度：
grep 'Learning iteration' outputs/preflight_stage3_migration_20260928/S3-C2.log | tail -1
grep 'Learning iteration' outputs/preflight_stage3_migration_20260928/S3-G0.log | tail -1
```

- 预检：`outputs/preflight_stage3_migration_20260928/{S3-C2,S3-G0}`。`completed.json` 才代表 200 更新完成，`failed.json` 表示失败。
- 单独的渲染/评测验收：`outputs/initial_stage3_R2_migration_rendercheck_20260928`。用于提前检查 EGL、视频编码和评测通路。
- 正式训练前基线：`outputs/initial_stage3_R2_migration_20260928`。
- 正式训练：`logs/rsl_rl/stage3_recovery/formal_20260928_replacement_v1/{S3-C2,S3-G0}`。
- 每 30 秒记录显存/利用率：`outputs/preflight_stage3_migration_20260928/gpu_memory.csv`，监测程序最长运行 8 小时。
- W&B 已验证认证与联网；正式训练使用 `tabletennis/smp`，group 为 `stage3_R2_20260928_replacement_v1`。预检使用 TensorBoard，不会冒充正式 W&B 实验。

状态与证据：[迁移快照](evidence/stage3_migration_20260928/manifest.json)、[依赖版本](evidence/stage3_migration_20260928/runtime.json)、[完整依赖固定清单](evidence/stage3_migration_20260928/runtime-pins.txt)、[C2 预检配置](evidence/stage3_migration_20260928/preflight/S3-C2/launch.json)、[G0 预检配置](evidence/stage3_migration_20260928/preflight/S3-G0/launch.json)。PID 和进度都是核查时快照，不能据此宣称后续仍在运行。
