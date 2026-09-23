# 上方杂物开发集：R2 / A6 / M2@2500（2026-09-23）

已完成3策略×（48受限＋4平地校准）=156次20s试验。所有初态配对，留出480例未运行，现有M训练未改变。M2是开始评测时固定的2500更新中途checkpoint，不是最终或最优checkpoint，也不是同总预算对照。

## 模型与复现

- R2：FT_R2@9000，checkpoint SHA `8b4889f80c6b7cc675f6e9b1e98f2d4a1886a15e372f86070f63c329ba9ca5d1`。
- A6：V33Path-A6@9999，SHA `4a2d4c8f0e710f760b0996237002824b4fd05872482ee96fb94f7330d09c8fd8`。
- M2：formal_20260923_v2/M2_A6_mix@2500，SHA `9340b57ee868f14c6b2ee0f5bbf12661dd7c1ab66648c0af7a47726546e6d8b6`，父代A6。

原生CPU MuJoCo 3.8.1，nominal机器人动力学，无push和观测噪声，场景质量/摩擦保留冻结值。这是统一评测条件，不代表训练关闭随机化。每次只运行一个20s trial，不因站起/低SMP reset。只在数值不稳定时停止，并记失败。

93D单帧actor，保留原checkpoint归一化，50Hz推理、500Hz饱和PD、先保持初态20ms，再执行10控制步启动插值；50Hz目标位置限幅与RoboMimic_Deploy一致。控制数组完全一致，R2/A6输出对既有部署导出fixture误差0和4.77e−6；IMU角速度/重力与MuJoCo传感器独立复核。三组四个平地校准案例全部通过站稳10s；全部156例无数值失败。

证据：[协议](evidence/clutter_eval_v1/protocol.json)、[接口检查](evidence/clutter_eval_v1/preflight.json)、[分组结果](evidence/clutter_eval_v1/summary.json)。原始结果和离线复核分别保存，没有覆盖原协议。

## 两个判据分开报告

视频发现部分策略已直立，但木条仍挂在手臂/腰旁，或箱子套在身上；不能把这些都解释为“没有学会站起”，也不能只看高度就认定完全脱困。开发阶段因此补报独立站稳指标；不是在留出测试后更换指标。

- **Q：安静站立。** 连续满足：相对脚部支撑面头高≥1.15m，upright≥0.93，双膝绝对角<0.8rad，根部速度<0.15m/s、角速度<0.3rad/s，关节速度RMS<0.5rad/s，双脚最大速度<0.1m/s，双脚各有>20N向上支持力，非脚外部接触竖直力绝对值总和<20N，脚间距0.12–0.45m，脚下支撑接触点速度<0.15m/s。它允许轻微非脚接触，仍需视频检查残留物。
- **E：Q且直接上移路径畅通。** 对实际碰撞分件做固定关节的竖直平移检查，10Hz采样，1cm步距，抬升到头高1.25m、限制在0.10–0.65m范围；连续0.3s没有>0.5mm碰撞穿透才记clear。C形架只把顶板列为上方物；梯架/箱体不填充空洞。
- E是保守的几何诊断，不是任意关节运动的可达性判定、卡死证明或人工审核后的论文成功率。挂在手上的木条也可能使它不通过。
- 原始trial成功字段`success_10s`对应E；离线复核新增`quiet_stable_10s`对应Q。复核从保存状态和实际控制恢复接触力，逐例验证原E最长保持时间误差≤0.04s。

## 配对结果

每格为 **Q / E**，每类12例。两者都要求20s内连续保持10s。

|场景|R2@9000|A6@9999|M2@2500|
|---|---:|---:|---:|
|交叉木条|8 / 6|12 / 11|12 / 8|
|梯架＋木板|9 / 5|11 / 9|12 / 12|
|固定C形空间|4 / 3|4 / 4|11 / 11|
|空心容器|8 / 6|8 / 7|8 / 8|
|总计（48例）|29 / 20|35 / 31|43 / 39|

固定C空间：R2/A6经常仍卡在低位，M2有明显改善。交叉木条：A6与M2均12/12站稳，但M2只有8/12同时上移畅通，不能把它整体判为优于A6。空箱仍有套在身体上、扶箱站立和一直低位失败；这是需要单列的接触残留失败类型。

## 执行器负载和移动

力矩/速度/功率每2ms记录。先对每个trial取所有29关节、全时程绝对峰值，再在48个trial峰值间算P95；包含失败。机械关节功率为abs(tau×dq)，不是总机器人电功率。力矩受模型限幅，不能据封顶值认定安全。

|指标（48例）|R2@9000|A6@9999|M2@2500|
|---|---:|---:|---:|
|力矩峰值P95 / Nm|137.68|139.00|132.04|
|关节速度峰值P95 / rad/s|13.55|18.07|17.20|
|关节机械功率峰值P95 / W|571.20|1005.63|820.47|
|单trial最重负载关节累计>90%力矩上限时长P95 / s|17.29|12.00|1.46|

A6的尾部功率/速度高于R2；M2相对A6降低，但仍有高峰值。持续大力尤其发生在未完成起身的固定C案例；R2最严重右踝pitch累计19.28s超过90%上限，A6有腰pitch和右肩pitch长时高负载。没有真机电流/温度模型，这些不是可直接上机的结论。

同一批三策略均达到E-clear的17例：从首次clear直到20s，平均base XY路程R2 **0.190m**、A6 **0.608m**、M2 **0.976m**。它们是累计路程而非直线位移；不同策略clear时刻不同。全库从首次Q保持1s之后的平均路程约0.012/0.022/0.051m（分母30/38/45，不直接作为配对比较）。主要移动发生在解除阻挡到稳定站立之间，而非最终站稳后持续走。

本轮Q保持1s后未观察到再倒地，但初次起身阶段的反复失败和高负载仍存在。不能用这一指标掩盖未完成起身的案例。

## 视频

每个视频20s、1倍速；列：R2/A6/M2，行：仰卧/伏卧/左侧/右侧。同场景都固定layout1、variant0，四方向全部展示，没有按成功筛视频。摄像机跟随base XY，字幕保留世界坐标。`quiet`为Q连续保持，`clear hold`为E连续保持，`blocked`仅指上移探针。

- [交叉木条四方向三策略](/home/luyd/workspace/smp-a6-egress/outputs/clutter_eval_v1/videos_reviewed/crossed_timber_paired_20s.mp4)
- [梯架＋木板四方向三策略](/home/luyd/workspace/smp-a6-egress/outputs/clutter_eval_v1/videos_reviewed/ladder_boards_paired_20s.mp4)
- [固定C形空间四方向三策略](/home/luyd/workspace/smp-a6-egress/outputs/clutter_eval_v1/videos_reviewed/fixed_c_space_paired_20s.mp4)
- [空心容器四方向三策略](/home/luyd/workspace/smp-a6-egress/outputs/clutter_eval_v1/videos_reviewed/hollow_container_paired_20s.mp4)

## 输出与命令

完整本地输出：`/home/luyd/workspace/smp-a6-egress/outputs/clutter_eval_v1`。`run/<policy>/<case>.npz`包含50Hz qpos/qvel/93D观测/原始动作和500Hz tau/dq；`review`补充分离指标。大文件保留outputs，Git只归档协议/分组证据。

```bash
cd /home/luyd/workspace/smp-a6-egress
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/luyd/workspace/smp/.venv/bin/python scripts/recovery_study/evaluate_clutter.py \
  --root outputs/recovery_study/overhead_clutter_v2/development \
  --policies outputs/clutter_eval_v1/policies.json \
  --calibration outputs/clutter_eval_v1/calibration \
  --out outputs/clutter_eval_repeat --workers 6
```

输出目录必须不存在。不重采样初态、不自动推进到留出集；新硬件/环境应先重跑控制与碰撞检查。

## 接下来

先保留A6作为冻结基线，继续观察原有M训练；待约定更新点再用同一开发集比较。优先解决过渡阶段多余移动、残留物挂住，以及固定空间中的持续饱和负载；暂不因为单次开发集结果为每种道具另训policy。正式论文测试前需冻结任务成功语义、实物质量尺寸、机器人动力学压力测试和新的独立测试库。
