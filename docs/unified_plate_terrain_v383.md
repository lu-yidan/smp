# V3.8.3 可部署压板与复杂地形统一策略

## 目标

V3.8.2 已确认 4-frame 比 10-frame 更合适，但 H4 model_25 的右侧卧压板
成功率仍为 0%，同时 stairs-prone 是地形弱项。本阶段先做最短闭环：不改变
部署接口，只用定向 replay 判断右侧弱点能否通过数据分布修复。

任务：`Smp-Getup-Plate-Terrain-V383-Right-Deploy-G1`。

## 部署约束

actor 保持 4 帧、每帧 96 维本体感知：zero base linear velocity、IMU
angular velocity、projected gravity、joint position、joint velocity、previous
action。actor 不使用真实 base linear velocity、reset type、压板位姿/质量、
接触真值、terrain type/level 或 height map。

因此 V3.8.3 可以沿用 V3.8.2 的真机观测和 policy 导出接口。

## 短程微调

从已冻结的 `v382_h4_model25_selected.pt` 开始，并清空 optimizer state：

- 4096 env，seed 383；
- learning rate 5e-7；
- 25 updates；
- save interval 12，检查 model_12 与 model_24；
- pose weights：prone/supine/left/right = 2.0/1.5/1.0/2.0；
- 条件压板概率：0.95/0.95/0.65/1.0；
- 压板仍只出现在 flat 和 stairs-center，避免压板与 edge/rough/slope
  同时变化导致难以归因。

4096-env、seed 383 的实际 reset 审计：

- 总压板 1513/4096（36.94%）；
- prone plate 483（11.79%）；
- supine plate 361（8.81%）；
- left-side plate 167（4.08%）；
- right-side plate 502（12.26%）；
- terrain 分布为 flat 948、stairs-center 565，slope/rough 均为 0。

右侧 cohort 从 V3.8.2 的约 4.5% 提高到 12.3%，而旧 prone/supine 压板合计
仍有 20.6%，适合作为 25-update 的定向诊断，不作为长期最终采样分布。

旧 prone/supine cohort 没有被丢弃，left-side 也继续出现；新增采样主要用于
验证“right-side 0% 是数据不足还是结构性非对称”。如果 25 updates 后右侧
仍接近 0%，停止继续堆训练步数，转向动作/关节镜像和接触几何审计。

## 选择门槛

冻结 seed 后配对评测 model_12/model_24：

1. 8 kg plate 四姿态分别报告，不只报告 overall；
2. right-side 至少从 0% 变为可重复的非零成功率；
3. prone/supine/left 的平均成功率不能相对 H4 model_25 下降超过 5 个百分点；
4. flat/stairs L0/L1 保持评测不能退化；
5. penetration、peak force、joint speed 和 power 不得明显恶化。

只有满足门槛才替换 V3.8.2 selected checkpoint。

## from-scratch 对照

可以从零训练，但需要区分两种含义：

- **policy scratch**：actor/critic 随机初始化，继续使用当前已训练好的 SMP
  motion prior；这是论文中最有解释力、也最可执行的从零对照。
- **full scratch**：policy 和 motion prior 都重新训练；它同时引入先验质量、
  数据切片和 RL curriculum 三类变量，不用于第一轮快速验证。

policy scratch 不应直接从随机动作进入最困难的“压板 + 台阶边缘 + L1”混合
分布。建议固定部署观测，使用三阶段 curriculum：

1. flat 无压板/轻压板，学习安全起身和手部支撑；
2. flat/stairs-center 压板 + flat/stairs L0，学习脱困与台阶恢复；
3. 加入 stairs edge、slope、rough 和 L1，同时保留阶段 1/2 replay floor。

每一阶段通过独立冻结评测晋级，避免 aggregate reward 掩盖遗忘。scratch 训练
保存间隔使用 100--250 updates，控制磁盘占用。

## 后续顺序

1. 完成 V3.8.3 25-update 定向微调；
2. 若右侧修复且无遗忘，以它作为统一部署候选；
3. 在同一 task/obs/reward 下启动 policy-scratch curriculum 作为公平对照；
4. 再单独增加 stairs-prone replay floor，不与右侧修复同时改参数。
