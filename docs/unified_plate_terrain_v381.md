# V3.8.1 能力均衡统一恢复

## 目标

V3.8 的压板只覆盖 reset type 1/2（俯卧、仰卧），侧卧恢复实际上属于无压板样本。V3.8.1 在不改变部署观测接口的前提下：

- 加入左侧卧和右侧卧压板；
- 保留约四分之一到三分之一压板 episode，防止压板能力被地形训练遗忘；
- 增加俯卧、slope L1 和 stairs 困难样本；
- 从 V3.8 model_100.pt 小学习率续训，不从最后一个已遗忘的 checkpoint 续训；
- 每 100 iterations 保存并冻结评测，不按总 reward 选择模型。

任务 ID：Smp-Getup-Plate-Terrain-V381-Deploy-G1。

## 部署观测

actor 仍是 4-frame term-wise history。单帧 96 维，总计 384 维：

| term | 单帧 | 4 帧 |
|---|---:|---:|
| zero base linear velocity | 3 | 12 |
| IMU angular velocity | 3 | 12 |
| projected gravity | 3 | 12 |
| joint position | 29 | 116 |
| joint velocity | 29 | 116 |
| previous action | 29 | 116 |
| 合计 | 96 | 384 |

actor 不包含真实 base linear velocity、terrain label/level、plate pose/mass、reset type 或接触真值。critic 保持 10 帧、960 维，只在训练中使用。

## 能力均衡采样

- pose weights（prone/supine/left/right）：2.0/1.5/1.0/1.0；
- terrain proportions（flat/slope/stairs/rough）：0.25/0.25/0.35/0.15；
- 非平地 level floor：L0 55%，L1 45%；
- 四种程序化倒地姿态均可激活压板；
- 条件压板概率由 0.90 降到 0.55，避免加入侧卧后压板 episode 翻倍；
- 压板仍只用于 flat 和 stairs-center；slope、rough 和 stair edge 不放水平压板；
- prone/supine 必须在 12 steps 内接触压板，side pose 放宽到 20 steps；
- PPO learning rate 从 5e-6 降至 2e-6，save interval 为 100。

256-env reset 审计得到 59 个压板 episode（23.0%），四种姿态均有压板；压板未出现在 slope、rough 或非 center stair cohort，所有 qpos 有限。

## 训练前冻结基线

使用 V3.8 model_100.pt、8 kg 压板、四姿态、256 env、500 steps：

- 总 escape-and-stable-stand：26.67%；
- prone：41.67%；
- supine：40.00%；
- left side：25.00%；
- right side：0%；
- setup invalid：1.90%；
- invalid contact：30.48%；
- peak penetration：16.5 mm；
- peak contact force：3.74 kN。

右侧卧压板是明确的新增弱项。训练后必须分别报告左右侧卧，不能只报告四姿态平均值。

## 训练

将 V3.8 主候选复制到新 experiment 的 seed 目录后：

    uv run scripts/train.py Smp-Getup-Plate-Terrain-V381-Deploy-G1 \
      --env.scene.num-envs 4096 \
      --agent.resume True \
      --agent.load-run v38_model100_seed \
      --agent.load-checkpoint model_100.pt \
      --agent.max-iterations 300 \
      --agent.run-name v381_ability_balanced_from_v38_model100

## 选择门槛

V3.8 seed checkpoint 自带 iteration 100，因此续训后的首个百步 checkpoint
是 model_200，之后是 model_300 和最终 model_399；model_100 仅相当于
加载后的第一次更新。冻结评测 model_200/300/399，选择时同时满足：

- prone/supine 8 kg 压板不明显低于 V3.8 model_100；
- left/right side 压板明显高于训练前基线；
- flat recovery 不低于 95%；
- stairs、slope、rough 分姿态报告，不用总体均值掩盖弱项；
- 无 invalid dynamics；
- penetration、contact force、joint speed、torque、power 不恶化；
- 站立后脚间距和 foot speed 继续记录。

若第一个 100-iteration checkpoint 已出现压板遗忘，立即停止，不继续跑满 300。
## 2026-08-27 训练与早停结果

- W&B：tabletennis/smp/t7x0jb1d；
- 从 V3.8 model_100（iteration 100）续训；
- model_200 生成后立即冻结评测；
- 训练进程在 W&B step 243 终止，未继续生成 model_300/399。

同一 512-env、8 kg、四姿态、750-step 压板评测：

| checkpoint | overall | prone | supine | left side | right side |
|---|---:|---:|---:|---:|---:|
| V3.8 seed | 33.50% | 55.10% | 46.94% | 28.33% | 0% |
| V3.8.1 model_200 | 19.00% | 20.41% | 40.82% | 13.33% | 0% |

model_200 的 flat/stairs L0/L1 prone/supine 均值为 56.05%，只比 seed
的 54.69% 高 1.36 个百分点；同时最大压板接触力达到 12.49 kN，
penetration 达到 19.2 mm。因此该 checkpoint 不满足任何综合选择门槛。

本次失败的主要采样原因：总压板 episode 约 23%，但四姿态共享后，
原有 prone/supine 压板 cohort 下降到约 15%，低于 V3.8 的约 26%。
全局 obstacle probability 不能同时“保留旧压板技能”和“增加侧卧技能”。

model_200 仅作为 rejected audit 保存在服务器：
/mnt/workspace/user/luyidan/baselines/G1_Recovery_Below_Block/v381/audit/，
SHA256 为 d3c03b735768ec5bb02bf42dc4d3a3f890e52032638210d680b674f8201c478b。
V3.8 model_100 继续作为有效部署基线。
