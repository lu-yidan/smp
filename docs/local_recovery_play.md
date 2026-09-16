# 本地比较 L4 与 V33-FT 12k

新增播放器使用 `/home/luyd/workspace/smp-flat93` 的已审计部署对齐评测环境，加载 master 架构的93D actor/960D critic及checkpoint内观测归一化。与训练reset比例无关：这里选定验证库中的同一个姿态便于比较。默认关闭actor噪声、自动push和域随机化；仍可在native viewer中人工施扰。一次episode20秒，不按站起/低SMP结束。

两份checkpoint已复制到 `/home/luyd/workspace/smp-master-repro/outputs/recovery_local_play/checkpoints/`：
- L4_9999.pt：SHA256 7e98c06aa72f4ced7ab1f5bcfc67c011096fb5b3b4cdc453e43f7aac97409813
- V33_FT_12000.pt：SHA256 8f05543b644b0a1d11246e85778f99220460ef2a744417ea940296eed768c768

```bash
cd /home/luyd/workspace/smp-master-repro
./scripts/play_recovery_local.sh --preset ft12k --pose supine --speed 0.25 --paused
./scripts/play_recovery_local.sh --preset l4 --pose supine --speed 0.25 --paused
```

退出前一个窗口再运行下一条。无需切换分支、重新安装环境或替换部署模型。

- `--pose supine|prone|left|right|all`：固定方向；all同时四个环境。
- `--source procedural|natural`：程序化或自然LAFAN低位验证库。
- `--sample 0`：该方向库内序号；换1、2等查看不同姿态。两checkpoint使用相同参数可比较相同初始qpos。
- `--checkpoint /absolute/path/model.pt`：替代preset，加载其他同架构93D checkpoint。
- `--speed .25`只改变墙钟播放速度，不改变物理步长、控制频率和策略。

键盘：Space暂停/继续；暂停后右箭头前进一个20ms控制步；Enter重新开始相同姿态；减号/等号减速/加速；逗号/句号切换查看的环境；A显示全部；P切换奖励图。注意评测环境的SMP/奖励不是训练奖励，不能据此比较训练SMP分数。

播放器两份checkpoint均已通过本地四方向100控制步无界面冒烟检查，确认93D输入、有限奖励、正常加载。未在用户桌面自动打开交互窗口。

## 统一负载比较

原始L4是本次单独重新测量，没有使用FT更新4次后的近似。规整四方向128环境、20秒、每2ms采样；每环境取所有关节/全过程峰值，再求跨环境P95。

|checkpoint|严格稳定10秒|首次直立中位数（曾直立样本）|速度峰值P95|功率峰值P95|力矩峰值P95|头部竖直速度峰值P95|
|---|---:|---:|---:|---:|---:|---:|
|L4 9999|128/128|1.30s|34.08rad/s|2681W|139Nm|3.05m/s|
|FT 12000|119/128|3.00s|17.00rad/s|844W|139Nm|2.56m/s|

FT12k另外的自然低位400/408、程序化496/512（伏卧114/128）。不同姿态分布结果不能混为同一个成功率。指标是理想PD下仿真关节机械功率，不是电池/驱动器输入功率，也不是电机额定值。全体负载统计会受失败样本影响，应同时查看成功率。单seed、标称动力学评测尚非真机安全验证。

## 速度与坐姿过渡

0.20m/s是头部竖直速度绝对值的超速代价阈值。阶段目标实际为.06/.08/.10/0m/s，并随剩余高度减小。这不是硬速度上限。FT12k仍出现2.56m/s的头部速度峰值，表明软代价没有保证慢速。

从FT12k仰卧样例可见收腿与躯干运动连续完成；仅看连续轨迹不能证明静止坐姿可达蹲姿。暂停播放器保留速度状态，慢放也不移除动量，均不能替代中间姿态零速度reset测试。下一步应独立测试坐姿/深蹲接地状态（接触有效），以及部署端摩擦、延迟和执行器限制下脚能否收至合适支撑位置。半蹲不稳可能涉及支撑与身体前移，单独提高竖直速度不足以验证原因。

FT12k可作为部署MuJoCo的优先候选，L4作为恢复覆盖对照；当前证据不足以推荐直接自由起身上真机。现有 RoboMimic_Deploy 分支支持93D但使用ONNX：需嵌入观测归一化导出，验证动作数值和控制/观测合同，再测试中间坐姿与深蹲。此任务没有替换部署模型或执行真机动作。
