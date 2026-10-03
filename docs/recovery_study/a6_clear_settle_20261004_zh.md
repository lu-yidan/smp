# A6：α × 脱困后位移修复，两个配对 seed

2026-10-04。用户要求不只重复α扫描，而要同步修复脱困后位移较大。本轮在新服务器运行，不改原dsw上的五seed组件消融。最初拟议的纯α复跑与历史seed重跑方案已在任何训练启动前取消。

## 历史证据和问题

同一冻结验证协议下：AC1(α=.2)@9500的SR10=75.1%，脱困后3秒最大位移P95=1.97m；其最终9999为15.3%、2.51m。AC2(α=.5)@9999为59.8%、1.59m；已部署A6_no_alpha@9999为69.7%、0.53m。9500是事后选点，不能与其他组固定最终点混成公平模型选择。AC5/AC6已有位置成本，并未证明提升成功率，所以不直接照搬旧强约束。

## 固定设计

新seed20261026、20261027，各六组，共12个模型。每个seed六组共用R2@9000 actor/normalizer、新critic/optimizer及初始化。历史seed20261013只用于固定评测，不作为新独立训练seed。

|ID|受阻α|脱困后处理|
|---|---:|---|
|CS_a020_base|.2|原配置|
|CS_a020_settle|.2|关闭向外进展＋分阶段位移成本|
|CS_a050_base|.5|原配置|
|CS_a050_settle|.5|关闭向外进展＋分阶段位移成本|
|CS_a100_base|1|A6_no_alpha配置|
|CS_a100_settle|1|关闭向外进展＋分阶段位移成本|

这是一项3×2因子对照：同α比较位移修复包的作用，相同修复设置比较α的作用；不能由本设计单独归因于关闭进展或位置惩罚中的某一项。基线配置逐字段与历史AC对应配置相等。除声明的三项修改外，动力学、奖励、PPO、reset、随机化和终止均保持相同。

4096环境、24步/update、20k更新、每500更新保存、训练episode10秒。保留推力、actor噪声、部件质量/惯量、电机Kp/Kd、0–10ms延迟等；保留SMP奖励，不因站起或低SMP终止。平地50%、上下板25%、自由板25%；板下低位，平地L/M/H=40/40/20；低位自然/程序化75/25、四方向均衡。

## 位移修复包

沿用A6连续15控制步确认脱困的状态及共享anchor记账，不增加随机采样，不向actor加入位置或板信息。

1. 当前确认脱困时，plate_separation=0，plate_clearance固定为1（权重仍+.08）；G本来就关闭。重新受阻时恢复原脱困奖励。
2. 首次确认脱困时记录base XY锚点。新episode重置；重新受阻时暂停位置成本，锚点保持，不因来回穿越边界反复移动。
3. 位置成本仅作用于障碍episode、已脱困且无异常的状态，平地为0。令d为距锚点的XY距离，g为头高.95–1.15m与直立度.80–.93的smoothstep乘积：

   `cost = clip(max(d-.35, 0)/.50, 0, 2)^2 * g`，权重 `-.05`。

35cm内不罚；低位或大幅前倾时不罚。完全直立时，距离.60/.85/1.10/1.35m对应未加权成本.25/1/2.25/4，最大加权幅值.20。与旧AC5相比，激活更晚、自由范围更宽，同时超过旧封顶位置后仍有梯度。这是待验证的折中，不能预先保证减少位移或保持成功率；它也不能约束到达门控高度之前的全部位移。

## 评价

固定20k终点model_19999为主要结果，10k为中期；额外预先记录9500，以检查旧α=.2在9500附近的崩溃是否再次出现。9500/10000/19999使用2048条每套（平地1024、上下板512、自由板512），标称和upper130两套。5000/15000为192条监测。所有轨迹20秒，SR10定义不变。

同时报告SR1/脱困率、两类板分别SR10、脱困后3秒位移P95及完整窗口数、站稳后位移、Slip、HLS95、力矩/速度/功率P95。位移仅条件于完整窗口，须同时看成功率并保留共同成功子集对比，避免“躺着不动”的模型被当作改进。

不单独选择Ours最好的seed，不把失败seed删除。报告两个新seed的全部结果与训练曲线；与dsw实验分服务器列出，跨GPU非确定性不混为纯奖励效应。验证库仍为复用有限状态，不称独立held-out测试。未启动就登记本轮数量，不能看结果后不断增加seed直到获胜。

## 服务器与文件

- SSH别名 `a6-replica-4090`（连接端口11678），7×RTX4090。
- 工作目录 `/root/workspace/smp-a6-paper`，GPU0–5训练，GPU6串行评测。
- Python3.12虚拟环境，包版本固定为原dsw runtime-pins；mjlab和mujoco_warp复用原wheel。
- SSH用本地公钥登录，无迁移本地私钥。W&B授权文件复制到权限0600的private目录，不纳入git。
- 入口 `scripts/train_a6_clear_settle.py`，奖励 `src/smp/rl/tasks/getup/a6_clear_settle.py`。
- 队列 `scripts/queue_a6_clear_settle.py`，正式目录 `logs/rsl_rl/a6_clear_settle/formal_20261004_20k`。
- W&B group `a6_clear_settle_4090_20261004`，project `tabletennis/smp`。
- 先完成配置差异/成本边界测试、各组4096环境32更新预检及部分reset隔离检查，再启动正式训练。

## 新机预检与历史策略复测

六组各4096环境、32更新预检全部完成；配置隔离、奖励表达式及部分reset检查通过。Python为新机3.12.3，全部锁定包版本与DSW相同，EGL渲染通过。SMP f2s2 prior SHA256为9439d9d9f58940f5472015a57da27c72e2305aafbd29cfd9be44f93da675cc59，两机一致。

同一个历史A6_no_alpha@9999、完全相同初态的2048条标称评测：新4090的压板SR10为65.5%，旧DSW为69.7%；脱困后3秒位移P95分别0.521m、0.530m。跨硬件/运行环境存在结果波动，不能把新旧成绩差直接归因于奖励。新CS配对结果统一在4090上比较，论文历史69.7%保留原始来源，不换成新机结果，也不将这次复测当成新的独立训练seed。对照记录见 evidence/a6_clear_settle_20261004/runtime_checkpoint_comparison.json。

## 正式启动

seed20261026六组已在GPU0–5进入正式PPO训练，seed20261027六组排队；GPU6保留评测。六组实际actor/critic/初始qpos/reset计数核对一致。运行证据见 `evidence/a6_clear_settle_20261004/formal_launch.json`。

- CS_a020_base: https://wandb.ai/tabletennis/smp/runs/71aoro69
- CS_a020_settle: https://wandb.ai/tabletennis/smp/runs/k5aurqze
- CS_a050_base: https://wandb.ai/tabletennis/smp/runs/ikxyi313
- CS_a050_settle: https://wandb.ai/tabletennis/smp/runs/yaoazy8p
- CS_a100_base: https://wandb.ai/tabletennis/smp/runs/1j90ysrn
- CS_a100_settle: https://wandb.ai/tabletennis/smp/runs/ciid0ow3
