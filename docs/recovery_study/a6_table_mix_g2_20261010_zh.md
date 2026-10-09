# A6 桌下脱困进展奖励 ×2

用户要求增大脱困奖励后重新训练。基于诊断，仅把 plate_geometry_progress 权重由0.45增至0.90；该函数原有受阻/未脱困门控保留。clearance、separation、completion、Q/L及所有其他奖励不变，不新增脱困后位置奖励，不改变gamma或探索噪声。目的是单独验证更强即时几何进展奖励是否缓解退化，不预设一定成功。

- 初始化：原始 A6_no_alpha@9999，SHA256 24f1a4513cf5cabb9bd0b9ef54e90a3d50e625a87774878379af5da5dcf8fbd1；不使用退化的1000次checkpoint。
- bj `/root/workplace/smp-v34-table-mix`；GPU0/1训练，GPU3/4验证；seed20261009/20261010，与之前两组对应。
- 4096环境，6000更新，LR1e-5固定；actor/critic继承、optimizer新建、actor归一化冻结；f2s2/ws6及原SMP参考保持。
- 平地30%／上下板30%／固定桌40%；固定桌净高32–40cm，沿用相同reset bank。平地一半低位、一半中间态；板/桌仰卧伏卧均分。
- 保留actor观测噪声、push、质量/惯量、Kp/Kd、延迟等随机化。具体以env.yaml为准。
- 每500保存并进行512环境、相同seed的分场景SR3评测；每1000录像。恢复起始时限20秒、连续站立3秒、观察至23秒。不能将几何脱困率当作SR3。
- 原有连续两次三场景SR3全零且迭代至少1000的停止规则保持。比较之前相同seed的原权重结果，同时检查几何脱困、位移和动作负载，不只看训练奖励。

入口：scripts/train_a6_table_mix.py --escape-progress-scale 2
日志：logs/A6Mix_G2_seed20261009、logs/A6Mix_G2_seed20261010。
