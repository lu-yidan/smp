# A6 几何脱困进展 G4/G8

用户要求增加A6的4倍、8倍实验。两组均从原始A6_no_alpha@9999初始化，不从已经适应的G2继续。与现有G2 seed20261009相同seed、reset、奖励体系和优化设置，仅plate_geometry_progress权重不同：G2=0.90，G4=1.80，G8=3.60（相对原始0.45）。completion、clearance、separation及全部安全项不变。保留未脱困/低头高/手部接触/历史最好进展门控，倍增不能解决奖励门控为零的情形。

4090-bj：GPU2训练G4，GPU6训练G8。共用GPU5评测，新增入口使用文件锁串行评测，避免争抢显存。现有A6 G2的GPU0/1训练和GPU3/4评测不受影响。

- 4096环境、6000更新、固定LR1e-5，保存/评测每500，录像每1000。
- 平地30%（一半低位，一半过渡），上下板30%，固定桌40%；桌和板仰卧伏卧均分；沿用32–40cm桌底净高bank及深处/边缘比例。
- f2s2/ws6和A6冻结SMP参考；actor/critic精确继承，观测归一化冻结，新optimizer。
- 保留质量/惯量、Kp/Kd、延迟、push、观测噪声；无SMP或站起提前结束。
- SR3仍为20秒内开始连续站稳3秒，观察23秒；按场景/姿态分别报告。目标不只是增大回报，还需结合录像判断动作剧烈和位移，不能声称更大奖励必定更好。

入口scripts/train_a6_escape_scale_mix.py --escape-progress-scale 4或8。
日志logs/A6Mix_G4_seed20261009、logs/A6Mix_G8_seed20261009。
