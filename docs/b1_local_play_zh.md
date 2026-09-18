# B1 8500 本地播放

使用真实 balanced 训练环境与 checkpoint；无真机连接。模型 `outputs/b1_play/B1_8500.pt`，SHA256 `8838a87fafaf43d6cd9a169a86a7c3e89a2897c6936bacd06b8fd03023e8f1c5`，与服务器一致。

```bash
/home/luyd/workspace/smp-p5-balanced/scripts/play_b1.sh --scene flat --pose supine
/home/luyd/workspace/smp-p5-balanced/scripts/play_b1.sh --scene plate --pose prone
```

默认 LAFAN 自然低位验证样本0、0.5倍速、20秒自动reset。`--pose supine/prone/left/right`，`--source natural/procedural`，`--sample 0..15`；所有方向用相同样本序号。`--speed 1`正常速度，`--paused`初始暂停。`--upper-mass 1.3`启用上半身增重压力条件；默认没有新增动力学随机化、push或actor噪声，观察策略确定性表现。

空格暂停/继续，Enter重新reset相同样本，右箭头单步，`-`/`=`调速度。`,`/`.`切换环境：32环境，依次仰卧/伏卧/左/右，每方向8个相同初始样本。默认仅显示一个机器人。

压板为6kg、被动竖直滑动，可被顶起；不能平移或倾斜，不是自由刚体板。仍有接触，未关闭碰撞。平地组压板停放到远处。

换checkpoint：`--checkpoint /绝对路径/model_xxxx.pt`，要求同一balanced系列的保存格式。脚本会读取checkpoint自带SMP参考统计，不依赖服务器FT参考文件。
