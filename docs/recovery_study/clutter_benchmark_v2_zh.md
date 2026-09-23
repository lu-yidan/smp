# 上方杂物评测集合 v2

本目录对应独立 CPU MuJoCo 场景和冻结初态集合，**没有修改现有 M 系列训练**。主代码仍在 `codex/recovery-study`，不为每个道具创建分支。V1 是开发中间产物，未完成的集合不使用；唯一交付入口为 `outputs/recovery_study/overhead_clutter_v2`。

## 场景与分组

|场景ID|三种布局|物理实现|
|---|---|---|
|crossed_timber|两件交叉、三件扇形、四件交错|2–4个独立自由刚体；宽窄搭配，主要覆盖机器人上方|
|ladder_boards|横置、斜置、偏置且叠两板|两侧梁＋四横档组成一个刚体，另有1–2个独立板，保留横档空隙|
|fixed_c_space|开口朝前／左／右|固定顶板＋后侧板＋底座，无自由关节；身体位置按实际后墙避碰，保持原方向|
|hollow_container|开口朝下、侧倾、斜向倾斜|五个壁面组成一个自由刚体，开口和内部空腔保留；当前为刚性塑料箱代理，不包含软篮/衣物|

- 开发集48条：4类×4方向×3布局×1变体。
- 留出集480条：4类×4方向×3布局×10变体；每类120条。
- 四方向分别是supine/prone/left_side_down/right_side_down，使用pelvis重力方向复核，不能用摄像机左右判断。
- 自然/程序化比例75/25。留出集的交叉木条中，70%仅上方物体，30%额外有一块低位支撑；其他场景不额外散放下方方块。C形空间自身底座仍是真实支撑物。
- 源姿态来自既有 `datasets/recovery_benchmark/validation.npz`。按源关节/朝向配置分组拆分开发与留出，忽略全局位置；独立审计也检查四元数q/-q等价。因此是**此新基准内的留出姿态/几何参数组合**，不是对历史先验或所有训练从未见过的动作数据。
- 同几何布局的各方向使用同一geometry seed；源姿态、摆放小扰动和碰撞筛选独立。尺寸、质量、摩擦、物体数、机器人实际位置均在逐case manifest中。几何为参数化代理，尚未由购买实物标定。

固定C空间需区分顶板底面绝对高度和净空：底座厚4.5cm，manifest里的 `clear_height_m` 才是底座以上净空。留出顶板底面约0.55–0.80m，因此净空约0.505–0.755m。四方向不保证躯干处于完全相同的覆盖深度，具体qpos已冻结，必须分布局报告。

## 数据入口

本地：`/home/luyd/workspace/smp-a6-egress/outputs/recovery_study/overhead_clutter_v2/`。

服务器整包：`/root/workplace/smp-recovery-study/outputs/overhead_clutter_v2/`（SSH端口17535）。已核对两份index SHA相同，并在服务器MuJoCo 3.8.1分别重载四类伏卧案例。

[四类总览](evidence/clutter_v2/overview.png) · [四方向图](evidence/clutter_v2/four_directions.png) · [完整验收报告](evidence/clutter_v2/audit.json)。

每条case包含：

- `scene.xml`：部署G1几何＋场景，`reset` keyframe。
- `reset.npz`：完整qpos/qvel；初始速度为零。
- `manifest.json`：源姿态索引与SHA、尺寸/质量/摩擦、角色、摆放种子、模型SHA、开口检查、碰撞及短时物理检查。
- 开发集另含 `preview.png`，为真实MuJoCo初态渲染，不是策略结果。

每个split有`index.json`，根目录有`assets/meshes`及冻结robot XML/control。XML mesh路径相对这个包，复制整包即可使用；单独复制一个case不带assets无法加载。MuJoCo版本记录在index及验收报告。

## 本地看场景

```bash
cd /home/luyd/workspace/smp-a6-egress
./recovery clutter-list
./recovery clutter-play --case outputs/recovery_study/overhead_clutter_v2/development/ladder_boards__prone__l1__v00
```

默认仅看冻结初态，可旋转相机；**不运行policy**。加 `--passive` 才进行零控制物理演示，不能把该行为解读为恢复策略。无图形界面时可直接渲染：

```bash
MUJOCO_GL=egl ./recovery clutter-play \
  --case outputs/recovery_study/overhead_clutter_v2/development/ladder_boards__prone__l1__v00 \
  --render /tmp/ladder_preview.png
```

## 重建与验收

```bash
cd /home/luyd/workspace/smp-a6-egress
export PYTHONPATH="$PWD/src"
CLUTTER_PY=/home/luyd/workspace/smp/.venv/bin/python
MUJOCO_GL=egl "$CLUTTER_PY" scripts/recovery_study/build_clutter_benchmark.py \
  --out outputs/new_clutter/development --split development --variants 1 --render
MUJOCO_GL=egl "$CLUTTER_PY" scripts/recovery_study/build_clutter_benchmark.py \
  --out outputs/new_clutter/heldout --split heldout --variants 10
"$CLUTTER_PY" scripts/recovery_study/check_clutter_benchmark.py \
  --root outputs/new_clutter --out outputs/new_clutter/audit.json
```

构建器拒绝覆盖已有split目录。完整528例验收包括：XML/keyframe重载、初始穿透≤2mm、姿态标签、独立自由关节/正惯量/质量、100ms零控制有限性与穿透、空腔/梯格/出口射线、上移阻挡与移走后的反例、故意深穿透可检出、split无重复。XML参数有十进制舍入，误差单独记录，不以逐位一致冒充精确重载。

上移探针以固定关节配置将身体竖直平移，调用真实分件碰撞判断，确认每个标记物体能阻挡这条直接上移路径。它**不是关节可达性或存在安全逃逸路径的证明**，也不是新的在线reward。100ms零控制检查不是长时间稳定性保证；初态摆放之后不宣称所有物体已经压实静止。

## 已完成验收

开发48例、留出480例全部通过；最大初始穿透分别0.716mm和0.916mm，100ms零控制期间最大穿透分别4.871mm和5.765mm。源姿态（包括q/-q等价）和几何种子在两split之间无交集，但相同三类布局模板在两个split中均存在，不称为未见布局类别泛化。

索引与验收证据保存在 `evidence/clutter_v2/`，完整模型包留在outputs，避免把约95MB的重复模型加入Git。索引的 `commit_at_build` 是构建时父提交；本轮未提交的生成器内容另外由 `builder_sha256` 和 `geometry_sha256` 固定，可对照提交后的源码。

## 下一步如何评策略

先在开发集接入R2、A6@9999和固定选择的M checkpoint，核对93D观测、归一化、动作顺序、50Hz及PD/warmup。当前新包不直接兼容原M评测脚本的固定GPU拓扑，不把旧脚本指向新目录就称完成评测。

正式评估沿用20s，分别统计脱困/持续1s和10s站稳、再跌倒、脱困后路程及关节力矩/速度/功率与持续负载。空隙/空腔需要逐碰撞几何及支撑语义，不能用整个箱子的实心AABB作为完成判据。**本轮交付场景、初态和验收，尚无这些新场景的策略成功率。**

若随后根据留出集的策略失败调参，该集合就转为开发集，另建新的最终测试；保持版本和索引，不覆盖原case。拍摄故事、动态砸落、柔性衣物、椅子、机器人搬运控制器不在当前集合范围。
