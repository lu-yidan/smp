# Recovery 项目统一入口

**当前开发：`/home/luyd/workspace/smp-a6-egress`，分支 `codex/recovery-study`。**

**论文：`/home/luyd/workspace/G1_Recovery_Below_Block`，分支 `codex/paper-recovery-scope-audit`。**

工作区目录保留原名，避免破坏视频、checkpoint、论文证据中的绝对路径。以后新场景与实验导航在当前开发分支集中维护；历史工作树不再作为默认入口，不删除、不搬走、不把它们当前checkout当历史run快照。没有新增每场景分支。

## 最常用的操作

```bash
cd /home/luyd/workspace/smp-a6-egress
./recovery list
./recovery find V33PATH-A6
./recovery find FT_R2
MUJOCO_GL=egl ./recovery build table --direction prone --out outputs/my_table --render
MUJOCO_GL=egl ./recovery build pyramid_stairs --site edge --direction right_side_down --out outputs/my_stair --render
```

输出目录必须不存在，防止覆盖证据。`--seed`固定箱体布局和reset采样；`--overrides file.json`显式设置该场景已支持的尺寸、质量、台阶高度等。`SMP_PYTHON`可指定Python环境，默认复用当前或邻近smp的venv。场景build本身不启动训练；正式混合训练入口见下方运行记录。

场景预览：[10类几何初态](assets/scene_overview.png)。已验证导出文件保存在 `outputs/recovery_study/scenes_v1/<scene>/`。

## 去哪里找什么

|要找的内容|唯一入口|
|---|---|
|实验ID、所属系列、actor父代、原始配置/模型来源|[实验注册表](../../configs/recovery_study/experiments.json)，`./recovery find 关键词`|
|场景参数、运动机制、位置采样点|[场景注册表](../../configs/recovery_study/scenes.json)|
|场景构建、角色标记、初态接触校验|[场景模块](../../src/smp/recovery/scenes.py)|
|64项几何/自由度审计|[检查脚本](../../scripts/recovery_study/check_scenes.py)，输出`outputs/recovery_study/scene_checks.json`|
|历史各工作树/分支位置|[worktree快照](worktree_snapshot.txt)、[分支快照](branch_snapshot.txt)|
|本文主线和待做实验|[论文研究路线](/home/luyd/workspace/G1_Recovery_Below_Block/docs/research_scope_20260923_zh.md)|
|三阶段reward总表及细表|[LaTeX总表](/home/luyd/workspace/G1_Recovery_Below_Block/sections/reward_stage_table.tex)、[公式细表](/home/luyd/workspace/G1_Recovery_Below_Block/sections/appendix_implementation.tex)|
|最近ES结果|[最终分析](../es_final_review_20260923_zh.md)|
|M系列训练完成与新杂物最终评估|[2026-09-24最终评估与视频](clutter_final_review_20260924_zh.md)|
|上方杂物R2/A6/M2开发集结果|[首轮评测与视频](clutter_eval_v1_zh.md)|
|上方杂物冻结场景与评测初态（新）|[v2 使用与验收](clutter_benchmark_v2_zh.md)，`./recovery clutter-list`|
|上方木条、梯架、空箱的评估与道具候选|[场景与采购计划](overhead_clutter_evaluation_plan_20260923_zh.md)（采购设计稿；场景已实现）|

## 不再使用裸标签定位策略

主链：`L4-9999 → FT_R2-9000 → V33PATH-A6-9999`。SMP奖励归一化参考已独立导出，FT12K只是其历史来源，不是actor链中的一段；旧脚本的文件依赖仍保留。

历史scratch R2与FT_R2不同；旧from-scratch A6与V33Path A6不同。训练分支、run、checkpoint更新数、checkpoint SHA共同标识模型。注册表记录的是原实验分支；工作树现在可能处于后续分支，不能据当前脚本重跑一个历史run并称“精确复现”。

历史文档按内容去重索引见 `configs/recovery_study/documents.json`；`./recovery find v36`、`./recovery find scratch` 同时检索实验登记与文档标题/文件名。索引是快照，可用 `python scripts/recovery_study/index_documents.py` 重建。

## 下一轮混合训练设计

[M系列运行记录](mixed_training_20260923_zh.md) 是当前训练入口，包含4096环境预检、冻结基线、场景/reset份额、随机化和run路径。原[设计稿](mixed_scene_plan_20260923_zh.md)保留为决策历史。

## 新场景已实现的范围

|场景ID|物理实现|默认参数/位置|
|---|---|---|
|flat|部署模型地面|四方向低位|
|fixed_ceiling|固定顶板，零自由度|1.2×0.9×0.06m，底面0.60m|
|table|固定桌面＋四腿|1.6×1.2×0.06m，净高0.70m，腿宽0.07m|
|guided_plate|仅竖直滑动|0.9×0.64×0.07m，6kg，相对初态向下0.30m/向上0.60m|
|free_plate|六自由度薄板|0.9×0.64×0.07m，6kg|
|pyramid_stairs|四层方形台阶|级高0.10m、踏面0.30m、顶平台0.55m；center/tread/edge/corner|
|box_terrain|不同长宽高/yaw的固定箱体|可复现seed布局；center/edge/corner|
|free_box|两box组成L形自由刚体|包络0.8×0.6×0.18m，总质量8kg；几何自动计算惯量|
|double_plate|两块独立自由板叠置|每块0.9×0.64×0.05m、4kg，小水平错位|
|mixed_clutter|固定箱体支撑＋自由板|5kg板，center/edge|

这是新版本基准fixture，不是历史A6或V3.6场景的逐项复现；台阶借鉴V3.5/V3.6的金字塔结构，层数等参数显式登记。无限平面保留完整地面，不把有限地形网格外自由落体作为策略失败。将来的局部地形成功区需要独立定义。

每次build导出`scene.xml`、`reset.npz`、`manifest.json`，加`--render`生成`preview.png`。MuJoCo初始keyframe名`reset`；加载XML后需`mj_resetDataKeyframe(model,data,0)`再`mj_forward`，不能用默认站姿冒充筛选初态。XML mesh路径是本机绝对路径；换机请重新build，勿只拷XML。

manifest保存输入配置、随机种子、reset bank索引/SHA、机器人XML SHA、模型XML SHA、物体角色与初始接触检查。free body的惯量由实际碰撞几何与总质量计算；固定桌体质量不作为可推动质量随机化。角色support/overhead/barrier分开，支撑箱不参与上方投影脱困判断。

## 验证边界

64项场景/位置/方向组合通过：初始穿透不低于−2mm、质量与惯量为正、固定/竖直滑动/自由刚体自由度正确、角色分离、20ms被动仿真有限。10个展示场景的XML/keyframe重载也检查了姿态和初始接触。

这不等于恢复成功、长时间稳定reset、出口可达或真机安全。reset先几何贴合有效支撑，并未执行policy或settling训练。双板是叠置实例，不代表任意堆叠。以上是旧CPU fixture的验收边界。新GPU统一拓扑、逐物体reward、93D推理与PPO已另接入 `mixed_geometry.py` / `mixed_task.py`，详情见M系列运行记录，不把新旧fixture的参数混写。

板下奖励不能直接套到支撑箱体：正常踩箱不应被算作受困。将来的统一评测必须先明确台阶上恢复/允许撤离、物体角色、真正阻挡与全局完成条件。

## 数据与复现

本次初态库复用服务器`/root/workplace/smp-a6-egress/outputs/ceiling_bank/validation.npz`，本地`datasets/recovery_benchmark/validation.npz`，只用于开发和可视化，不能据此声称独立测试。若缺失：

```bash
mkdir -p datasets/recovery_benchmark
rsync -az -e 'ssh -p 17535' root@192.168.11.91:/root/workplace/smp-a6-egress/outputs/ceiling_bank/validation.npz datasets/recovery_benchmark/validation.npz
PYTHONPATH=src /home/luyd/workspace/smp/.venv/bin/python scripts/recovery_study/check_scenes.py
```

## 文档维护规则

新实验先登记限定ID、父checkpoint、代码commit、run路径、场景配置SHA与证据状态，再启动。原始配置/launch是事实源；中文说明是解释；论文只导入可追溯证据。历史文档不批量改名或重写；新索引注明其系列和地位。新增结果优先记入对应系列页面，不再另起同名R/B/F/A文件。
