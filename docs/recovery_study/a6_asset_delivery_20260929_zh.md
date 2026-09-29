# 历史 A6 交接资产：下载、安装、校验

本页补齐原交接说明中“必须单独交付的数据”。资产维护在 `codex/recovery-study`，训练实现仍固定为 `1b6d7e61ddbcd2caf9ea17b02160dd1cf0d119d2`。不要把当前分支的训练代码复制到历史版本。

## 1. 现在可以直接取得的文件

- [训练与历史验证 bank 压缩包](assets/a6_reset_v1/banks.tar.gz)（2,360,039 bytes，约 2.36 MB）。[直接下载](https://raw.githubusercontent.com/lu-yidan/smp/codex/recovery-study/docs/recovery_study/assets/a6_reset_v1/banks.tar.gz)。
- [逐文件 SHA256、schema、机器人哈希](assets/a6_reset_v1/manifest.json)，[压缩包校验值](assets/a6_reset_v1/banks.tar.gz.sha256)。
- [安装/校验工具](../../scripts/recovery_study/a6_assets.py)，[实际安装后的数据与模型检查结果](assets/a6_reset_v1/audit.json)。
- 机器人 `src/smp/assets/deploy_g1/` **已在上述历史 Git 提交中**：`source.xml`、`control.yaml`、`manifest.json` 和 64 个 mesh，共 67 个文件。本次逐一核验，与冻结提交相同，不需要向同学再索取另一个模型。

压缩包 SHA256：

```text
77e15f57c17eb322f09feeed3ff02be1978a80c330f129f9d6c1dc5e13fdb78f
```

只有 banks 原先被 Git 的 `datasets/`、`outputs/` 忽略。模型并非缺失；原说明笼统写“另需交付 XML/mesh”容易造成误解，本次已纠正。

## 2. 推荐使用方式：两个目录，避免切换版本后丢失工具

```bash
git clone --branch codex/recovery-study https://github.com/lu-yidan/smp.git smp-a6-handoff
git -C smp-a6-handoff worktree add --detach ../smp-a6-baseline 1b6d7e61ddbcd2caf9ea17b02160dd1cf0d119d2
python3 smp-a6-handoff/scripts/recovery_study/a6_assets.py install --root smp-a6-baseline
python3 smp-a6-handoff/scripts/recovery_study/a6_assets.py verify --root smp-a6-baseline
```

`install` 会先检查压缩包和每个成员的哈希、目标路径及已有文件，再写入历史路径。已存在且一致的文件保留；遇到不同文件立即报错，**不覆盖、不替换、不更改训练代码**。重复运行可用。`verify` 同时检查 9 个 bank/清单文件和 67 个模型文件。基础安装和哈希校验只依赖 Python 标准库，不需要 CUDA 或训练环境。

已经有历史 checkout 的同学只需把 `--root` 改为该目录；如果当前 HEAD 不等于冻结 A6 提交，默认拒绝。`--allow-other-code` 仅用于外部实现的资产检查，不代表其代码已与 A6 对齐。

有 NumPy 后可以检查形状、方向分布与片段组；有 MuJoCo 后还可编译机器人并检查关节地址：

```bash
python3 smp-a6-handoff/scripts/recovery_study/a6_assets.py verify --root smp-a6-baseline --audit --mujoco --report a6-assets-check.json
```

本次 CPU 模型检查使用 MuJoCo 3.8.1；这不是 GPU PPO 训练通过、动力学完全一致或外部方法已经接入的证明。本包不安装训练环境，不含 SMP 网络、SMP reference checkpoint 或任何 actor/discriminator。HoST/AMP/FIRM 使用自己的可续训系统；复现 SMP 本身还需原交接文档规定的额外策略/先验资产。

## 3. bank 清单

每类目录还附有原始 `manifest.json`。下列验证集用于历史复现/开发验证，不是新论文独立最终测试集。

| 类型 | 安装路径 | 样本数 | SHA256 |
|---|---|---:|---|
| 多地形训练 | `outputs/multiterrain_bank/train.npz` | 4096 | `287eae8e8840c1b3281e7010a84182b824fefacd34439c4f993e9f130af1027a` |
| 多地形历史验证 | `outputs/multiterrain_bank/validation.npz` | 1024 | `d0c4755474df9626a20167843eb453d00e76420771d29f34cd35d5a3a8d29b45` |
| 自然课程训练 | `datasets/reset_banks/natural_curriculum_v1/train.npz` | 3046 | `e9f94540520d7927dee01150a0f0bae39ad2aad5ae008b1c7c71f521650e44b9` |
| 自然课程历史验证 | `datasets/reset_banks/natural_curriculum_v1/validation.npz` | 804 | `7c0f10a217f05673cf987e7094d6be449228f5546756f11d1424f428bff0dfaf` |
| 程序化低位训练 | `datasets/reset_banks/procedural_low_v1/train.npz` | 4096 | `e289bed8d1c93e87fbdcf969fc5fc741f2d8500a0908ed145d2a4e02e7fe116d` |
| 程序化低位历史验证 | `datasets/reset_banks/procedural_low_v1/validation.npz` | 512 | `eb0d2d2ab1131386c21f70d10dc5be36444728bb0581c452fdc3bb3551bb6d31` |

这些是归档原文件，未重采样、裁切或重新生成，三份训练文件与历史 A6 `launch.json` 的 SHA 完全一致。历史 `test.npz` 本次不分发，避免在联调中进一步使用已有测试数据；新的最终评测仍需预先登记场景、姿态组与成功判据。

## 4. 外部方法读取时最容易出错的接口

所有 `qpos` 都是 `[N,36]`：世界坐标 xyz（米）、根姿态 quaternion **wxyz**、29 个关节角（弧度）。不是 93D observation，也不含速度。应按 `source.xml`/`control.yaml` 的关节名映射到自己的模型，不能凭数组长度直接赋值。29 个关节的顺序与本次编译得到的 qpos 地址 7..35 一致，名称列表见 audit.json。速度归零、history 填充方式按主交接协议处理。

- 自然课程 bank：`labels/stages/clips/origins/source_indices`；stages 为 low/middle/late。原始文件频率不均衡，**文件占比不等于训练采样比例**。不要直接均匀抽所有行来代替 A6 的分层 reset。
- 程序化 bank：`labels/stages`，四方向均匀，每条都为 low。
- 多地形 bank：`direction` 为 0 仰卧、1 伏卧、2 左侧、3 右侧；`source` 为 0 自然、1 程序化；`stratum` 为 0 平地、1 导向板、2 自由板、3–7 台阶/斜坡相关旧场景。
- **A6 只使用 stratum 0、1、2**。原文件仍保留 3–7，不能把这些行直接用于 A6 reset，也不要删行后继续标原始 SHA。保留原文件，通过冻结 sampler 选择对应行。
- 多地形文件内 source 原始占比是 50/50，A6 运行时低位 source 配额是约 75/25；场景 50/25/25、平地 L/M/H 40/40/20 也由 sampler 分配。不能把文件频数当成训练配额。
- 原 `source.xml` 还包含部署场景对象，直接加载的 nq 并非 36。训练按 `master_deployment_contract.deployment_robot_spec()` 提取机器人，再创建训练场景。CPU 检查已验证提取后的 `nq=36, nv=35`；不要向完整场景的 qpos 盲目写入 36 列。

## 5. 生成来源与独立性边界

本包交付的是经归档 SHA 确认的文件，不宣称重新运行旧生成器一定逐字节生成相同 zip。

| bank | 可追踪的生成实现与规则 |
|---|---|
| 自然课程 | [`54b3e5c` 的 build_natural_curriculum_bank.py](https://github.com/lu-yidan/smp/blob/54b3e5c2db8e529cbaae4ec9c741d8422804be09/scripts/build_natural_curriculum_bank.py)。源切片 SHA 为 `9c056f85dfe3541801bc86990e7228e956a30d06af0ea07d8e9d9010ad0220e8`；按原片段组哈希切分，镜像同组，曾为伏卧覆盖移动完整组；原 manifest 标有 validation_used_for_curriculum=true。 |
| 程序化低位 | [`1464628` 的 build_procedural_low_bank.py](https://github.com/lu-yidan/smp/blob/1464628f959346105ae40685a90bcd292de45538/scripts/build_procedural_low_bank.py)。四姿态加扰动、部署模型接触检查；train/validation seed 为 2026091301/2026091302，属于同分布独立随机扰动，不自动等于 OOD。 |
| 多地形 | [`c694c78` 的 build_multiterrain_bank.py](https://github.com/lu-yidan/smp/blob/c694c78c85afa1fc8e030ca3e40f76df5a7353eb/scripts/build_multiterrain_bank.py) 与同提交 `merge_multiterrain_bank.py`、`multiterrain_geometry.py`。原清单保存 8 个分片 SHA、尝试次数及接触筛选数据，原 seed 为 202609171/202609172。旧 8 类地形的生成几何不同于后来的 A6 三场景几何，不能直接在 A6 提交下生成全部旧分片并期望得到原文件。 |

检查得到自然 train/validation 去除 `__mirror` 后片段组交集为 0。**多地形 bank 没保留每一行的原始 clip ID / source row**，仅有 source 类别和分片清单，不能在现有文件上重新认证完整端到端独立性。历史验证集已经参与模型选择/课程开发，因此仍只能用于联调和历史验证。未来独立最终测试应重新登记并保留逐行来源。

## 6. 交付验收

本次在干净的冻结 A6 checkout 中实际执行过安装、重复安装、哈希/schema/四元数检查、自然片段组隔离检查和 CPU 模型编译/关节顺序检查；另检查了目标文件不一致时拒绝安装且不覆盖。下一步仍是各同学自己的模型/obs/动作映射、reset 截图和短程训练预检，不能因资产校验通过就直接认为算法移植完成。
