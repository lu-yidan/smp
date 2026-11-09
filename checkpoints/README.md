# V36 与 V34 balanced4 权重交付

分支：`lyd/v36-v34-balanced4-checkpoints`。

| 策略 | 文件 | Actor / Critic | checkpoint 内记录轮次 |
|---|---|---|---:|
| 原始 V36 | [v36_original/model_102000.pt](v36_original/model_102000.pt) | 96D / 960D | 102000 |
| V34 balanced4 hard19000 | [v34_balanced4_hard19000/model_19000.pt](v34_balanced4_hard19000/model_19000.pt) | 93D / 960D | 19000 |

两者均为29维动作、512/256/128 ELU网络的完整PyTorch checkpoint，保留actor、critic、观测归一化、优化器和原有infos；文件未裁剪或重新保存。SHA256见 [manifest.json](manifest.json)，可运行 `sha256sum -c checkpoints/SHA256SUMS` 校验。

```bash
git clone --branch lyd/v36-v34-balanced4-checkpoints --single-branch git@github.com:lu-yidan/smp.git
cd smp
sha256sum -c checkpoints/SHA256SUMS
```

权重直接存储于Git，无需Git LFS，也无需访问W&B。

## 代码与来源

- 本分支以原始V36代码 `codex/terrain-recovery-v36` 的 `019a9ab6ab47abf0cebf04cd375eb17866f7c9d1` 为基础。
- V36来自W&B `tabletennis/smp/de2kit7e/model_102000.pt`，包含基座线速度输入。这是原始96D模型，不是93D零速度投影，也不是最近台阶续训的1000/3000轮版本。
- V34来自服务器 `gpu4090-bj` 的 `/root/workplace/smp-v34-plate-transfer/logs/V34ER_balanced4_hard20_2ms_continue20k_20261011/model_19000.pt`。同目录原始 `env.yaml` 和 `agent.yaml` 已附带，记录训练配置；其中绝对资源路径需按新机器调整。
- V34的完整训练实现位于该服务器 `/root/workplace/smp-v34-plate-transfer`，入口 `scripts/train_v34_er_continue_2ms.py`；奖励、场景和困难重采样分别见 `scripts/v34_escape_recovery.py`、`scripts/v34_model_only_tables.py`、`scripts/v34_hard_reset.py`。这些运行时增补代码未整合进本权重交付分支。

本分支交付两份起点权重，不表示两者已适配成相同任务。V34为部署模型、2ms物理步/20ms控制步；原V36为历史训练模型、5ms物理步/20ms控制步。开展台阶＋压板对照前应对齐机器人、观测、动作、控制参数和评测协议；不能直接把93D V34权重加载到96D V36 actor。
