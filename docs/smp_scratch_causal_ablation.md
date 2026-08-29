# SMP from-scratch causal ablation

## Goal

Identify the smallest change that expands a deployable, velocity-free SMP
policy beyond its native GSI manifold while preserving reliable policy
learning from a random initialization.

This screen uses flat ground only. It deliberately excludes stairs, slopes,
rough terrain, and the plate obstacle so failures can be attributed to the
motion prior, reset coverage, off-manifold termination, or reward sparsity.

## Shared configuration

- actor: one frame, no true base linear velocity (93 dimensions);
- critic: original privileged 10-frame observation (960 dimensions);
- random actor, critic, and observation normalizers;
- original adaptive PPO, learning rate 1e-3, 24 steps per update;
- original task-SMP reward and automatic velocity pushes unless an arm
  explicitly changes the procedural cohort;
- original 5 s episode and success termination;
- 4096 environments, seed 20260830, 30,000 updates;
- checkpoint every 1,000 updates;
- contact capacity raised uniformly to 64 for safe lying-pose simulation.

The pretrained motion prior is part of SMP and is not a policy checkpoint.
Every policy in this experiment is trained from scratch.

## Eight paired arms

| GPU | Task suffix | Prior | Reset | Low-SMP termination | Procedural reward |
| ---: | --- | --- | --- | --- | --- |
| 0 | A0 F2S2 GSI | F2S2 | 100% GSI | original | exact product |
| 1 | A1 V7 GSI | LAFAN V7 | 100% GSI | original | exact product |
| 2 | A2 F2S2 strict | F2S2 | 80% GSI + 20% procedural | original | exact product |
| 3 | A3 V7 strict | LAFAN V7 | 80% GSI + 20% procedural | original | exact product |
| 4 | A4 F2S2 reset-aware | F2S2 | 80/20 | GSI only | exact product |
| 5 | A5 V7 reset-aware | LAFAN V7 | 80/20 | GSI only | exact product |
| 6 | A6 F2S2 bridge | F2S2 | 80/20 | GSI only | 10% floor on procedural only |
| 7 | A7 V7 bridge | LAFAN V7 | 80/20 | GSI only | 10% floor on procedural only |

Formal runs started from commit `28addcd`:

| GPU | W&B run | Run ID |
| ---: | --- | --- |
| 0 | [A0 F2S2 GSI](https://wandb.ai/tabletennis/smp/runs/zqrc8jmf) | `zqrc8jmf` |
| 1 | [A1 V7 GSI](https://wandb.ai/tabletennis/smp/runs/5xcp7tru) | `5xcp7tru` |
| 2 | [A2 F2S2 strict](https://wandb.ai/tabletennis/smp/runs/qpo8vd2a) | `qpo8vd2a` |
| 3 | [A3 V7 strict](https://wandb.ai/tabletennis/smp/runs/a6az0q91) | `a6az0q91` |
| 4 | [A4 F2S2 reset-aware](https://wandb.ai/tabletennis/smp/runs/6ok4oe7g) | `6ok4oe7g` |
| 5 | [A5 V7 reset-aware](https://wandb.ai/tabletennis/smp/runs/adg5qrxg) | `adg5qrxg` |
| 6 | [A6 F2S2 bridge](https://wandb.ai/tabletennis/smp/runs/hk30sstb) | `hk30sstb` |
| 7 | [A7 V7 bridge](https://wandb.ai/tabletennis/smp/runs/32nhutcb) | `32nhutcb` |

The procedural 20% is split uniformly across prone, supine, left-side, and
right-side resets. GSI samples always retain the original exact task-times-SMP
reward. The bridge arms use

`task * (0.10 + 0.90 * SMP)`

only for procedural reset labels. This prevents the bridge from changing the
positive-control GSI objective.

## Causal interpretation

- A0 vs A1 isolates the prior.
- A0 vs A2 and A1 vs A3 isolate the addition of procedural resets.
- A2 vs A4 and A3 vs A5 test whether low-SMP termination kills exploration.
- A4 vs A6 and A5 vs A7 test whether exact SMP gating removes the task
  gradient in off-prior states.

No safety shaping, symmetry loss, terrain, plate, replay buffer, LAFAN
milestone reset, or staged recovery reward is included.

## Checkpoint gates

Evaluate updates 8k, 15k, 25k, and 29999 with
`scripts/evaluate_smp_baseline.py`:

1. native GSI with original automatic pushes;
2. clean fixed prone, supine, left-side, and right-side resets;
3. strict stable-standing success and recovery time;
4. peak joint speed, torque, and power.

Rapid screen gates:

- native GSI success >= 95%;
- fixed-pose macro success >= 40%;
- worst fixed pose >= 20%;
- finite actions and no late numerical collapse.

The frozen matrix is resumable and writes one atomic JSON file per arm, reset
mode, and evaluation seed. A manifest contains the policy-training seed
separately from the rollout evaluation seed, for example:

```json
{
  "experiment": "scratch-causal-8k",
  "commit": "207c956",
  "runs": [
    {
      "name": "a0_f2s2_gsi",
      "task": "Smp-Getup-Scratch-A0-F2S2-GSI-G1",
      "checkpoint": "/absolute/path/to/model_8000.pt",
      "policy_seed": 20260830
    }
  ]
}
```

Run it with:

```bash
uv run scripts/build_smp_causal_manifest.py \
  --checkpoint-step 8000 \
  --output run_control/scratch_causal_8k_manifest.json

uv run scripts/run_smp_frozen_eval_matrix.py \
  --manifest run_control/scratch_causal_8k_manifest.json \
  --output-dir run_control/scratch_causal_eval/8k \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7

uv run scripts/analyze_smp_frozen_matrix.py \
  --summary run_control/scratch_causal_eval/8k/summary.json
```

Existing valid result files are skipped. `summary.json`, `summary.csv`, and
`_COMPLETE.json` are written only after the full matrix succeeds. Each result
includes rollout-level success counts and Wilson 95% intervals plus optional
per-environment outcomes. These rollout intervals quantify evaluation noise;
they do not replace the three independent policy-training seeds required for
RAL evidence. Manifest construction is also all-or-nothing: it refuses to
write a file until the same requested checkpoint exists for all eight arms,
and records each checkpoint SHA-256, training seed, W&B run, task, run
directory, and code commit.

The analyzer applies the frozen rapid-screen thresholds, produces
`analysis.json` and `analysis.md`, and emits either `NO_PROMOTION` or
`SCREEN_PASS_NOT_FINAL`. Passing arms are Pareto-ranked by worst-pose success,
fixed-pose macro success, power, and joint speed. It also reports the planned
paired contrasts for prior, procedural resets, reset-aware termination, and
the reward bridge with conservative rollout intervals. A screen pass never
authorizes a RAL claim or terrain/plate promotion before checkpoint stability
and three independently trained policy seeds are available.

When several devices are supplied, each GPU receives one sequential queue and
the queues run concurrently. No GPU receives two simultaneous evaluator
processes. Existing valid case files are removed from the queue before device
assignment, so an interrupted matrix resumes only missing work. Summary and
completion files are written atomically after every device queue succeeds.
Every case includes evaluation schema version 2, physics/control periods,
actor and critic dimensions, and the complete strict-success definition. The
resume check requires the current schema version, so a result produced before
new metrics or protocol metadata were added is rerun instead of silently mixed
with current evidence.

The base evaluator also records contact-conditioned foot slip, root planar
excursion, post-success root drift, secondary falls, foot separation at first
stable stand, action first differences, and action second differences. This
closes the gap between simulation success and the real-robot failure modes
observed as pelvic sliding, repeated struggling, wide final stance, small
steps, and backward falls. These measures are used for Pareto ranking; they do
not silently change the task reward in the causal screen.

Each heartbeat first creates an atomic execution-health snapshot:

```bash
uv run scripts/monitor_smp_training_health.py \
  --control-dir run_control/scratch_causal_30k_seed20260830 \
  --output run_control/automation_state/training_health_latest.json
```

It alerts on missing or stale logs, fatal exceptions, non-finite parsed
metrics, and throughput below half of the eight-job median. The JSON explicitly
states that reward, SMP score, and termination counts are diagnostics only and
cannot rank policies without the frozen evaluator.

Safety is reported but is not optimized in this first causal screen. The
winning recovery configuration must later receive a separate speed/power
ablation before real-robot use.

## Follow-up

After selecting one or two arms:

1. rerun the winner with three independent policy-training seeds;
2. introduce flat/rough/slope/stair terrain via a from-scratch curriculum;
3. validate the plate on flat ground;
4. combine a small terrain-plus-plate cohort only after both components pass
   independently.

Three-seed confirmation must aggregate at the policy level:

```bash
uv run scripts/aggregate_smp_policy_seeds.py \
  --summaries seed_1/summary.json seed_2/summary.json seed_3/summary.json \
  --output-json policy_seed_aggregate.json
```

The tool first pools repeated evaluation rollouts within each trained policy,
then bootstraps the mean across independent policy seeds. It refuses duplicate
policy-seed identifiers and reports `INSUFFICIENT_POLICY_SEEDS` until at least
three are present. This prevents thousands of parallel environments from
being misreported as thousands of independent training replicates.

Continuing a policy that originally began randomly through a curriculum is
still a from-scratch training pipeline. Loading V3.3, V8, or another external
policy checkpoint is finetuning and is excluded from the baseline evidence.
