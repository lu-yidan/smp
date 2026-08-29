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
- 4096 environments, effective policy and environment seed 42, 30,000 updates;
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

The historical run-name suffix `seed20260830` is only a label. Inspection of
every saved `params/agent.yaml` and `params/env.yaml`, together with the live
training logs, proves that both effective seeds are 42. Frozen manifests read
and validate these saved parameters instead of trusting a run name. The screen
therefore remains one valid common-seed comparison, but `20260830` must not be
reported as its policy seed.

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

Final flat-policy promotion is deliberately stricter than an early screen.
An arm is eligible only if it passes the rapid thresholds at 15k, 25k, and
29999, and its final checkpoint has native-GSI success >= 95%, fixed-pose
macro success >= 80%, worst fixed-pose success >= 60%, and finite safety
metrics. Neither fixed-pose macro nor worst-pose success may fall by more than
10 percentage points from 25k to 29999. If no arm qualifies, the frozen
decision is `NO_PROMOTION`; thresholds are not relaxed after inspecting the
matrix.

At most two eligible arms advance to independent policy-seed replication.
Selection is lexicographic: highest lower Wilson bound for worst-pose success,
then highest lower Wilson bound for fixed-pose macro success, then fewer
secondary falls, lower post-success root drift, lower contact-conditioned foot
slip, lower peak power, and lower peak joint speed. Differences inside
overlapping rollout intervals are reported as unresolved rather than used to
claim a winner. This single-seed choice is only a resource-allocation rule;
the three-seed aggregate remains the inferential unit for the paper.

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

The recurring job advances the complete evidence pipeline with one idempotent
controller:

```bash
uv run scripts/advance_smp_ral_pipeline.py \
  --control-dir run_control/scratch_causal_30k_seed20260830 \
  --evidence-dir run_control/scratch_causal_eval \
  --state run_control/automation_state/ral_pipeline_latest.json \
  --launch-when-ready
```

While any training job is active, the controller only updates health and
manifest state. It never launches evaluation against busy training GPUs. A
manifest is created only after all eight checkpoints for that gate exist, and
every recorded checkpoint hash is revalidated on later invocations. The 8k
manifest is additionally protected by its frozen manifest hash. After all
eight 30k jobs finish and the GPU process table is empty, the controller
launches the earliest incomplete gate as a detached, resumable eight-GPU
matrix followed by the analyzer. It records `TRAINING_ACTIVE`,
`WAITING_FREE_GPU`, `EVAL_RUNNING`, or `ANALYSIS_COMPLETE` atomically. A gate
counts as complete only when its schema-v2 completion marker, protocol fields,
40 expected cases, summary, and admissible analysis decision all validate.
Interrupted work therefore resumes missing cases instead of overwriting valid
evidence or being mistaken for a finished experiment.

After all four gate analyses are complete, the same controller invokes the
checkpoint-stability selector:

```bash
uv run scripts/select_smp_stable_arm.py \
  --evidence-dir run_control/scratch_causal_eval
```

The selector verifies the gate checkpoint name, common arm set, common policy
seed, and SHA-256 of every source analysis. It enforces the frozen 15k/25k/final
thresholds and late-regression rule, writes `stable_selection.json` and
`stable_selection.md` atomically, and promotes at most two configurations for
independent policy-seed training. Overlapping rollout intervals are explicitly
reported as unresolved rather than converted into a winner claim.

For each promoted configuration, confirmatory policy seeds are frozen to
`20260901`, `20260902`, and `20260903`. Both `--agent.seed` and `--env.seed`
must be set to the same declared value; changing only the environment seed does
not create an independent policy initialization. The idempotent launcher
records the stable-selection hash, code commit, full commands, GPU assignment,
PIDs, logs, and transition budget before reporting a launch as complete:

```bash
uv run scripts/launch_smp_policy_seed_confirmation.py \
  --selection run_control/scratch_causal_eval/stable_selection.json \
  --control-dir run_control/scratch_causal_policy_seed_confirmation \
  --launch
```

The recurring controller passes `--launch-confirmation-when-ready` and invokes
this launcher only after every frozen screen gate is complete and no GPU
process remains. One promoted arm creates three jobs; two unresolved promoted
arms create six jobs. A changed source analysis, unknown arm, reused seed,
insufficient GPU set, active GPU process, or conflicting prior launch plan
causes a hard refusal rather than an altered experiment.

When all confirmation jobs reach update 29999, the controller verifies their
saved agent/environment seeds and checkpoint hashes, then creates one immutable
manifest per policy seed. Each seed receives the same five-mode, 512-environment
frozen evaluation in its own output directory. Keeping each summary to one
trained seed prevents the rollout matrix from being mistaken for independent
training replication. After all three matrices pass schema and completion
checks, `aggregate_smp_policy_seeds.py` bootstraps across the three policy-level
summaries and writes `policy_seed_aggregate.json`; only
`MINIMUM_POLICY_SEEDS_MET` can advance to terrain and plate experiments.

Safety is reported but is not optimized in this first causal screen. The
winning recovery configuration must later receive a separate speed/power
ablation before real-robot use.

## Audited A6 continuation after a numerical failure

On 2026-08-30, `a6_f2s2_mix_bridge` stopped after learning iteration 25530
because the environment returned a NaN in the 93-dimensional actor observation.
The immediately preceding log also contained contact-match overflow warnings
and a sharp value-loss increase. The original log is retained at
`run_control/scratch_causal_30k_seed20260830/incidents/gpu6_nan_25530/original.log`;
the other seven arms were not stopped or modified.

The already-written `model_25000.pt` was audited before continuation. It records
iteration 25000, its actor and critic tensors are finite, and its SHA-256 is
`2e2aa6c96acc4676ce6c2515dda6af1d5c2a4277e8197ca57531a4532ff3f63a`.
Only the failed A6 job was continued, with the same task, 93-dimensional actor,
policy/environment seed 42, 4096 environments, model, optimizer, and
normalizers. The continuation is a distinct W&B run, `a6n25530`, and writes only
the final checkpoint so it cannot shadow the frozen 25k artifact. Its runtime
provenance record is version-locked at SHA-256
`35f88b665ee514a714cc676f33b8886f03f8b70d00b105021994402f03094213`.

This is not an uninterrupted replicate: the environment random stream was
reinitialized and the RSL-RL resume boundary repeats one update index. The
continuation is therefore admissible only for this single-seed resource screen.
It cannot provide policy-level uncertainty or final paper evidence. Any arm
selected by the frozen matrix, including A6, must still pass three independent
from-scratch policy seeds. A second NaN or fatal error in the A6 continuation is
a terminal training alert and must not trigger another automatic restart.

The 25k eight-arm manifest was generated from the pre-failure A6 checkpoint,
independently rehashed, and version-locked at
`1709b96d2a71f0821315cc98a43412a7f71af4181d367110233b59410ea93029`.
The final manifest additionally requires every checkpoint to load with embedded
iteration 29999 and finite actor/critic tensors. No frozen evaluation may start
until the final manifest is complete and its SHA is committed to the versioned
lock table.

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

RA-L readiness is separately tracked in `docs/ral_evidence_matrix.json` and
audited with:

```bash
uv run scripts/audit_smp_ral_readiness.py
```

The audit returns `RAL_READY` only when every required criterion is explicitly
`met` and has valid evidence. Historical 96D simulation policies, qualitative
real-robot logs, an implemented evaluator without results, or an
`in_progress` experiment cannot satisfy a criterion for the current 93D
deployable policy.

Continuing a policy that originally began randomly through a curriculum is
still a from-scratch training pipeline. Loading V3.3, V8, or another external
policy checkpoint is finetuning and is excluded from the baseline evidence.
