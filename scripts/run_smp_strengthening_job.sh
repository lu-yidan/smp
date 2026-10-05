#!/usr/bin/env bash
set -euo pipefail
smp_arm=${1:?off or on required}
smp_gpu=${2:?GPU required}
smp_out=${3:?output directory required}
smp_revision=${4:?source commit required}
case "$smp_arm" in off|on) ;; *) exit 2;; esac
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="$smp_gpu" PYTHONPATH=src:scripts:. MUJOCO_GL=egl
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export NETRC=/root/workplace/amp-runtime/wandb.netrc
export SMP_SOURCE_COMMIT="$smp_revision"
exec .venv/bin/python -u scripts/train_smp_guidance_strengthening.py \
    --arm "$smp_arm" --num-envs 4096 --updates 10000 --seed 20261006 \
    --out "$smp_out" --wandb online
