#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# ============================================================================
# Baseline for in-process restart — runs the same training with NVRx Wrapper
# DISABLED (`--disable_nvrx_wrapper`). Faults crash the Slurm job.
#
# Mirrors kubernetes training-job-inprocess-baseline.yaml: exception + hang
# faults, relies on Slurm `--requeue`-like behaviour (here: job failure).
# ============================================================================

#SBATCH --job-name=nvrx-inprocess-baseline
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
TOTAL_GPUS=$(( SLURM_NNODES * GPUS_PER_NODE ))

SHARED_STORAGE="${SHARED_STORAGE:-/fsx/$USER}"
SRC_DIR="${SRC_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../src" && pwd)}"
CKPT_DIR="${CKPT_DIR:-${SHARED_STORAGE}/checkpoints/nvrx-inprocess-baseline-${SLURM_NNODES}N}"

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-500}"
TRAINING_DURATION="${TRAINING_DURATION:-120}"
MAX_STEPS="${MAX_STEPS:-2000}"
DATASET_NAME="${DATASET_NAME:-allenai/c4}"

FAULT_COUNT="${FAULT_COUNT:-5}"
FAULT_SEED="${FAULT_SEED:-42}"
FAULT_AFTER_STEP="${FAULT_AFTER_STEP:-50}"
# Baseline includes both exception and hang to exercise the full failure surface.
FAULT_TYPES="${FAULT_TYPES:-exception,hang}"
FAULT_WEIGHTS="${FAULT_WEIGHTS:-0.6,0.4}"

NCCL_TIMEOUT="${NCCL_TIMEOUT:-60}"

CONTAINER_IMAGE="${CONTAINER_IMAGE:-nvcr.io#nvidia/pytorch:25.08-py3}"
CONTAINER_NAME="${CONTAINER_NAME:-nvrx-inprocess-baseline-${SLURM_NNODES}N}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-${SHARED_STORAGE}:${SHARED_STORAGE}}"

mkdir -p logs "${CKPT_DIR}"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=${MASTER_PORT:-29500}

echo "=== In-Process Baseline (NVRx Wrapper disabled) ==="
echo "  Nodes:         ${SLURM_NNODES}"
echo "  Model:         ${MODEL_NAME}"
echo "  Faults:        count=${FAULT_COUNT}, seed=${FAULT_SEED}, types=${FAULT_TYPES}, weights=${FAULT_WEIGHTS}"
echo "  NCCL timeout:  ${NCCL_TIMEOUT}s"
echo ""

srun \
    --container-mounts "${CONTAINER_MOUNTS}" \
    --container-image "${CONTAINER_IMAGE}" \
    --container-name "${CONTAINER_NAME}" \
    --ntasks-per-node=1 \
    --ntasks=${SLURM_NNODES} \
    --kill-on-bad-exit=0 \
    --mpi=none \
    bash -c '
        pip install --no-input -r '"${SRC_DIR}"'/../requirements.txt
        pip install --no-input nvidia_resiliency_ext
    '

srun \
    --container-mounts "${CONTAINER_MOUNTS}" \
    --container-image "${CONTAINER_IMAGE}" \
    --container-name "${CONTAINER_NAME}" \
    --ntasks-per-node=1 \
    --ntasks=${SLURM_NNODES} \
    --kill-on-bad-exit=0 \
    --mpi=none \
    bash -c '
        export PYTHONPATH='"${SRC_DIR}"'
        export PYTHONUNBUFFERED=1
        export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
        export FI_PROVIDER=${FI_PROVIDER:-efa}
        export FI_EFA_FORK_SAFE=${FI_EFA_FORK_SAFE:-1}
        export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-^docker,lo,veth}
        export PYTORCH_ALLOC_CONF=expandable_segments:True
        export HF_TOKEN="${HF_TOKEN:-}"
        export HF_HOME='"${SHARED_STORAGE}"'/.cache/huggingface

        torchrun \
            --nnodes='"${SLURM_NNODES}"' \
            --nproc_per_node='"${GPUS_PER_NODE}"' \
            --node_rank=$SLURM_NODEID \
            --master_addr='"${MASTER_ADDR}"' \
            --master_port='"${MASTER_PORT}"' \
            '"${SRC_DIR}"'/train_inprocess.py \
                --model_name='"${MODEL_NAME}"' \
                --batch_size='"${BATCH_SIZE}"' \
                --max_seq_length='"${MAX_SEQ_LENGTH}"' \
                --learning_rate='"${LEARNING_RATE}"' \
                --checkpoint_path='"${CKPT_DIR}"' \
                --checkpoint_interval='"${CHECKPOINT_INTERVAL}"' \
                --training_duration_minutes='"${TRAINING_DURATION}"' \
                --max_steps='"${MAX_STEPS}"' \
                --dataset_name='"${DATASET_NAME}"' \
                --inject_faults \
                --fault_count='"${FAULT_COUNT}"' \
                --fault_seed='"${FAULT_SEED}"' \
                --fault_after_step='"${FAULT_AFTER_STEP}"' \
                --fault_types='"${FAULT_TYPES}"' \
                --fault_type_weights='"${FAULT_WEIGHTS}"' \
                --nccl_timeout_seconds='"${NCCL_TIMEOUT}"' \
                --disable_nvrx_wrapper
    '
