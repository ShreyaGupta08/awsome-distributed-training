#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# ============================================================================
# NVRx ft_launcher in-job restart on Slurm. Detects hangs and exception
# faults, then respawns workers within the same Slurm allocation.
#
# Mirrors kubernetes training-job-ft-launcher.yaml.
# ============================================================================

#SBATCH --job-name=nvrx-ft-launcher
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
CKPT_DIR="${CKPT_DIR:-${SHARED_STORAGE}/checkpoints/nvrx-ft-launcher-${SLURM_NNODES}N}"

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-500}"
TRAINING_DURATION="${TRAINING_DURATION:-120}"
MAX_STEPS="${MAX_STEPS:-2000}"
DATASET_NAME="${DATASET_NAME:-allenai/c4}"

# ft_launcher-only mode — matches kubernetes training-job-ft-launcher.yaml
FAULT_COUNT="${FAULT_COUNT:-5}"
FAULT_SEED="${FAULT_SEED:-42}"
FAULT_AFTER_STEP="${FAULT_AFTER_STEP:-50}"
FAULT_TYPES="${FAULT_TYPES:-exception,hang}"
FAULT_WEIGHTS="${FAULT_WEIGHTS:-0.6,0.4}"

FT_MAX_RESTARTS="${FT_MAX_RESTARTS:-20}"
FT_RANK_HEARTBEAT_TIMEOUT="${FT_RANK_HEARTBEAT_TIMEOUT:-120}"
FT_INITIAL_RANK_HEARTBEAT_TIMEOUT="${FT_INITIAL_RANK_HEARTBEAT_TIMEOUT:-180}"
NCCL_TIMEOUT="${NCCL_TIMEOUT:-120}"

CONTAINER_IMAGE="${CONTAINER_IMAGE:-nvcr.io#nvidia/pytorch:25.08-py3}"
CONTAINER_NAME="${CONTAINER_NAME:-nvrx-ft-launcher-${SLURM_NNODES}N}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-${SHARED_STORAGE}:${SHARED_STORAGE}}"

mkdir -p logs "${CKPT_DIR}"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
RDZV_PORT=${RDZV_PORT:-29500}

echo "=== NVRx ft_launcher In-Job Restart ==="
echo "  Nodes:         ${SLURM_NNODES}"
echo "  Total GPUs:    ${TOTAL_GPUS}"
echo "  Model:         ${MODEL_NAME}"
echo "  Checkpoints:   ${CKPT_DIR}"
echo "  Faults:        count=${FAULT_COUNT}, seed=${FAULT_SEED}, types=${FAULT_TYPES}, weights=${FAULT_WEIGHTS}"
echo "  ft_launcher:   max-restarts=${FT_MAX_RESTARTS}, heartbeat=${FT_RANK_HEARTBEAT_TIMEOUT}s"
echo "  Container:     ${CONTAINER_IMAGE}"
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
        export NCCL_NVLS_ENABLE=0
        export TORCH_CPP_LOG_LEVEL=ERROR
        export PYTORCH_ALLOC_CONF=expandable_segments:True
        export HF_TOKEN="${HF_TOKEN:-}"
        export HF_HOME='"${SHARED_STORAGE}"'/.cache/huggingface
        export PG_MASTER_ADDR='"${MASTER_ADDR}"'

        if [ $SLURM_NODEID -eq 0 ]; then
            RDZV_CONF="is_host=1"
        else
            RDZV_CONF="is_host=0"
        fi

        ft_launcher \
            --nnodes='"${SLURM_NNODES}"':'"${SLURM_NNODES}"' \
            --nproc_per_node='"${GPUS_PER_NODE}"' \
            --max-restarts='"${FT_MAX_RESTARTS}"' \
            --rdzv_backend=c10d \
            --rdzv_endpoint='"${MASTER_ADDR}"':'"${RDZV_PORT}"' \
            --rdzv-conf="$RDZV_CONF" \
            --ft-restart-policy=min-healthy \
            --ft-rank-heartbeat-timeout='"${FT_RANK_HEARTBEAT_TIMEOUT}"' \
            --ft-initial-rank-heartbeat-timeout='"${FT_INITIAL_RANK_HEARTBEAT_TIMEOUT}"' \
            --monitor-interval=5 \
            '"${SRC_DIR}"'/train_ft_launcher.py \
                --model_name='"${MODEL_NAME}"' \
                --batch_size='"${BATCH_SIZE}"' \
                --max_seq_length='"${MAX_SEQ_LENGTH}"' \
                --learning_rate='"${LEARNING_RATE}"' \
                --checkpoint_path='"${CKPT_DIR}"' \
                --checkpoint_interval='"${CHECKPOINT_INTERVAL}"' \
                --dataset_name='"${DATASET_NAME}"' \
                --max_steps='"${MAX_STEPS}"' \
                --training_duration_minutes='"${TRAINING_DURATION}"' \
                --inject_faults \
                --fault_count='"${FAULT_COUNT}"' \
                --fault_seed='"${FAULT_SEED}"' \
                --fault_after_step='"${FAULT_AFTER_STEP}"' \
                --fault_types='"${FAULT_TYPES}"' \
                --fault_type_weights='"${FAULT_WEIGHTS}"' \
                --nccl_timeout_seconds='"${NCCL_TIMEOUT}"'
    '
