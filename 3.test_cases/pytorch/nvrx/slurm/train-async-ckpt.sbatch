#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# ============================================================================
# NVRx async checkpointing on Slurm (LLaMA-3.1-8B + FSDP).
# Non-blocking checkpoint saves that overlap with training compute.
#
# Usage:
#   sbatch --nodes=4  train-async-ckpt.sbatch
#   sbatch --nodes=16 train-async-ckpt.sbatch
#   MAX_STEPS=500 sbatch --nodes=8 train-async-ckpt.sbatch
#
# All training parameters can be overridden via environment variables.
# ============================================================================

#SBATCH --job-name=nvrx-async-ckpt
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

# ---- PATHS ----
SHARED_STORAGE="${SHARED_STORAGE:-/fsx/$USER}"
SRC_DIR="${SRC_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../src" && pwd)}"
CKPT_DIR="${CKPT_DIR:-${SHARED_STORAGE}/checkpoints/nvrx-async-ckpt-${SLURM_NNODES}N}"

# ---- TRAINING PARAMETERS (mirrors env_vars.template defaults) ----
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1000}"
TRAINING_DURATION="${TRAINING_DURATION:-120}"
MAX_STEPS="${MAX_STEPS:-5000}"

# ---- CONTAINER ----
CONTAINER_IMAGE="${CONTAINER_IMAGE:-nvcr.io#nvidia/pytorch:25.08-py3}"
CONTAINER_NAME="${CONTAINER_NAME:-nvrx-async-ckpt-${SLURM_NNODES}N}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-${SHARED_STORAGE}:${SHARED_STORAGE}}"

mkdir -p logs "${CKPT_DIR}"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=${MASTER_PORT:-29500}

echo "=== NVRx Async Checkpointing ==="
echo "  Job ID:        ${SLURM_JOB_ID}"
echo "  Nodes:         ${SLURM_NNODES}"
echo "  GPUs/node:     ${GPUS_PER_NODE}"
echo "  Total GPUs:    ${TOTAL_GPUS}"
echo "  Master:        ${MASTER_ADDR}:${MASTER_PORT}"
echo "  Source:        ${SRC_DIR}"
echo "  Checkpoints:   ${CKPT_DIR}"
echo "  Model:         ${MODEL_NAME}"
echo "  Batch size:    ${BATCH_SIZE} (global: $(( BATCH_SIZE * TOTAL_GPUS )))"
echo "  Seq length:    ${MAX_SEQ_LENGTH}"
echo "  Ckpt interval: ${CHECKPOINT_INTERVAL} steps"
echo "  Max steps:     ${MAX_STEPS}"
echo "  Container:     ${CONTAINER_IMAGE}"
echo ""

# ---- Step 1: Install NVRx + deps in the container on all nodes ----
echo "=== Installing NVRx + Python deps on ${SLURM_NNODES} nodes ==="
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

# ---- Step 2: Run training ----
echo "=== Running async-ckpt training (${SLURM_NNODES} nodes, ${TOTAL_GPUS} GPUs) ==="
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
        # AWS EFA defaults — safe to remove on non-EFA clusters (e.g. InfiniBand).
        export FI_PROVIDER=${FI_PROVIDER:-efa}
        export FI_EFA_FORK_SAFE=${FI_EFA_FORK_SAFE:-1}
        export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-^docker,lo,veth}
        export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
        export PYTORCH_ALLOC_CONF=expandable_segments:True
        export HF_TOKEN="${HF_TOKEN:-}"
        export HF_HOME='"${SHARED_STORAGE}"'/.cache/huggingface

        echo "=== Node $SLURM_NODEID / '"${SLURM_NNODES}"' launching torchrun at $(date) ==="

        torchrun \
            --nnodes='"${SLURM_NNODES}"' \
            --nproc_per_node='"${GPUS_PER_NODE}"' \
            --node_rank=$SLURM_NODEID \
            --master_addr='"${MASTER_ADDR}"' \
            --master_port='"${MASTER_PORT}"' \
            '"${SRC_DIR}"'/train_async_ckpt.py \
                --model_name='"${MODEL_NAME}"' \
                --batch_size='"${BATCH_SIZE}"' \
                --max_seq_length='"${MAX_SEQ_LENGTH}"' \
                --learning_rate='"${LEARNING_RATE}"' \
                --checkpoint_path='"${CKPT_DIR}"' \
                --checkpoint_interval='"${CHECKPOINT_INTERVAL}"' \
                --training_duration_minutes='"${TRAINING_DURATION}"' \
                --max_steps='"${MAX_STEPS}"' \
                --use_async_checkpoint=true
    '
