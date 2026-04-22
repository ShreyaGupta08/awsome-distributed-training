#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# ============================================================================
# NVRx local checkpointing on Slurm — each rank writes its shard to node-local
# storage instead of shared FSx/Lustre. Avoids shared-FS write contention.
#
# NOTE: tested with GPT-2 on g5 instances only. FSDP sharded state dicts are
# not yet compatible with NVRx BasicTensorAwareStateDict on larger models.
#
# Mirrors kubernetes training-job-local-ckpt.yaml.
# ============================================================================

#SBATCH --job-name=nvrx-local-ckpt
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
TOTAL_GPUS=$(( SLURM_NNODES * GPUS_PER_NODE ))

SHARED_STORAGE="${SHARED_STORAGE:-/fsx/$USER}"
SRC_DIR="${SRC_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../src" && pwd)}"
# Local checkpointing writes to node-local storage (not shared FS).
LOCAL_CKPT_DIR="${LOCAL_CKPT_DIR:-/tmp/nvrx-local-ckpt}"

MODEL_NAME="${MODEL_NAME:-gpt2}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-200}"
TRAINING_DURATION="${TRAINING_DURATION:-30}"
MAX_STEPS="${MAX_STEPS:-1000}"
NCCL_TIMEOUT="${NCCL_TIMEOUT:-60}"

CONTAINER_IMAGE="${CONTAINER_IMAGE:-nvcr.io#nvidia/pytorch:25.08-py3}"
CONTAINER_NAME="${CONTAINER_NAME:-nvrx-local-ckpt-${SLURM_NNODES}N}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-${SHARED_STORAGE}:${SHARED_STORAGE}}"

mkdir -p logs

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=${MASTER_PORT:-29500}

echo "=== NVRx Local Checkpointing (LocalCheckpointManager) ==="
echo "  Nodes:         ${SLURM_NNODES}"
echo "  Model:         ${MODEL_NAME}"
echo "  Local ckpt:    ${LOCAL_CKPT_DIR}"
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
        export PYTORCH_ALLOC_CONF=expandable_segments:True
        export HF_TOKEN="${HF_TOKEN:-}"
        export HF_HOME='"${SHARED_STORAGE}"'/.cache/huggingface

        mkdir -p '"${LOCAL_CKPT_DIR}"'

        torchrun \
            --nnodes='"${SLURM_NNODES}"' \
            --nproc_per_node='"${GPUS_PER_NODE}"' \
            --node_rank=$SLURM_NODEID \
            --master_addr='"${MASTER_ADDR}"' \
            --master_port='"${MASTER_PORT}"' \
            '"${SRC_DIR}"'/train_local_ckpt.py \
                --model_name='"${MODEL_NAME}"' \
                --batch_size='"${BATCH_SIZE}"' \
                --max_seq_length='"${MAX_SEQ_LENGTH}"' \
                --learning_rate='"${LEARNING_RATE}"' \
                --checkpoint_path='"${LOCAL_CKPT_DIR}"' \
                --checkpoint_interval='"${CHECKPOINT_INTERVAL}"' \
                --training_duration_minutes='"${TRAINING_DURATION}"' \
                --max_steps='"${MAX_STEPS}"' \
                --nccl_timeout_seconds='"${NCCL_TIMEOUT}"' \
                --use_local_checkpoint
    '
