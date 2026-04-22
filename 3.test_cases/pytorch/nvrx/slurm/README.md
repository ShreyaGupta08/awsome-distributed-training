# Run NVRx Resiliency Testing on Slurm

This guide walks you through running the NVRx resiliency experiments on a Slurm cluster (e.g. AWS ParallelCluster, SageMaker HyperPod Slurm) with GPU nodes, EFA networking, and shared storage (FSx for Lustre). The training scripts under `../src/` are identical to the EKS variant — only the launcher differs.

## 0. Prerequisites

### 0.1. Slurm Cluster

You need a Slurm cluster with GPU nodes and shared storage. Instructions for creating a cluster can be found in [1.architectures](../../../../1.architectures).

Your cluster must have:
- GPU nodes (g5, p4de, or p5 instances)
- [pyxis](https://github.com/NVIDIA/pyxis) with [enroot](https://github.com/NVIDIA/enroot) for container support
- EFA fabric (for multi-node training on p4de/p5)
- A shared filesystem (FSx for Lustre, FSx for OpenZFS, or NFS) accessible from all compute nodes

### 0.2. Clone the Repository onto Shared Storage

The sbatch scripts resolve `src/` via a relative path, so the repository must be cloned on a filesystem that every compute node can read:

```bash
cd /fsx/$USER          # or wherever your shared FS is mounted
git clone https://github.com/awslabs/awsome-distributed-training.git
cd awsome-distributed-training/3.test_cases/pytorch/nvrx/slurm
```

### 0.3. Create the `logs/` Directory

Slurm's `--output=logs/%x_%j.out` directive requires the directory to exist before `sbatch` is called:

```bash
mkdir -p logs
```

## 1. Configure Slurm Directives

Each sbatch script ships with neutral directives and reads cluster-specific values from environment variables. Set them up-front (or `export` in your shell profile):

| Variable | Purpose | Example |
|---|---|---|
| `SHARED_STORAGE` | Shared filesystem root | `/fsx/$USER` |
| `CONTAINER_IMAGE` | NGC PyTorch container (enroot URI uses `#`) | `nvcr.io#nvidia/pytorch:25.08-py3` |
| `CONTAINER_MOUNTS` | Bind mounts into the container | `/fsx:/fsx` |
| `GPUS_PER_NODE` | GPUs per compute node | `8` (p4de/p5), `1` (g5.8xlarge) |

If your cluster requires an account or a specific partition, add the directives at submit time:

```bash
sbatch --account=<your-account> --partition=<your-partition> --nodes=4 train-async-ckpt.sbatch
```

## 2. Set the HuggingFace Token

The training scripts download models (LLaMA) and datasets (C4) from HuggingFace. You need a [HuggingFace access token](https://huggingface.co/docs/hub/en/security-tokens).

**Do not commit your token to any file.** Export it in the submitting shell only — the sbatch script forwards `$HF_TOKEN` into the container as an env var:

```bash
export HF_TOKEN=<your-token>
```

The EKS equivalent is the `huggingface-token` Kubernetes secret created in [kubernetes/secret-hf-token.yaml](../kubernetes/secret-hf-token.yaml).

## 3. Prepare Dataset (Optional)

For fault-recovery experiments with frequent restarts, pre-download the C4 subset to shared storage so that restarts don't hit the HuggingFace API:

```bash
srun --container-image=$CONTAINER_IMAGE \
     --container-mounts=$SHARED_STORAGE:$SHARED_STORAGE \
     --pty bash -c "\
       pip install --no-input -r ../requirements.txt && \
       python ../prepare_dataset.py \
         --output_path=$SHARED_STORAGE/c4_subset \
         --num_samples=100000"
```

Then add `DATASET_PATH=$SHARED_STORAGE/c4_subset` and pass it through via `--dataset_path` by editing the relevant sbatch script's training args.

## 4. Launch Training Jobs

Each feature has an sbatch script (with NVRx) and a `-baseline.sbatch` companion (without). Scale by passing `--nodes=N`:

```bash
# NVRx async checkpointing on 4 nodes
sbatch --nodes=4 train-async-ckpt.sbatch

# Sync-save baseline for the same feature
sbatch --nodes=4 train-async-ckpt-baseline.sbatch

# NVRx in-process restart on 8 nodes with 10 deterministic faults
FAULT_COUNT=10 sbatch --nodes=8 train-inprocess.sbatch

# ft_launcher in-job restart on 4 nodes
sbatch --nodes=4 train-ft-launcher.sbatch

# Combined ft_launcher + in-process
sbatch --nodes=4 train-ft-launcher-inprocess.sbatch
```

### Available Manifests

| Script | NVRx Feature | Training Script | Fault Injection |
|---|---|---|---|
| `train-async-ckpt.sbatch` | Async checkpointing | `train_async_ckpt.py` | No |
| `train-async-ckpt-baseline.sbatch` | Sync checkpointing (torch.save) | `train_async_ckpt.py` | No |
| `train-inprocess.sbatch` | In-process restart (NVRx Wrapper) | `train_inprocess.py` | Yes (exception) |
| `train-inprocess-baseline.sbatch` | Baseline (Slurm job failure) | `train_inprocess.py --disable_nvrx_wrapper` | Yes (exception + hang) |
| `train-ft-launcher.sbatch` | ft_launcher in-job restart | `train_ft_launcher.py` | Yes (exception + hang) |
| `train-ft-launcher-inprocess.sbatch` | Combined (ft_launcher + in-process) | `train_ft_launcher.py --inprocess` | Yes (exception + sigkill) |
| `train-local-ckpt.sbatch` | NVRx local checkpointing | `train_local_ckpt.py --use_local_checkpoint` | No |
| `train-local-ckpt-baseline.sbatch` | Standard torch.save baseline | `train_local_ckpt.py` | No |

### Parameter Overrides

Every training parameter is environment-variable overridable. Common knobs:

```bash
MAX_STEPS=500   CHECKPOINT_INTERVAL=100 sbatch --nodes=8 train-async-ckpt.sbatch
MODEL_NAME=gpt2 BATCH_SIZE=4            sbatch --nodes=2 train-inprocess.sbatch
FAULT_COUNT=10  FAULT_SEED=7            sbatch --nodes=4 train-ft-launcher.sbatch
```

See the variable list at the top of each sbatch script for the full set.

## 5. Monitor Training

```bash
# Job status
squeue -u $USER

# Stream stdout / stderr
tail -f logs/<job-name>_<job-id>.out
tail -f logs/<job-name>_<job-id>.err

# Check fault injection and recovery events
grep -E "INJECTING FAULT|Recovery overhead|TRAINING SUMMARY" logs/<job-name>_<job-id>.out
```

## 6. Stop Training

```bash
scancel <job-id>
```

## Troubleshooting

- **`logs/%x_%j.out: No such file or directory`** — run `mkdir -p logs` before the first `sbatch`.
- **Container pull fails with enroot** — confirm the image URI uses `#` as the registry separator (e.g. `nvcr.io#nvidia/pytorch:25.08-py3`), not `/`.
- **`pip install` wheels unavailable** — the NGC container requires outbound HTTPS. On air-gapped clusters, prebuild an image with `nvidia_resiliency_ext` baked in and point `CONTAINER_IMAGE` at it.
- **HuggingFace 429 rate-limit during restart storms** — pre-download the dataset (§3) and pass `--dataset_path`.
- **NCCL fails at rendezvous on non-EFA clusters** — unset the EFA defaults: `FI_PROVIDER= FI_EFA_FORK_SAFE= sbatch ...`.
