#!/bin/bash
#SBATCH --job-name=broader_architecture
#SBATCH --output=/datastore/uittogether/LuuTru/Dungdm/Research/LowResolution/MFSR/Logs/out_%j.out
#SBATCH --error=/datastore/uittogether/LuuTru/Dungdm/Research/LowResolution/MFSR/Logs/out_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --gres=mps:a100:2
#SBATCH --time=72:00:00

REQUIRED_VRAM=60000

# =========================================================
# ENVIRONMENT
# =========================================================

unset LD_LIBRARY_PATH

module clear -f
module load slurm/slurm/24.11

CONDA_BASE="/datastore/uittogether/tools/miniconda3"
ENV_PATH="$CONDA_BASE/envs/mfsr"
TORCH_LIB="$ENV_PATH/lib/python3.10/site-packages/torch/lib"

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export LD_LIBRARY_PATH="$TORCH_LIB"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$ENV_PATH/lib"

export PATH="$ENV_PATH/bin:$PATH"


# =========================================================
# SELECT GPU - UIT gpu_check.sh
# =========================================================

unset CUDA_VISIBLE_DEVICES

CHECK_OUT=$(/usr/local/bin/gpu_check.sh $REQUIRED_VRAM $SLURM_JOB_ID)
EXIT_CODE=$?

if [ $EXIT_CODE -eq 10 ]; then
    echo "$CHECK_OUT"
    exit 0
elif [ $EXIT_CODE -eq 11 ]; then
    echo "$CHECK_OUT"
    exit 1
fi

BEST_GPU=$CHECK_OUT

echo "================================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Selected GPU : $BEST_GPU"
echo "================================================"


# =========================================================
# PRIVATE MPS
# =========================================================

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID

rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

mkdir -p "$CUDA_MPS_PIPE_DIRECTORY"
mkdir -p "$CUDA_MPS_LOG_DIRECTORY"

export CUDA_VISIBLE_DEVICES=$BEST_GPU


# =========================================================
# DEBUG GPU MAPPING
# =========================================================

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY"

echo "===== NVIDIA-SMI ====="
nvidia-smi

echo "===== PYTORCH CUDA TEST ====="

python - <<'PY'
import os
import torch

print("PID                  :", os.getpid())
print("CUDA_VISIBLE_DEVICES :", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("CUDA available       :", torch.cuda.is_available())
print("CUDA device count    :", torch.cuda.device_count())

if torch.cuda.is_available():
    print("torch current device :", torch.cuda.current_device())
    print("torch device name    :", torch.cuda.get_device_name(0))

    free, total = torch.cuda.mem_get_info(0)
    print("Free VRAM            : %.2f GiB" % (free / 1024**3))
    print("Total VRAM           : %.2f GiB" % (total / 1024**3))

    # allocate something so PID appears in nvidia-smi
    x = torch.zeros((1024, 1024), device="cuda")
    print("Tensor device        :", x.device)
PY


# =========================================================
# TRAIN
# =========================================================

# python -u crop_merge_data.py --workers 8
# python -u resize.py --size 112 --workers 8
# python -u resize.py --size 56 --workers 8

SCALE="${1:-2}"
if [[ "$SCALE" != "2" && "$SCALE" != "4" ]]; then
    echo "Usage: sbatch run.sh {2|4}"
    exit 2
fi

python -u train_mfsr_histology.py \
    --scale "$SCALE" \
    --epochs 20 \
    --batch-size 64 \
    --grad-accumulation 1 \
    --lr 1e-4 \
    --gradient-clip 1.0 \
    --amp-dtype bfloat16 \
    --workers 8

# =========================================================
# CLEANUP
# =========================================================

rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
