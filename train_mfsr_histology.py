#!/usr/bin/env python3
"""Train histopathology MFSR and export a WSSS-Tissue ResNet101 backbone.

Run x4 pretraining (DataMerge10x -> DataMerge40x)::

    python train_mfsr_histology.py --scale 4

Run x2 pretraining (DataMerge20x -> DataMerge40x)::

    python train_mfsr_histology.py --scale 2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from MFSR_Histology import HistologyMFSR


DATASET_ROOT = Path(
    "/datastore/uittogether/LuuTru/Dungdm/Research/LowResolution/Dataset"
)
DEFAULT_TARGET_DIR = DATASET_ROOT / "DataMerge40x"
DEFAULT_INPUT_DIRS = {
    4: DATASET_ROOT / "DataMerge10x",
    2: DATASET_ROOT / "DataMerge20x",
}
DEFAULT_CHECKPOINT_DIRS = {
    4: Path("checkpoints/mfsr_10x_to_40x"),
    2: Path("checkpoints/mfsr_20x_to_40x"),
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train RGB MFSR on paired DataMerge resolutions and export the "
            "OPT ResNet101 for WSSS-Tissue Stage 2."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scale", type=int, required=True, choices=(2, 4))
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Defaults to DataMerge10x for scale 4 and DataMerge20x for scale 2.",
    )
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for resumable and exported backbone checkpoints.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Must be at least 2 because training ASPP contains BatchNorm at 1x1.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=1.0,
        help="Clip the unscaled global gradient norm; 0 disables clipping.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--no-amp", action="store_true", help="Disable CUDA mixed precision."
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
        help="A100 supports bfloat16, which is much less prone to overflow.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not initialize the three ResNet101 branches from ImageNet.",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and one paired sample without creating the model.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="Development limit; 0 uses the full training split.",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=0,
        help="Development limit; 0 uses the full validation split.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    input_dir = args.input_dir or DEFAULT_INPUT_DIRS[args.scale]
    checkpoint_dir = args.checkpoint_dir or DEFAULT_CHECKPOINT_DIRS[args.scale]
    return (
        input_dir.expanduser().resolve(),
        args.target_dir.expanduser().resolve(),
        checkpoint_dir.expanduser().resolve(),
    )


def read_patch_size(dataset_dir: Path) -> int:
    info_path = dataset_dir / "dataset_info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing metadata: {info_path}")
    with info_path.open("r", encoding="utf-8") as handle:
        info = json.load(handle)

    image_size = info.get("image_size")
    if (
        isinstance(image_size, list)
        and len(image_size) == 2
        and image_size[0] == image_size[1]
    ):
        return int(image_size[0])
    patch_size = info.get("crop_config", {}).get("patch_size")
    if isinstance(patch_size, int):
        return patch_size
    raise ValueError(f"Cannot determine patch size from {info_path}")


def validate_args(
    args: argparse.Namespace, input_dir: Path, target_dir: Path
) -> Tuple[int, int]:
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2")
    if args.workers < 0:
        raise ValueError("--workers cannot be negative")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    if args.grad_accumulation < 1:
        raise ValueError("--grad-accumulation must be positive")
    if not 0 < args.beta1 < 1:
        raise ValueError("--beta1 must be in (0, 1)")
    for dataset_dir in (input_dir, target_dir):
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset does not exist: {dataset_dir}")
        if not (dataset_dir / "manifest.csv").is_file():
            raise FileNotFoundError(f"Missing manifest: {dataset_dir / 'manifest.csv'}")

    input_size = read_patch_size(input_dir)
    target_size = read_patch_size(target_dir)
    if target_size != input_size * args.scale:
        raise ValueError(
            f"Dataset sizes do not match scale {args.scale}: "
            f"{input_size} -> {target_size}"
        )
    if target_size % 4:
        raise ValueError("Target patch size must be divisible by 4")
    return input_size, target_size


def load_train_val_paths(
    manifest_path: Path,
    train_limit: int,
    val_limit: int,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """Read the large manifest once and return the training/validation paths."""
    split_paths = {"train": [], "val": []}
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"relative_path", "split"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Manifest must contain relative_path and split: {manifest_path}"
            )
        for row in reader:
            split = row["split"]
            relative_path = row["relative_path"]
            if (
                split in split_paths
                and Path(relative_path).suffix.lower() in IMAGE_SUFFIXES
            ):
                split_paths[split].append(relative_path)

    for split, limit, split_seed in (
        ("train", train_limit, seed),
        ("val", val_limit, seed + 1),
    ):
        paths = split_paths[split]
        if not paths:
            raise ValueError(f"No '{split}' images found in {manifest_path}")
        if limit > 0 and len(paths) > limit:
            sampled_paths = random.Random(split_seed).sample(paths, limit)
            sampled_paths.sort()
            split_paths[split] = sampled_paths

    return split_paths["train"], split_paths["val"]


def image_to_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.array(image.convert("RGB"), dtype=np.float32, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).div_(255.0)


def apply_joint_augmentation(
    low_resolution: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    operation = random.randrange(8)
    if operation & 1:
        low_resolution = low_resolution.flip(-1)
        target = target.flip(-1)
    if operation & 2:
        low_resolution = low_resolution.flip(-2)
        target = target.flip(-2)
    rotations = (operation >> 2) & 1
    if rotations:
        low_resolution = torch.rot90(low_resolution, rotations, (-2, -1))
        target = torch.rot90(target, rotations, (-2, -1))
    return low_resolution.contiguous(), target.contiguous()


class PairedPatchDataset(Dataset):
    def __init__(
        self,
        input_dir: Path,
        target_dir: Path,
        relative_paths: Sequence[str],
        input_size: int,
        target_size: int,
        augment: bool,
    ) -> None:
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.relative_paths = list(relative_paths)
        self.input_size = input_size
        self.target_size = target_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.relative_paths)

    def __getitem__(self, index: int):
        relative_path = self.relative_paths[index]
        input_path = self.input_dir / relative_path
        target_path = self.target_dir / relative_path
        if not input_path.is_file():
            raise FileNotFoundError(f"Missing LR patch: {input_path}")
        if not target_path.is_file():
            raise FileNotFoundError(f"Missing HR patch: {target_path}")

        low_resolution = image_to_tensor(input_path)
        target = image_to_tensor(target_path)
        if low_resolution.shape[-2:] != (self.input_size, self.input_size):
            raise ValueError(
                f"Unexpected LR size {tuple(low_resolution.shape[-2:])}: "
                f"{input_path}"
            )
        if target.shape[-2:] != (self.target_size, self.target_size):
            raise ValueError(
                f"Unexpected HR size {tuple(target.shape[-2:])}: {target_path}"
            )
        if self.augment:
            low_resolution, target = apply_joint_augmentation(
                low_resolution, target
            )
        return low_resolution, target, relative_path


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    device: torch.device,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        drop_last=shuffle,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def amp_context(enabled: bool, dtype: torch.dtype):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def create_grad_scaler(enabled: bool, dtype: torch.dtype):
    # BF16 has FP32-like exponent range and does not need loss scaling.
    scaler_enabled = enabled and dtype == torch.float16
    # torch.amp.GradScaler is current; the fallback keeps torch 2.0 support.
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    return torch.cuda.amp.GradScaler(enabled=scaler_enabled)


def reduce_learning_rate_after_nonfinite(
    optimizer: Adam,
    factor: float = 0.5,
    minimum: float = 1e-7,
) -> Tuple[float, float]:
    old_lr = max(group["lr"] for group in optimizer.param_groups)
    for group in optimizer.param_groups:
        group["lr"] = max(float(group["lr"]) * factor, minimum)
    new_lr = max(group["lr"] for group in optimizer.param_groups)
    return old_lr, new_lr


def report_nonfinite(message: str, count: int) -> None:
    # Avoid flooding a SLURM log if several consecutive batches are skipped.
    if count <= 5 or count % 100 == 0:
        tqdm.write(f"WARNING: {message} (non-finite count={count})")


def train_epoch(
    model: HistologyMFSR,
    loader: DataLoader,
    optimizer: Adam,
    scaler: Any,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    grad_accumulation: int,
    gradient_clip: float,
    epoch: int,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_samples = 0
    skipped_nonfinite = 0
    progress = tqdm(loader, desc=f"Train {epoch}", unit="batch")

    for step, (low_resolution, target, relative_paths) in enumerate(
        progress, start=1
    ):
        low_resolution = low_resolution.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with amp_context(amp_enabled, amp_dtype):
            prediction = model(low_resolution)
            loss = F.l1_loss(prediction, target)
            backward_loss = loss / grad_accumulation

        if not torch.isfinite(loss):
            prediction_finite = torch.isfinite(prediction)
            finite_values = prediction[prediction_finite]
            prediction_range = (
                "no finite values"
                if finite_values.numel() == 0
                else (
                    f"finite min={finite_values.min().item():.6g}, "
                    f"max={finite_values.max().item():.6g}"
                )
            )
            sample_names = ", ".join(list(relative_paths)[:4])
            optimizer.zero_grad(set_to_none=True)
            old_lr, new_lr = reduce_learning_rate_after_nonfinite(optimizer)
            skipped_nonfinite += 1
            report_nonfinite(
                f"Skipped epoch {epoch}, batch {step}: loss is not finite; "
                f"prediction {prediction_range}; lr {old_lr:.3e} -> "
                f"{new_lr:.3e}; samples: {sample_names}",
                skipped_nonfinite,
            )
            progress.set_postfix(
                l1=(
                    f"{total_loss / total_samples:.6f}"
                    if total_samples
                    else "n/a"
                ),
                skipped=skipped_nonfinite,
            )
            continue
        scaler.scale(backward_loss).backward()

        should_step = step % grad_accumulation == 0 or step == len(loader)
        if should_step:
            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                total_gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    gradient_clip,
                    error_if_nonfinite=False,
                )
                if not torch.isfinite(total_gradient_norm):
                    optimizer.zero_grad(set_to_none=True)
                    old_lr, new_lr = reduce_learning_rate_after_nonfinite(
                        optimizer
                    )
                    skipped_nonfinite += 1
                    report_nonfinite(
                        f"Skipped optimizer update at epoch {epoch}, batch "
                        f"{step}: gradient norm is not finite; lr "
                        f"{old_lr:.3e} -> {new_lr:.3e}",
                        skipped_nonfinite,
                    )
                    if scaler.is_enabled():
                        scaler.update()
                    continue
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch_samples = low_resolution.shape[0]
        total_loss += loss.item() * batch_samples
        total_samples += batch_samples
        progress.set_postfix(l1=f"{total_loss / total_samples:.6f}")
    if skipped_nonfinite:
        tqdm.write(
            f"Epoch {epoch}: skipped {skipped_nonfinite} non-finite "
            "batch(es)/optimizer update(s) and continued training."
        )
    return total_loss / total_samples if total_samples else float("inf")


@torch.inference_mode()
def validate(
    model: HistologyMFSR,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> Tuple[float, float]:
    model.eval()
    total_l1 = 0.0
    total_squared_error = 0.0
    total_values = 0
    total_samples = 0
    skipped_nonfinite = 0

    progress = tqdm(loader, desc="Validation", unit="batch")
    for low_resolution, target, _ in progress:
        low_resolution = low_resolution.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with amp_context(amp_enabled, amp_dtype):
            prediction = model(low_resolution)
            loss = F.l1_loss(prediction, target)

        if not torch.isfinite(loss):
            skipped_nonfinite += 1
            report_nonfinite(
                "Skipped a validation batch because its loss is not finite",
                skipped_nonfinite,
            )
            continue

        prediction = prediction.float().clamp(0.0, 1.0)
        target_float = target.float()
        batch_samples = low_resolution.shape[0]
        total_l1 += loss.item() * batch_samples
        total_samples += batch_samples
        total_squared_error += F.mse_loss(
            prediction, target_float, reduction="sum"
        ).item()
        total_values += target.numel()

    if not total_samples:
        return float("inf"), float("-inf")
    mean_l1 = total_l1 / total_samples
    mean_squared_error = total_squared_error / total_values
    psnr = 10.0 * math.log10(1.0 / max(mean_squared_error, 1e-12))
    return mean_l1, psnr


def atomic_torch_save(payload, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}"
    )
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def save_training_state(
    checkpoint_dir: Path,
    model: HistologyMFSR,
    optimizer: Adam,
    scheduler: ReduceLROnPlateau,
    scaler: Any,
    epoch: int,
    best_val_l1: float,
    args: argparse.Namespace,
    is_best: bool,
) -> None:
    state = {
        "epoch": epoch,
        "best_val_l1": best_val_l1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    atomic_torch_save(state, checkpoint_dir / "last_training_checkpoint.pth")
    atomic_torch_save(
        model.wsss_backbone_state_dict(),
        checkpoint_dir / "last_resnet101_for_wsss_stage2.pth",
    )
    if is_best:
        atomic_torch_save(
            model.wsss_backbone_state_dict(),
            checkpoint_dir / "best_resnet101_for_wsss_stage2.pth",
        )


def load_resume_checkpoint(
    path: Path,
    model: HistologyMFSR,
    optimizer: Adam,
    scheduler: ReduceLROnPlateau,
    scaler: Any,
) -> Tuple[int, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return int(checkpoint["epoch"]) + 1, float(checkpoint["best_val_l1"])


def write_training_config(
    checkpoint_dir: Path,
    args: argparse.Namespace,
    input_dir: Path,
    target_dir: Path,
    input_size: int,
    target_size: int,
    train_samples: int,
    val_samples: int,
) -> None:
    config: Dict[str, object] = {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "resolved_input_dir": str(input_dir),
        "resolved_target_dir": str(target_dir),
        "input_size": input_size,
        "target_size": target_size,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "exported_backbone": "encoder.OPT_resnet",
        "wsss_compatibility": "ResNet101 [3, 4, 23, 3], RGB, no CBAM",
    }
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_path = checkpoint_dir / "training_config.json"
    temporary = config_path.with_name(f".{config_path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, config_path)


def main() -> int:
    args = parse_args()
    input_dir, target_dir, checkpoint_dir = resolve_paths(args)
    input_size, target_size = validate_args(args, input_dir, target_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    manifest_path = input_dir / "manifest.csv"
    train_paths, val_paths = load_train_val_paths(
        manifest_path,
        args.max_train_samples,
        args.max_val_samples,
        args.seed,
    )
    train_dataset = PairedPatchDataset(
        input_dir,
        target_dir,
        train_paths,
        input_size,
        target_size,
        augment=True,
    )
    val_dataset = PairedPatchDataset(
        input_dir,
        target_dir,
        val_paths,
        input_size,
        target_size,
        augment=False,
    )
    if len(train_dataset) < args.batch_size:
        raise ValueError(
            "The training split must contain at least --batch-size samples"
        )

    if args.dry_run:
        low_resolution, target, relative_path = train_dataset[0]
        print(f"Input dataset  : {input_dir}")
        print(f"Target dataset : {target_dir}")
        print(f"Scale          : x{args.scale}")
        print(f"Train/val      : {len(train_dataset):,}/{len(val_dataset):,}")
        print(f"Sample         : {relative_path}")
        print(f"LR tensor      : {tuple(low_resolution.shape)}")
        print(f"HR tensor      : {tuple(target.shape)}")
        print("Dry run passed.")
        return 0

    device = torch.device(args.device)
    amp_enabled = device.type == "cuda" and not args.no_amp
    amp_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.amp_dtype]
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        args.workers,
        shuffle=True,
        device=device,
        seed=args.seed,
    )
    val_loader = make_loader(
        val_dataset,
        args.batch_size,
        args.workers,
        shuffle=False,
        device=device,
        seed=args.seed + 1,
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    write_training_config(
        checkpoint_dir,
        args,
        input_dir,
        target_dir,
        input_size,
        target_size,
        len(train_dataset),
        len(val_dataset),
    )

    use_imagenet_weights = not args.no_pretrained and args.resume is None
    model = HistologyMFSR(
        scale=args.scale,
        pretrained=use_imagenet_weights,
    ).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, 0.999),
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=args.lr_patience
    )
    scaler = create_grad_scaler(amp_enabled, amp_dtype)

    start_epoch = 1
    best_val_l1 = float("inf")
    if args.resume is not None:
        start_epoch, best_val_l1 = load_resume_checkpoint(
            args.resume.expanduser().resolve(),
            model,
            optimizer,
            scheduler,
            scaler,
        )
        model.to(device)

    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Device           : {device}")
    print(
        "Precision        : "
        + (args.amp_dtype if amp_enabled else "float32")
    )
    print(f"Task             : {input_size}x{input_size} -> {target_size}x{target_size}")
    print(f"Input dataset    : {input_dir}")
    print(f"Target dataset   : {target_dir}")
    print(f"Train/validation : {len(train_dataset):,}/{len(val_dataset):,}")
    print(f"Parameters       : {parameters:,}")
    print(f"Checkpoints      : {checkpoint_dir}")

    writer = SummaryWriter(log_dir=str(checkpoint_dir / "tensorboard"))
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            started = time.time()
            train_l1 = train_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device,
                amp_enabled,
                amp_dtype,
                args.grad_accumulation,
                args.gradient_clip,
                epoch,
            )
            val_l1, val_psnr = validate(
                model, val_loader, device, amp_enabled, amp_dtype
            )
            scheduler.step(val_l1)
            is_best = val_l1 < best_val_l1
            best_val_l1 = min(best_val_l1, val_l1)
            save_training_state(
                checkpoint_dir,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_val_l1,
                args,
                is_best,
            )

            learning_rate = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - started
            print(
                f"Epoch {epoch:03d}: train L1={train_l1:.6f}, "
                f"val L1={val_l1:.6f}, PSNR={val_psnr:.3f} dB, "
                f"lr={learning_rate:.3e}, time={elapsed / 60:.1f} min"
            )
            writer.add_scalar("loss/train_l1", train_l1, epoch)
            writer.add_scalar("loss/val_l1", val_l1, epoch)
            writer.add_scalar("metric/val_psnr_db", val_psnr, epoch)
            writer.add_scalar("optimizer/lr", learning_rate, epoch)
            writer.flush()
    finally:
        writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
