#!/usr/bin/env python3
"""Crop MIDOG++ and BRACS RoIs into a merged 224x224 patch dataset.

Output layout::

    DataMerge40x/
      train/                  # MIDOG++ and BRACS patches mixed together
      val/                    # MIDOG++ and BRACS patches mixed together
      test/                   # MIDOG++ and BRACS patches mixed together
      annotations/train.json    # MIDOG++ + BRACS
      annotations/val.json      # MIDOG++ + BRACS
      annotations/test.json     # MIDOG++ + BRACS
      manifest.csv
      dataset_info.json

Design choices
--------------
* BRACS uses ``BRACS_RoI/latest_version`` rather than the ~1 TB WSI set.
* BRACS keeps the official folder splits exactly.
* MIDOG++ has no provided split, so source images/cases are split 80/10/10,
  stratified by tumor type. Patches from one source never cross splits.
* Patches from both datasets are stored directly in the same split directory.
  Their filenames start with ``midogpp_`` or ``bracs_`` to prevent collisions.
* Patches are lossless RGB PNG files. ``manifest.csv`` and the annotation JSON
  files are unified indexes containing both source datasets. MIDOG++ detection
  boxes are exported as COCO-style xywh.
* Cropping is non-overlapping. Only the final patch on an axis may overlap the
  previous patch when the remaining border is smaller than 224 pixels. White
  padding is used only when an entire source dimension is smaller than 224.

Run a metadata-only check first::

    python crop_merge_data.py --dry-run

Create the complete dataset::

    python crop_merge_data.py --workers 8
"""

import argparse
import csv
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


DEFAULT_MIDOG_DIR = Path(
    "/mmlab_students/storageStudents/nguyenvd/thanh01/from_uittogether2/datasets/MIDOGpp/images/"
)
DEFAULT_BRACS_DIR = Path(
    "/mmlab_students/storageStudents/nguyenvd/thanh01/from_uittogether2/datasets/BRACS/"
)
DEFAULT_OUTPUT_DIR = Path(
    "/mmlab_students/storageStudents/nguyenvd/dungutrg/LuuTru/Research/LowResolution/Dataset/DataMerge40x"
)
PATCH_SIZE = 224

SPLITS = ("train", "val", "test")
BRACS_CLASSES = {
    "N": 0,
    "PB": 1,
    "UDH": 2,
    "FEA": 3,
    "ADH": 4,
    "DCIS": 5,
    "IC": 6,
}
BRACS_GROUPS = {
    "N": "benign",
    "PB": "benign",
    "UDH": "benign",
    "FEA": "atypical",
    "ADH": "atypical",
    "DCIS": "malignant",
    "IC": "malignant",
}
BRACS_GLOBAL_CATEGORY_OFFSET = 100
BRACS_NAME_RE = re.compile(
    r"^(?P<wsi>BRACS_\d+)_(?P<label>N|PB|UDH|FEA|ADH|DCIS|IC)_"
    r"(?P<roi>\d+)\.png$",
    re.IGNORECASE,
)

MANIFEST_COLUMNS = (
    "relative_path",
    "split",
    "dataset",
    "task",
    "source_path",
    "source_image",
    "source_id",
    "wsi_id",
    "class_id",
    "class_name",
    "group_name",
    "tumor_type",
    "x",
    "y",
    "valid_width",
    "valid_height",
    "patch_width",
    "patch_height",
    "tissue_fraction",
    "num_objects",
    "num_mitotic",
    "num_hard_negative",
    "image_id",
    "global_category_id",
)


@dataclass
class SourceImage:
    dataset: str
    split: str
    path: str
    source_image: str
    source_id: str
    width: int
    height: int
    task: str
    wsi_id: str = ""
    class_id: str = ""
    class_name: str = ""
    group_name: str = ""
    tumor_type: str = ""
    annotations: Optional[List[Dict[str, Any]]] = None


@dataclass
class CropConfig:
    output_dir: str
    patch_size: int
    min_tissue_fraction: float
    background_threshold: int
    png_compress_level: int
    overwrite: bool
    max_patches_per_source: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop and merge MIDOG++ and BRACS RoIs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--midog-dir", type=Path, default=DEFAULT_MIDOG_DIR)
    parser.add_argument("--bracs-dir", type=Path, default=DEFAULT_BRACS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--midog-split", type=float, nargs=3,
        metavar=("TRAIN", "VAL", "TEST"), default=(0.8, 0.1, 0.1),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--min-tissue-fraction", type=float, default=0.0,
        help="Remove mostly-white patches; 0 keeps every patch.",
    )
    parser.add_argument(
        "--background-threshold", type=int, default=240,
        help="Mean RGB below this value is counted as tissue.",
    )
    parser.add_argument(
        "--png-compress-level", type=int, choices=range(10), default=1,
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Rewrite existing PNG files; otherwise a rerun resumes safely.",
    )
    parser.add_argument(
        "--max-sources-per-dataset", type=int, default=0,
        help="Smoke-test limit; 0 means unlimited.",
    )
    parser.add_argument(
        "--max-patches-per-source", type=int, default=0,
        help="Smoke-test limit; 0 means unlimited.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate inputs and estimate counts without writing files.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if not 0.0 <= args.min_tissue_fraction <= 1.0:
        raise ValueError("--min-tissue-fraction must be in [0, 1]")
    if not 0 <= args.background_threshold <= 255:
        raise ValueError("--background-threshold must be in [0, 255]")
    if any(value < 0 for value in args.midog_split):
        raise ValueError("MIDOG++ split ratios cannot be negative")
    if not math.isclose(sum(args.midog_split), 1.0, abs_tol=1e-8):
        raise ValueError("--midog-split values must sum to 1")


def stable_domain_seed(name: str, seed: int) -> int:
    """Avoid Python's process-randomized hash() for reproducible splits."""
    return seed + sum((index + 1) * ord(char) for index, char in enumerate(name))


def assign_midog_splits(
    images: Sequence[Dict[str, Any]], ratios: Sequence[float], seed: int
) -> Dict[int, str]:
    by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for image in images:
        by_domain[image["tumor_type"]].append(image)

    assignment: Dict[int, str] = {}
    for domain in sorted(by_domain):
        group = sorted(by_domain[domain], key=lambda item: int(item["id"]))
        random.Random(stable_domain_seed(domain, seed)).shuffle(group)
        count = len(group)
        val_count = int(round(count * ratios[1]))
        test_count = int(round(count * ratios[2]))
        if ratios[1] > 0 and count >= 3:
            val_count = max(1, val_count)
        if ratios[2] > 0 and count >= 3:
            test_count = max(1, test_count)
        if val_count + test_count >= count:
            raise ValueError("No MIDOG++ train cases remain for domain %r" % domain)

        for image in group[:test_count]:
            assignment[int(image["id"])] = "test"
        for image in group[test_count:test_count + val_count]:
            assignment[int(image["id"])] = "val"
        for image in group[test_count + val_count:]:
            assignment[int(image["id"])] = "train"
    return assignment


def load_midog_sources(
    midog_dir: Path, ratios: Sequence[float], seed: int
) -> Tuple[List[SourceImage], List[str]]:
    annotation_path = midog_dir / "MIDOGpp.json"
    if not annotation_path.is_file():
        raise FileNotFoundError("MIDOG++ metadata not found: %s" % annotation_path)
    with annotation_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    present = [
        image for image in data["images"]
        if (midog_dir / image["file_name"]).is_file()
    ]
    missing = [
        image["file_name"] for image in data["images"]
        if not (midog_dir / image["file_name"]).is_file()
    ]
    messages = []
    if missing:
        messages.append(
            "Ignored %d metadata-only MIDOG++ entries (%s ... %s)."
            % (len(missing), missing[0], missing[-1])
        )

    annotations: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for annotation in data["annotations"]:
        annotations[int(annotation["image_id"])].append(annotation)
    split_by_id = assign_midog_splits(present, ratios, seed)

    sources = []
    for image in sorted(present, key=lambda item: int(item["id"])):
        image_id = int(image["id"])
        sources.append(SourceImage(
            dataset="midogpp",
            split=split_by_id[image_id],
            path=str(midog_dir / image["file_name"]),
            source_image=image["file_name"],
            source_id=Path(image["file_name"]).stem,
            width=int(image["width"]),
            height=int(image["height"]),
            task="detection",
            tumor_type=image["tumor_type"],
            annotations=annotations.get(image_id, []),
        ))
    return sources, messages


def read_png_size(path: Path) -> Tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Invalid PNG: %s" % path)
    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    return width, height


def load_bracs_sources(bracs_dir: Path) -> Tuple[List[SourceImage], List[str]]:
    roi_root = bracs_dir / "BRACS_RoI" / "latest_version"
    if not roi_root.is_dir():
        raise FileNotFoundError("BRACS RoI directory not found: %s" % roi_root)

    sources: List[SourceImage] = []
    excluded: List[str] = []
    for path in sorted(roi_root.rglob("*.png")):
        relative = path.relative_to(roi_root)
        parts = relative.parts
        if len(parts) != 3 or parts[0] not in SPLITS:
            excluded.append(path.name)
            continue
        match = BRACS_NAME_RE.match(path.name)
        if match is None:
            # Excludes the 9 known .1/.2/... duplicate copies.
            excluded.append(path.name)
            continue
        split = parts[0]
        class_name = match.group("label").upper()
        folder_class = parts[1].split("_", 1)[-1].upper()
        if class_name != folder_class:
            raise ValueError("BRACS class/folder mismatch: %s" % path)
        width, height = read_png_size(path)
        sources.append(SourceImage(
            dataset="bracs",
            split=split,
            path=str(path),
            source_image=path.name,
            source_id=path.stem,
            width=width,
            height=height,
            task="classification",
            wsi_id=match.group("wsi"),
            class_id=str(BRACS_CLASSES[class_name]),
            class_name=class_name,
            group_name=BRACS_GROUPS[class_name],
            annotations=[],
        ))

    messages = []
    if excluded:
        messages.append(
            "Excluded %d noncanonical/duplicate BRACS PNG files: %s"
            % (len(excluded), ", ".join(excluded))
        )

    expected = {"train": 3657, "val": 312, "test": 570}
    actual = Counter(source.split for source in sources)
    for split in SPLITS:
        if actual[split] != expected[split]:
            messages.append(
                "WARNING: expected %d BRACS %s RoIs, found %d."
                % (expected[split], split, actual[split])
            )
    return sources, messages


def axis_origins(length: int, patch_size: int) -> List[int]:
    """Return non-overlapping origins, shifting only the final border patch.

    Examples for patch_size=224:
      * length=672 -> [0, 224, 448] (no overlap)
      * length=700 -> [0, 224, 448, 476] (only the last patch overlaps)
      * length=200 -> [0] (the caller must pad because no prior patch exists)
    """
    if length <= patch_size:
        return [0]
    origins = list(range(0, length - patch_size + 1, patch_size))
    border_origin = length - patch_size
    if origins[-1] != border_origin:
        origins.append(border_origin)
    return origins


def grid_origins(
    width: int, height: int, patch_size: int
) -> Iterable[Tuple[int, int]]:
    xs = axis_origins(width, patch_size)
    ys = axis_origins(height, patch_size)
    for y in ys:
        for x in xs:
            yield x, y


def estimate_grid_count(source: SourceImage, config: CropConfig) -> int:
    nx = len(axis_origins(source.width, config.patch_size))
    ny = len(axis_origins(source.height, config.patch_size))
    count = nx * ny
    if config.max_patches_per_source:
        count = min(count, config.max_patches_per_source)
    return count


def crop_midog_objects(
    annotations: Sequence[Dict[str, Any]], x: int, y: int, patch_size: int
) -> List[Dict[str, Any]]:
    """Clip MIDOG++ source xyxy boxes and return patch-local COCO xywh."""
    objects = []
    for annotation in annotations:
        x1, y1, x2, y2 = map(float, annotation["bbox"])
        left = max(float(x), x1)
        top = max(float(y), y1)
        right = min(float(x + patch_size), x2)
        bottom = min(float(y + patch_size), y2)
        if right <= left or bottom <= top:
            continue
        width = right - left
        height = bottom - top
        objects.append({
            "category_id": int(annotation["category_id"]),
            "bbox": [left - x, top - y, width, height],
            "area": width * height,
            "iscrowd": 0,
            "source_annotation_id": int(annotation["id"]),
            "expert_labels": annotation.get("labels", []),
        })
    return objects


def calculate_tissue_fraction(patch: Image.Image, threshold: int) -> float:
    pixels = np.asarray(patch, dtype=np.uint8)
    return float((pixels.mean(axis=2) < threshold).mean())


def save_png_atomic(image: Image.Image, path: Path, compress_level: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    image.save(str(temporary), format="PNG", compress_level=compress_level)
    os.replace(str(temporary), str(path))


def process_source(
    item: Tuple[SourceImage, CropConfig]
) -> List[Dict[str, Any]]:
    source, config = item
    rows: List[Dict[str, Any]] = []
    Image.MAX_IMAGE_PIXELS = None

    with Image.open(source.path) as opened:
        # Human melanoma MIDOG++ files contain a pyramid; level/frame 0 is HR.
        if getattr(opened, "n_frames", 1) > 1:
            opened.seek(0)
        image = opened.convert("RGB")
        if image.size != (source.width, source.height):
            raise ValueError(
                "Image/metadata size mismatch for %s: %s != %s"
                % (source.path, image.size, (source.width, source.height))
            )

        origins = grid_origins(
            source.width, source.height, config.patch_size,
        )
        for patch_index, (x, y) in enumerate(origins):
            if (
                config.max_patches_per_source > 0
                and patch_index >= config.max_patches_per_source
            ):
                break
            valid_width = min(config.patch_size, source.width - x)
            valid_height = min(config.patch_size, source.height - y)
            crop = image.crop((x, y, x + valid_width, y + valid_height))
            if crop.size != (config.patch_size, config.patch_size):
                patch = Image.new(
                    "RGB", (config.patch_size, config.patch_size), (255, 255, 255)
                )
                patch.paste(crop, (0, 0))
            else:
                patch = crop

            objects = (
                crop_midog_objects(
                    source.annotations or [], x, y, config.patch_size
                )
                if source.dataset == "midogpp" else []
            )
            fraction: Optional[float] = None
            if config.min_tissue_fraction > 0:
                fraction = calculate_tissue_fraction(
                    patch, config.background_threshold
                )
                # Do not remove a labeled MIDOG++ object based on a crude mask.
                if fraction < config.min_tissue_fraction and not objects:
                    continue

            # Put both datasets directly into the same train/val/test folder.
            # The source prefix and coordinates make every filename unique.
            patch_name = "%s_%s_x%06d_y%06d.png" % (
                source.dataset, source.source_id, x, y
            )
            relative_path = Path(source.split) / patch_name
            output_path = Path(config.output_dir) / relative_path
            if config.overwrite or not output_path.is_file():
                save_png_atomic(patch, output_path, config.png_compress_level)

            rows.append({
                "relative_path": relative_path.as_posix(),
                "split": source.split,
                "dataset": source.dataset,
                "task": source.task,
                "source_path": source.path,
                "source_image": source.source_image,
                "source_id": source.source_id,
                "wsi_id": source.wsi_id,
                "class_id": source.class_id,
                "class_name": source.class_name,
                "group_name": source.group_name,
                "tumor_type": source.tumor_type,
                "x": x,
                "y": y,
                "valid_width": valid_width,
                "valid_height": valid_height,
                "patch_width": config.patch_size,
                "patch_height": config.patch_size,
                "tissue_fraction": "" if fraction is None else "%.6f" % fraction,
                "num_objects": len(objects),
                "num_mitotic": sum(o["category_id"] == 1 for o in objects),
                "num_hard_negative": sum(o["category_id"] == 2 for o in objects),
                "image_id": "",
                "global_category_id": (
                    BRACS_GLOBAL_CATEGORY_OFFSET + int(source.class_id)
                    if source.dataset == "bracs" else ""
                ),
                "objects": objects,
            })
    return rows


def limit_sources(
    sources: Sequence[SourceImage], per_dataset: int
) -> List[SourceImage]:
    if per_dataset <= 0:
        return list(sources)
    counts: Counter = Counter()
    selected = []
    for source in sources:
        if counts[source.dataset] >= per_dataset:
            continue
        selected.append(source)
        counts[source.dataset] += 1
    return selected


def print_source_counts(sources: Sequence[SourceImage]) -> None:
    counts = Counter((source.dataset, source.split) for source in sources)
    print("Source images:")
    for dataset in ("midogpp", "bracs"):
        values = [counts[(dataset, split)] for split in SPLITS]
        print(
            "  %-8s train=%d val=%d test=%d total=%d"
            % (dataset, values[0], values[1], values[2], sum(values))
        )


def dry_run(sources: Sequence[SourceImage], config: CropConfig) -> None:
    counts: Counter = Counter()
    for source in sources:
        counts[(source.dataset, source.split)] += estimate_grid_count(source, config)
    print("Estimated patches%s:" % (
        " (upper bound before tissue filtering)"
        if config.min_tissue_fraction > 0 else ""
    ))
    grand_total = 0
    for dataset in ("midogpp", "bracs"):
        values = [counts[(dataset, split)] for split in SPLITS]
        grand_total += sum(values)
        print(
            "  %-8s train=%d val=%d test=%d total=%d"
            % (dataset, values[0], values[1], values[2], sum(values))
        )
    print("  combined total=%d" % grand_total)


def write_json_atomic(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    os.replace(str(temporary), str(path))


def create_dataset(
    sources: Sequence[SourceImage], config: CropConfig,
    workers: int, args: argparse.Namespace, audit_messages: Sequence[str],
) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    manifest_tmp = output_dir / "manifest.csv.tmp"

    merged_images: Dict[str, List[Dict[str, Any]]] = {s: [] for s in SPLITS}
    merged_annotations: Dict[str, List[Dict[str, Any]]] = {s: [] for s in SPLITS}
    next_image_id = {s: 1 for s in SPLITS}
    next_annotation_id = {s: 1 for s in SPLITS}
    patch_counts: Counter = Counter()
    tasks = [(source, config) for source in sources]

    executor = None
    if workers == 1:
        results = map(process_source, tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        # executor.map preserves input order, keeping the manifest deterministic.
        results = executor.map(process_source, tasks, chunksize=1)

    try:
        with manifest_tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            for source_rows in tqdm(
                results, total=len(tasks), desc="Cropping", unit="source"
            ):
                for row in source_rows:
                    objects = row.pop("objects")
                    split = row["split"]
                    patch_counts[(row["dataset"], split)] += 1
                    image_id = next_image_id[split]
                    next_image_id[split] += 1
                    row["image_id"] = image_id
                    image_record: Dict[str, Any] = {
                        "id": image_id,
                        "file_name": row["relative_path"],
                        "width": config.patch_size,
                        "height": config.patch_size,
                        "split": split,
                        "source_dataset": row["dataset"],
                        "task": row["task"],
                        "source_image": row["source_image"],
                        "source_id": row["source_id"],
                        "x": row["x"],
                        "y": row["y"],
                    }
                    if row["dataset"] == "midogpp":
                        image_record["tumor_type"] = row["tumor_type"]
                        for obj in objects:
                            obj["id"] = next_annotation_id[split]
                            obj["image_id"] = image_id
                            obj["annotation_type"] = "detection"
                            obj["source_dataset"] = "midogpp"
                            next_annotation_id[split] += 1
                            merged_annotations[split].append(obj)
                    else:
                        local_class_id = int(row["class_id"])
                        global_category_id = (
                            BRACS_GLOBAL_CATEGORY_OFFSET + local_class_id
                        )
                        image_record.update({
                            "wsi_id": row["wsi_id"],
                            "class_id": local_class_id,
                            "class_name": row["class_name"],
                            "group_name": row["group_name"],
                        })
                        merged_annotations[split].append({
                            "id": next_annotation_id[split],
                            "image_id": image_id,
                            "category_id": global_category_id,
                            "source_category_id": local_class_id,
                            "annotation_type": "classification",
                            "source_dataset": "bracs",
                        })
                        next_annotation_id[split] += 1
                    merged_images[split].append(image_record)
                    writer.writerow({key: row.get(key, "") for key in MANIFEST_COLUMNS})
        os.replace(str(manifest_tmp), str(manifest_path))
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    merged_categories = [
        {
            "id": 1,
            "name": "mitotic figure",
            "task": "detection",
            "source_dataset": "midogpp",
            "source_category_id": 1,
        },
        {
            "id": 2,
            "name": "not mitotic figure",
            "task": "detection",
            "source_dataset": "midogpp",
            "source_category_id": 2,
        },
    ]
    merged_categories.extend(
        {
            "id": BRACS_GLOBAL_CATEGORY_OFFSET + class_id,
            "name": class_name,
            "task": "classification",
            "source_dataset": "bracs",
            "source_category_id": class_id,
            "group": BRACS_GROUPS[class_name],
        }
        for class_name, class_id in BRACS_CLASSES.items()
    )
    for split in SPLITS:
        write_json_atomic(
            output_dir / "annotations" / ("%s.json" % split),
            {
                "info": {
                    "description": "Unified MIDOG++ and BRACS patch annotations",
                    "patch_size": config.patch_size,
                    "bbox_format": "COCO xywh for detection annotations only",
                    "schema": (
                        "Detection annotations contain bbox/area/iscrowd; "
                        "classification annotations contain one category per image."
                    ),
                },
                "images": merged_images[split],
                "annotations": merged_annotations[split],
                "categories": merged_categories,
            },
        )

    counts = {
        dataset: {
            split: patch_counts[(dataset, split)] for split in SPLITS
        }
        for dataset in ("midogpp", "bracs")
    }
    serializable_args = {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }
    write_json_atomic(output_dir / "dataset_info.json", {
        "format": (
            "Unified RGB PNG dataset + CSV manifest + one merged annotation "
            "JSON per split"
        ),
        "arguments": serializable_args,
        "crop_config": asdict(config),
        "source_counts": dict(Counter(s.dataset for s in sources)),
        "patch_counts": counts,
        "bracs_classes": BRACS_CLASSES,
        "audit_messages": list(audit_messages),
    })
    print("Dataset written to: %s" % output_dir)
    print("Manifest: %s" % manifest_path)
    print("Patch counts: %s" % counts)


def main() -> None:
    args = parse_args()
    validate_args(args)

    midog_sources, midog_messages = load_midog_sources(
        args.midog_dir, args.midog_split, args.seed
    )
    bracs_sources, bracs_messages = load_bracs_sources(args.bracs_dir)
    messages = midog_messages + bracs_messages
    for message in messages:
        print("AUDIT: %s" % message, file=sys.stderr)

    sources = sorted(
        midog_sources + bracs_sources,
        key=lambda source: (source.dataset, source.split, source.source_id),
    )
    sources = limit_sources(sources, args.max_sources_per_dataset)
    print_source_counts(sources)
    config = CropConfig(
        output_dir=str(args.output_dir),
        patch_size=PATCH_SIZE,
        min_tissue_fraction=args.min_tissue_fraction,
        background_threshold=args.background_threshold,
        png_compress_level=args.png_compress_level,
        overwrite=args.overwrite,
        max_patches_per_source=args.max_patches_per_source,
    )
    if args.dry_run:
        dry_run(sources, config)
    else:
        create_dataset(sources, config, args.workers, args, messages)


if __name__ == "__main__":
    main()
