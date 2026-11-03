#!/usr/bin/env python3
"""Resize DataMerge40x while preserving its unified dataset structure.

The script is intentionally run once per target resolution. Typical usage::

    python resize.py --size 112 --workers 8
    python resize.py --size 56 --workers 8

Images keep the same relative paths and filenames. COCO-style detection boxes
and their areas are scaled to the resized patch coordinate system. Source crop
coordinates (x, y, valid_width, valid_height) remain in the original 40x image
coordinate system so a patch can still be traced back to its source image.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import ijson
from PIL import Image
from tqdm import tqdm


DEFAULT_INPUT_DIR = Path(
    "/datastore/uittogether/LuuTru/Dungdm/Research/LowResolution/"
    "Dataset/DataMerge40x"
)
DEFAULT_OUTPUT_DIRS = {
    112: Path(
        "/datastore/uittogether/LuuTru/Dungdm/Research/LowResolution/"
        "Dataset/DataMerge20x"
    ),
    56: Path(
        "/datastore/uittogether/LuuTru/Dungdm/Research/LowResolution/"
        "Dataset/DataMerge10x"
    ),
}
SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
FORMAT_BY_SUFFIX = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".tif": "TIFF",
    ".tiff": "TIFF",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resize all patches in a unified DataMerge dataset and update its "
            "manifest/annotations."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output dataset directory. If omitted, size 112 maps to "
            "DataMerge20x and size 56 maps to DataMerge10x."
        ),
    )
    parser.add_argument(
        "--size",
        type=int,
        required=True,
        help="Target square patch size, for example 112 or 56.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--png-compress-level", type=int, choices=range(10), default=1
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite images that already exist; otherwise resume by skipping them.",
    )
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()
    if args.size not in DEFAULT_OUTPUT_DIRS:
        raise ValueError(
            "--output-dir is required when --size is not one of: 112, 56"
        )
    return DEFAULT_OUTPUT_DIRS[args.size]


def validate_layout(input_dir: Path) -> None:
    required = [
        input_dir / "manifest.csv",
        input_dir / "dataset_info.json",
        *(input_dir / split for split in SPLITS),
        *(input_dir / "annotations" / f"{split}.json" for split in SPLITS),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        formatted = "\n  - ".join(missing)
        raise FileNotFoundError(f"Input dataset is missing:\n  - {formatted}")


def validate_args(args: argparse.Namespace, input_dir: Path, output_dir: Path) -> None:
    if args.size < 1:
        raise ValueError("--size must be a positive integer")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if input_dir == output_dir:
        raise ValueError("--output-dir must be different from --input-dir")
    if output_dir.is_relative_to(input_dir):
        raise ValueError("--output-dir cannot be located inside --input-dir")
    validate_layout(input_dir)


def read_dataset_info(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def source_patch_size(dataset_info: Dict[str, Any]) -> int:
    candidates = (
        dataset_info.get("image_size"),
        dataset_info.get("crop_config", {}).get("patch_size"),
    )
    for candidate in candidates:
        if isinstance(candidate, int) and candidate > 0:
            return candidate
        if (
            isinstance(candidate, list)
            and len(candidate) == 2
            and candidate[0] == candidate[1]
        ):
            return int(candidate[0])
    raise ValueError("Cannot determine source patch size from dataset_info.json")


def validate_existing_output(
    output_dir: Path, target_size: int, overwrite: bool
) -> None:
    info_path = output_dir / "dataset_info.json"
    if not info_path.exists() or overwrite:
        return
    info = read_dataset_info(info_path)
    existing_size = info.get("resize_config", {}).get("target_size")
    if existing_size is None:
        existing_size = info.get("image_size")
        if isinstance(existing_size, list) and existing_size:
            existing_size = existing_size[0]
    if existing_size != target_size:
        raise ValueError(
            f"{output_dir} already describes size {existing_size}, not "
            f"{target_size}. Choose another directory or use --overwrite."
        )


def expected_image_count(dataset_info: Dict[str, Any]) -> Optional[int]:
    patch_counts = dataset_info.get("patch_counts")
    if not isinstance(patch_counts, dict):
        return None
    total = 0
    try:
        for dataset_counts in patch_counts.values():
            total += sum(int(dataset_counts[split]) for split in SPLITS)
    except (KeyError, TypeError, ValueError):
        return None
    return total


def iter_image_jobs(
    input_dir: Path,
    output_dir: Path,
    source_size: int,
    target_size: int,
    overwrite: bool,
    png_compress_level: int,
) -> Iterator[Tuple[str, str, int, int, bool, int]]:
    for split in SPLITS:
        split_dir = input_dir / split
        for root, _, files in os.walk(split_dir):
            root_path = Path(root)
            for name in files:
                source_path = root_path / name
                if source_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                relative_path = source_path.relative_to(input_dir)
                yield (
                    str(source_path),
                    str(output_dir / relative_path),
                    source_size,
                    target_size,
                    overwrite,
                    png_compress_level,
                )


def resize_one(job: Tuple[str, str, int, int, bool, int]) -> str:
    (
        source_name,
        destination_name,
        source_size,
        target_size,
        overwrite,
        png_compress_level,
    ) = job
    source_path = Path(source_name)
    destination_path = Path(destination_name)
    if destination_path.exists() and not overwrite:
        with Image.open(destination_path) as existing_image:
            if existing_image.size != (target_size, target_size):
                raise ValueError(
                    f"Existing output has size {existing_image.width}x"
                    f"{existing_image.height}, expected {target_size}x"
                    f"{target_size}: {destination_path}. Use --overwrite to "
                    "replace it."
                )
        return "skipped"

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination_path.with_name(
        f".{destination_path.name}.tmp-{os.getpid()}"
    )
    try:
        with Image.open(source_path) as image:
            if image.size != (source_size, source_size):
                raise ValueError(
                    f"Expected {source_size}x{source_size}, got "
                    f"{image.width}x{image.height}: {source_path}"
                )
            resized = image.convert("RGB").resize(
                (target_size, target_size), Image.Resampling.LANCZOS
            )
            save_options: Dict[str, Any] = {}
            if destination_path.suffix.lower() == ".png":
                save_options["compress_level"] = png_compress_level
            resized.save(
                temporary_path,
                format=FORMAT_BY_SUFFIX[destination_path.suffix.lower()],
                **save_options,
            )
        os.replace(temporary_path, destination_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return "created"


def create_output_directories(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "annotations").mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (output_dir / split).mkdir(parents=True, exist_ok=True)


def resize_images(
    input_dir: Path,
    output_dir: Path,
    source_size: int,
    target_size: int,
    workers: int,
    overwrite: bool,
    png_compress_level: int,
    total: Optional[int],
) -> Tuple[int, int]:
    jobs = iter_image_jobs(
        input_dir,
        output_dir,
        source_size,
        target_size,
        overwrite,
        png_compress_level,
    )
    created = 0
    skipped = 0
    with Pool(processes=workers) as pool:
        results = pool.imap_unordered(resize_one, jobs, chunksize=32)
        for status in tqdm(results, total=total, unit="image", desc="Resize"):
            if status == "created":
                created += 1
            else:
                skipped += 1
    return created, skipped


def temporary_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp-{os.getpid()}")


def scaled_number(value: Any, scale: float) -> float:
    return round(float(value) * scale, 6)


def transform_image_record(
    image: Dict[str, Any], source_size: int, target_size: int
) -> Dict[str, Any]:
    if image.get("width") != source_size or image.get("height") != source_size:
        raise ValueError(
            f"Annotation image id={image.get('id')} has unexpected size "
            f"{image.get('width')}x{image.get('height')}"
        )
    image["width"] = target_size
    image["height"] = target_size
    return image


def transform_annotation_record(
    annotation: Dict[str, Any], scale: float
) -> Dict[str, Any]:
    bbox = annotation.get("bbox")
    if bbox is None:
        return annotation
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(
            f"Annotation id={annotation.get('id')} has invalid bbox: {bbox}"
        )
    scaled_bbox = [scaled_number(value, scale) for value in bbox]
    annotation["bbox"] = scaled_bbox
    annotation["area"] = round(scaled_bbox[2] * scaled_bbox[3], 6)
    return annotation


def read_json_value(path: Path, prefix: str) -> Any:
    with path.open("rb") as handle:
        iterator = ijson.items(handle, prefix, use_float=True)
        try:
            return next(iterator)
        except StopIteration as error:
            raise ValueError(f"Missing JSON field '{prefix}' in {path}") from error


def write_streamed_array(
    output_handle: Any,
    source_path: Path,
    prefix: str,
    transform: Any,
) -> int:
    count = 0
    first = True
    with source_path.open("rb") as input_handle:
        for item in ijson.items(input_handle, prefix, use_float=True):
            if not first:
                output_handle.write(",\n")
            output_handle.write("    ")
            json.dump(transform(item), output_handle, ensure_ascii=False)
            first = False
            count += 1
    if count:
        output_handle.write("\n")
    return count


def resize_annotation_file(
    source_path: Path,
    destination_path: Path,
    source_size: int,
    target_size: int,
) -> Tuple[int, int]:
    scale = target_size / source_size
    info = read_json_value(source_path, "info")
    categories = read_json_value(source_path, "categories")
    info["patch_size"] = target_size
    info["resized_from_patch_size"] = source_size
    info["resize_scale"] = scale
    info["resize_interpolation"] = "Pillow LANCZOS"

    temporary_path = temporary_sibling(destination_path)
    try:
        with temporary_path.open("w", encoding="utf-8") as output_handle:
            output_handle.write("{\n  \"info\": ")
            json.dump(info, output_handle, ensure_ascii=False, indent=2)
            output_handle.write(",\n  \"images\": [\n")
            image_count = write_streamed_array(
                output_handle,
                source_path,
                "images.item",
                lambda item: transform_image_record(
                    item, source_size, target_size
                ),
            )
            output_handle.write("  ],\n  \"annotations\": [\n")
            annotation_count = write_streamed_array(
                output_handle,
                source_path,
                "annotations.item",
                lambda item: transform_annotation_record(item, scale),
            )
            output_handle.write("  ],\n  \"categories\": ")
            json.dump(categories, output_handle, ensure_ascii=False, indent=2)
            output_handle.write("\n}\n")
        os.replace(temporary_path, destination_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return image_count, annotation_count


def resize_manifest(
    source_path: Path, destination_path: Path, target_size: int
) -> int:
    temporary_path = temporary_sibling(destination_path)
    row_count = 0
    try:
        with source_path.open("r", encoding="utf-8", newline="") as source_handle:
            reader = csv.DictReader(source_handle)
            if reader.fieldnames is None:
                raise ValueError(f"Manifest has no header: {source_path}")
            required = {"patch_width", "patch_height"}
            missing = required.difference(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"Manifest is missing columns: {', '.join(sorted(missing))}"
                )
            with temporary_path.open(
                "w", encoding="utf-8", newline=""
            ) as destination_handle:
                writer = csv.DictWriter(destination_handle, fieldnames=reader.fieldnames)
                writer.writeheader()
                for row in reader:
                    row["patch_width"] = str(target_size)
                    row["patch_height"] = str(target_size)
                    writer.writerow(row)
                    row_count += 1
        os.replace(temporary_path, destination_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return row_count


def write_dataset_info(
    source_info: Dict[str, Any],
    source_dir: Path,
    output_dir: Path,
    source_size: int,
    target_size: int,
    workers: int,
) -> None:
    info = copy.deepcopy(source_info)
    original_crop_config = copy.deepcopy(info.get("crop_config", {}))
    scale = target_size / source_size

    info["format"] = (
        "Unified resized RGB image dataset + CSV manifest + one merged "
        "annotation JSON per split"
    )
    info["parent_dataset"] = str(source_dir)
    info["image_size"] = [target_size, target_size]
    info["source_crop_config"] = original_crop_config
    info["resize_config"] = {
        "input_dir": str(source_dir),
        "output_dir": str(output_dir),
        "source_size": source_size,
        "target_size": target_size,
        "scale": scale,
        "interpolation": "Pillow LANCZOS",
        "workers": workers,
        "coordinate_policy": (
            "annotation bbox/area use resized patch coordinates; manifest and "
            "image-record x/y plus valid_width/valid_height remain in original "
            "40x source-image coordinates"
        ),
    }
    if isinstance(info.get("arguments"), dict):
        info["arguments"]["output_dir"] = str(output_dir)
        info["arguments"]["resize_input_dir"] = str(source_dir)
        info["arguments"]["resize_target_size"] = target_size
    if isinstance(info.get("crop_config"), dict):
        info["crop_config"]["output_dir"] = str(output_dir)
        info["crop_config"]["patch_size"] = target_size

    destination_path = output_dir / "dataset_info.json"
    temporary_path = temporary_sibling(destination_path)
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(info, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_path, destination_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def copy_extra_metadata(input_dir: Path, output_dir: Path) -> None:
    handled = {
        Path("manifest.csv"),
        Path("dataset_info.json"),
        *(Path("annotations") / f"{split}.json" for split in SPLITS),
    }
    excluded_roots = {input_dir / split for split in SPLITS}
    for root, directories, files in os.walk(input_dir):
        root_path = Path(root)
        directories[:] = [
            name for name in directories if root_path / name not in excluded_roots
        ]
        for name in files:
            source_path = root_path / name
            relative_path = source_path.relative_to(input_dir)
            if relative_path in handled:
                continue
            destination_path = output_dir / relative_path
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = resolve_output_dir(args)
    validate_args(args, input_dir, output_dir)
    validate_existing_output(output_dir, args.size, args.overwrite)

    dataset_info = read_dataset_info(input_dir / "dataset_info.json")
    original_size = source_patch_size(dataset_info)
    if args.size >= original_size:
        raise ValueError(
            f"--size must be smaller than source size {original_size}; "
            f"received {args.size}"
        )

    create_output_directories(output_dir)
    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Resize: {original_size}x{original_size} -> {args.size}x{args.size}")

    created, skipped = resize_images(
        input_dir=input_dir,
        output_dir=output_dir,
        source_size=original_size,
        target_size=args.size,
        workers=args.workers,
        overwrite=args.overwrite,
        png_compress_level=args.png_compress_level,
        total=expected_image_count(dataset_info),
    )
    actual_image_count = created + skipped
    expected_count = expected_image_count(dataset_info)
    if expected_count is not None and actual_image_count != expected_count:
        raise ValueError(
            f"Found {actual_image_count:,} image files, but dataset_info.json "
            f"describes {expected_count:,} patches"
        )

    split_counts: Dict[str, Dict[str, int]] = {}
    for split in SPLITS:
        images, annotations = resize_annotation_file(
            input_dir / "annotations" / f"{split}.json",
            output_dir / "annotations" / f"{split}.json",
            original_size,
            args.size,
        )
        split_counts[split] = {"images": images, "annotations": annotations}

    manifest_rows = resize_manifest(
        input_dir / "manifest.csv", output_dir / "manifest.csv", args.size
    )
    annotation_image_count = sum(
        counts["images"] for counts in split_counts.values()
    )
    if manifest_rows != actual_image_count:
        raise ValueError(
            f"Manifest has {manifest_rows:,} rows, but there are "
            f"{actual_image_count:,} images"
        )
    if annotation_image_count != actual_image_count:
        raise ValueError(
            f"Annotations describe {annotation_image_count:,} images, but "
            f"there are {actual_image_count:,} image files"
        )
    copy_extra_metadata(input_dir, output_dir)
    write_dataset_info(
        dataset_info,
        input_dir,
        output_dir,
        original_size,
        args.size,
        args.workers,
    )

    print(f"Images created: {created:,}; skipped: {skipped:,}")
    print(f"Manifest rows: {manifest_rows:,}")
    for split, counts in split_counts.items():
        print(
            f"{split}: {counts['images']:,} images, "
            f"{counts['annotations']:,} annotations"
        )
    print("Done.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2)
