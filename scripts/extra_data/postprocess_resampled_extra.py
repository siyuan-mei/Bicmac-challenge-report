from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk

try:
    from .common import (
        CaseResult,
        DEFAULT_EXTRA_ROOT,
        center_crop_to_fov,
        center_crop_xy,
        clip_image,
        estimate_center_fov_crop,
        make_body_mask_from_ct,
        outputs_exist,
        print_summary,
        read_image,
        write_image,
        write_manifest,
    )
except ImportError:
    from common import (
        CaseResult,
        DEFAULT_EXTRA_ROOT,
        center_crop_to_fov,
        center_crop_xy,
        clip_image,
        estimate_center_fov_crop,
        make_body_mask_from_ct,
        outputs_exist,
        print_summary,
        read_image,
        write_image,
        write_manifest,
    )


DATASET_NAMES = ("autopet", "quadra_hc")


@dataclass(frozen=True)
class PairCase:
    case_id: str
    case_dir: Path
    ct_path: Path
    pet_path: Path


@dataclass(frozen=True)
class PostprocessConfig:
    dataset_name: str
    dry_run: bool
    overwrite: bool
    fov_crop: bool
    fov_padding_value: float | None
    fov_padding_tolerance: float
    fov_margin: int
    body_threshold: float
    autopet_clip: tuple[float, float]
    no_autopet_clip: bool
    autopet_crop_size: int
    skip_processed: bool
    itk_threads: int


def same_path(left: Path, right: Path) -> bool:
    return str(left.resolve()).casefold() == str(right.resolve()).casefold()


def find_pair_cases(dataset_root: Path) -> list[PairCase]:
    cases = []
    if not dataset_root.exists():
        return cases

    for case_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        ct_path = case_dir / "ct.nii.gz"
        pet_path = case_dir / "pet.nii.gz"
        if not ct_path.exists() or not pet_path.exists():
            print(f"[missing] {case_dir}: ct.nii.gz or pet.nii.gz")
            continue
        cases.append(
            PairCase(
                case_id=case_dir.name,
                case_dir=case_dir,
                ct_path=ct_path,
                pet_path=pet_path,
            )
        )

    return cases


def crop_like_ct(image: sitk.Image, crop_info: dict[str, float | int | bool]) -> sitk.Image:
    if not crop_info["applied"]:
        return image
    return center_crop_xy(
        image,
        int(crop_info["start_x"]),
        int(crop_info["start_y"]),
        int(crop_info["size_x"]),
        int(crop_info["size_y"]),
    )


def center_crop_fixed_xy(
    image: sitk.Image,
    requested_size: int,
) -> tuple[sitk.Image, dict[str, float | int | bool]]:
    size = image.GetSize()
    side = min(int(requested_size), int(size[0]), int(size[1]))
    if side <= 0:
        raise ValueError(f"Crop size must be positive, got {requested_size}")

    start_x = int((int(size[0]) - side) // 2)
    start_y = int((int(size[1]) - side) // 2)
    applied = side < int(size[0]) or side < int(size[1])
    crop_info = {
        "applied": bool(applied),
        "padding_value": 0.0,
        "tolerance": 0.0,
        "start_x": start_x,
        "start_y": start_y,
        "size_x": side,
        "size_y": side,
        "orig_size_x": int(size[0]),
        "orig_size_y": int(size[1]),
    }

    if not applied:
        return image, crop_info

    return center_crop_xy(image, start_x, start_y, side, side), crop_info


def min_max(image: sitk.Image) -> tuple[float, float]:
    array = sitk.GetArrayViewFromImage(image)
    return float(np.nanmin(array)), float(np.nanmax(array))


def set_simpleitk_threads(n_threads: int) -> None:
    n_threads = max(1, int(n_threads))
    if hasattr(sitk, "ProcessObject_SetGlobalDefaultNumberOfThreads"):
        sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(n_threads)
    elif hasattr(sitk, "ProcessObject"):
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(n_threads)


def headers_match(paths: list[Path]) -> bool:
    if not all(path.exists() for path in paths):
        return False

    readers = []
    for path in paths:
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(path))
        reader.ReadImageInformation()
        readers.append(reader)

    first_size = readers[0].GetSize()
    first_spacing = readers[0].GetSpacing()
    first_direction = readers[0].GetDirection()
    first_origin = readers[0].GetOrigin()
    for reader in readers[1:]:
        if reader.GetSize() != first_size:
            return False
        if any(abs(a - b) > 1e-4 for a, b in zip(reader.GetSpacing(), first_spacing)):
            return False
        if any(abs(a - b) > 1e-4 for a, b in zip(reader.GetDirection(), first_direction)):
            return False
        if any(abs(a - b) > 1e-3 for a, b in zip(reader.GetOrigin(), first_origin)):
            return False

    return True


def already_processed(
    output_case_dir: Path,
    dataset_name: str,
    autopet_crop_size: int,
) -> tuple[bool, str]:
    ct_path = output_case_dir / "ct.nii.gz"
    pet_path = output_case_dir / "pet.nii.gz"
    body_path = output_case_dir / "body_mask.nii.gz"
    if not headers_match([ct_path, pet_path, body_path]):
        return False, ""

    reader = sitk.ImageFileReader()
    reader.SetFileName(str(ct_path))
    reader.ReadImageInformation()
    size = reader.GetSize()

    if dataset_name == "autopet" and autopet_crop_size > 0:
        if int(size[0]) != int(autopet_crop_size) or int(size[1]) != int(autopet_crop_size):
            return False, ""
        return True, f"already processed: size={size[0]}x{size[1]}"

    return True, f"already processed: size={size[0]}x{size[1]}"


def process_case(
    case: PairCase,
    output_case_dir: Path,
    config: PostprocessConfig,
) -> tuple[CaseResult, dict[str, str]]:
    in_place = same_path(case.case_dir, output_case_dir)
    ct_out = output_case_dir / "ct.nii.gz"
    pet_out = output_case_dir / "pet.nii.gz"
    body_out = output_case_dir / "body_mask.nii.gz"

    if (
        config.skip_processed
        and not config.overwrite
        and outputs_exist([ct_out, pet_out, body_out])
    ):
        done, message = already_processed(
            output_case_dir,
            config.dataset_name,
            config.autopet_crop_size,
        )
        if done:
            result = CaseResult(case.case_id, "skipped", output_case_dir, message)
            return result, manifest_row(case, output_case_dir, result)

    if not in_place and not config.overwrite and outputs_exist([ct_out, pet_out, body_out]):
        result = CaseResult(case.case_id, "skipped", output_case_dir, "outputs exist")
        return result, manifest_row(case, output_case_dir, result)

    ct_image = read_image(case.ct_path, sitk.sitkFloat32)
    original_size = ct_image.GetSize()

    crop_info = {
        "applied": False,
        "padding_value": 0.0,
        "tolerance": config.fov_padding_tolerance,
        "start_x": 0,
        "start_y": 0,
        "size_x": original_size[0],
        "size_y": original_size[1],
        "orig_size_x": original_size[0],
        "orig_size_y": original_size[1],
    }
    crop_mode = "none"
    estimated_fov = None
    if config.fov_crop and config.dataset_name == "autopet" and config.autopet_crop_size > 0:
        estimated_fov = estimate_center_fov_crop(
            ct_image,
            padding_value=config.fov_padding_value,
            tolerance=config.fov_padding_tolerance,
            margin=config.fov_margin,
        )
        ct_cropped, crop_info = center_crop_fixed_xy(ct_image, config.autopet_crop_size)
        crop_mode = f"fixed_{config.autopet_crop_size}"
    elif config.fov_crop:
        ct_cropped, crop_info = center_crop_to_fov(
            ct_image,
            padding_value=config.fov_padding_value,
            tolerance=config.fov_padding_tolerance,
            margin=config.fov_margin,
        )
        crop_mode = "fov"
    else:
        ct_cropped = ct_image

    if config.dataset_name == "autopet" and not config.no_autopet_clip:
        ct_cropped = clip_image(
            ct_cropped,
            config.autopet_clip[0],
            config.autopet_clip[1],
            sitk.sitkFloat32,
        )

    new_size = ct_cropped.GetSize()
    message = "crop=none"
    if crop_info["applied"]:
        message = (
            f"crop={crop_mode} "
            f"{original_size[0]}x{original_size[1]}->{new_size[0]}x{new_size[1]}"
        )
    if estimated_fov is not None:
        fov_size = f"{estimated_fov['size_x']}x{estimated_fov['size_y']}"
        message = f"{message}, estimated_fov={fov_size}"
        if (
            int(estimated_fov["size_x"]) > int(new_size[0])
            or int(estimated_fov["size_y"]) > int(new_size[1])
        ):
            message = f"{message}, warning=fov_exceeds_crop"

    result = CaseResult(
        case.case_id,
        "dry_run" if config.dry_run else "written",
        output_case_dir,
        message,
    )
    row = manifest_row(case, output_case_dir, result)
    row.update(
        {
            "dataset": config.dataset_name,
            "orig_size": "x".join(str(v) for v in original_size),
            "new_size": "x".join(str(v) for v in new_size),
            "spacing": "x".join(f"{v:g}" for v in ct_cropped.GetSpacing()),
            "crop_applied": str(bool(crop_info["applied"])),
            "crop_mode": crop_mode,
            "crop_start_x": str(crop_info["start_x"]),
            "crop_start_y": str(crop_info["start_y"]),
            "crop_size_x": str(crop_info["size_x"]),
            "crop_size_y": str(crop_info["size_y"]),
            "body_threshold": f"{config.body_threshold:g}",
        }
    )

    if config.dataset_name == "autopet" and not config.no_autopet_clip:
        lo, hi = min_max(ct_cropped)
        row["ct_min_after_clip"] = f"{lo:g}"
        row["ct_max_after_clip"] = f"{hi:g}"
    if estimated_fov is not None:
        row["estimated_fov_size_x"] = str(estimated_fov["size_x"])
        row["estimated_fov_size_y"] = str(estimated_fov["size_y"])
        row["fov_exceeds_crop"] = str(
            int(estimated_fov["size_x"]) > int(new_size[0])
            or int(estimated_fov["size_y"]) > int(new_size[1])
        )

    if config.dry_run:
        return result, row

    pet_image = read_image(case.pet_path, sitk.sitkFloat32)
    if pet_image.GetSize() != original_size:
        raise ValueError(
            f"PET size {pet_image.GetSize()} does not match CT size {original_size}"
        )

    pet_cropped = crop_like_ct(pet_image, crop_info)
    body_mask = make_body_mask_from_ct(ct_cropped, threshold=config.body_threshold)

    write_image(ct_cropped, ct_out)
    write_image(pet_cropped, pet_out)
    write_image(body_mask, body_out)

    return result, row


def process_case_job(
    index: int,
    case: PairCase,
    output_case_dir: Path,
    config: PostprocessConfig,
) -> tuple[int, CaseResult, dict[str, str]]:
    set_simpleitk_threads(config.itk_threads)
    result, row = process_case(case, output_case_dir, config)
    return index, result, row


def manifest_row(case: PairCase, output_case_dir: Path, result: CaseResult) -> dict[str, str]:
    return {
        "case_id": case.case_id,
        "input_dir": str(case.case_dir),
        "output_dir": str(output_case_dir),
        "ct_output": str(output_case_dir / "ct.nii.gz"),
        "pet_output": str(output_case_dir / "pet.nii.gz"),
        "body_mask_output": str(output_case_dir / "body_mask.nii.gz"),
        "status": result.status,
        "message": result.message,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Postprocess already resampled/reoriented AutoPET and QUADRA_HC "
            "extra_data cases: center-crop CT/PET FOV, clip AutoPET CT, "
            "and write body_mask.nii.gz."
        )
    )
    parser.add_argument("--extra-root", type=Path, default=DEFAULT_EXTRA_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Defaults to --extra-root, so CT/PET are updated in place.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_NAMES,
        default=DATASET_NAMES,
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional per-dataset limit.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-skip-processed",
        action="store_true",
        help="Revisit cases that already have CT/PET/body_mask with expected headers.",
    )
    parser.add_argument("--no-manifest", action="store_true")
    parser.add_argument("--no-fov-crop", action="store_true")
    parser.add_argument("--fov-padding-value", type=float, default=None)
    parser.add_argument("--fov-padding-tolerance", type=float, default=1.0)
    parser.add_argument("--fov-margin", type=int, default=0)
    parser.add_argument("--body-threshold", type=float, default=-500.0)
    parser.add_argument("--autopet-clip", nargs=2, type=float, default=(-1024.0, 3072.0))
    parser.add_argument("--no-autopet-clip", action="store_true")
    parser.add_argument(
        "--autopet-crop-size",
        type=int,
        default=400,
        help=(
            "Fixed center crop size for already-resampled AutoPET CT/PET. "
            "Use 0 to fall back to per-case FOV crop."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel worker processes. 0 picks a conservative automatic value.",
    )
    parser.add_argument(
        "--itk-threads",
        type=int,
        default=1,
        help="SimpleITK threads per worker; keep 1 when using multiple workers.",
    )
    return parser


def resolve_workers(value: int) -> int:
    if value < 0:
        raise ValueError("--workers must be >= 0")
    if value > 0:
        return value
    return max(1, min(4, (os.cpu_count() or 1)))


def build_config(args: argparse.Namespace, dataset_name: str) -> PostprocessConfig:
    return PostprocessConfig(
        dataset_name=dataset_name,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        fov_crop=not args.no_fov_crop,
        fov_padding_value=args.fov_padding_value,
        fov_padding_tolerance=args.fov_padding_tolerance,
        fov_margin=args.fov_margin,
        body_threshold=args.body_threshold,
        autopet_clip=tuple(args.autopet_clip),
        no_autopet_clip=args.no_autopet_clip,
        autopet_crop_size=args.autopet_crop_size,
        skip_processed=not args.no_skip_processed,
        itk_threads=args.itk_threads,
    )


def run_dataset(args: argparse.Namespace, dataset_name: str) -> list[CaseResult]:
    input_dataset_root = args.extra_root / dataset_name
    output_root = args.output_root if args.output_root is not None else args.extra_root
    output_dataset_root = output_root / dataset_name

    cases = find_pair_cases(input_dataset_root)
    if args.limit is not None:
        cases = cases[: args.limit]

    in_place = same_path(input_dataset_root, output_dataset_root)
    workers = resolve_workers(args.workers)
    config = build_config(args, dataset_name)
    print(
        f"{dataset_name}: cases={len(cases)}, "
        f"mode={'in-place' if in_place else 'copy'}, "
        f"output={output_dataset_root}, "
        f"workers={workers}, itk_threads={config.itk_threads}, "
        f"skip_processed={config.skip_processed}"
    )

    rows = []
    results = []
    jobs = [
        (index, case, output_dataset_root / case.case_id)
        for index, case in enumerate(cases, start=1)
    ]

    if workers == 1:
        set_simpleitk_threads(config.itk_threads)
        for index, case, output_case_dir in jobs:
            print(f"[{index}/{len(cases)}] {case.case_id}", flush=True)
            try:
                result, row = process_case(case, output_case_dir, config)
            except Exception as exc:
                result = CaseResult(case.case_id, "failed", output_case_dir, str(exc))
                row = manifest_row(case, output_case_dir, result)
                row["dataset"] = dataset_name
                print(f"[failed] {case.case_id}: {exc}", flush=True)

            results.append(result)
            rows.append(row)
            if result.message:
                print(f"  {result.status}: {result.message}", flush=True)
    else:
        completed: list[tuple[int, CaseResult, dict[str, str]]] = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_job = {
                executor.submit(
                    process_case_job,
                    index,
                    case,
                    output_case_dir,
                    config,
                ): (index, case, output_case_dir)
                for index, case, output_case_dir in jobs
            }

            for future in as_completed(future_to_job):
                index, case, output_case_dir = future_to_job[future]
                try:
                    _, result, row = future.result()
                except Exception as exc:
                    result = CaseResult(case.case_id, "failed", output_case_dir, str(exc))
                    row = manifest_row(case, output_case_dir, result)
                    row["dataset"] = dataset_name
                    print(f"[failed] [{index}/{len(cases)}] {case.case_id}: {exc}", flush=True)
                else:
                    message = f"[{index}/{len(cases)}] {case.case_id}: {result.status}"
                    if result.message:
                        message = f"{message}: {result.message}"
                    print(message, flush=True)

                completed.append((index, result, row))

        completed.sort(key=lambda item: item[0])
        results = [result for _, result, _ in completed]
        rows = [row for _, _, row in completed]

    if not args.no_manifest and not args.dry_run:
        write_manifest(output_dataset_root / "postprocess_manifest.csv", rows)

    print_summary(results)
    return results


def main() -> None:
    args = build_arg_parser().parse_args()
    for dataset_name in args.datasets:
        print(f"\n=== {dataset_name} ===")
        run_dataset(args, dataset_name)


if __name__ == "__main__":
    main()
