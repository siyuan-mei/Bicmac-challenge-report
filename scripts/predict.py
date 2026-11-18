#!/usr/bin/env python3
"""Synthesize CT for BIC-MAC subjects.

Input layout (challenge layout; extra files are ignored)::

    <input_dir>/sub-XXX/features/nacpet.nii.gz
                                 mri_combined_in_phase.nii.gz
                                 mri_combined_out_phase.nii.gz
                                 topogram.nii.gz

Output layout::

    <output_dir>/sub-XXX/ct.nii.gz   (float32 HU, NAC-PET affine/header)

Example (submitted S2 fold-1 + fold-4 ensemble)::

    python scripts/predict.py \
        --input_dir /path/to/challenge_val \
        --output_dir outputs/s2_f1f4 \
        --config configs/final/stage2_conditioning.yaml \
        --checkpoints /path/to/s2_fold1_best.ckpt /path/to/s2_fold4_best.ckpt
"""

import argparse
import sys
import time
from pathlib import Path

import _paths  # noqa: F401  (adds ./src to sys.path)
from bicmac.inference import predict_ct, prepare_predictor


def discover_subjects(input_dir, subjects=None):
    input_dir = Path(input_dir)
    if subjects:
        return [input_dir / subject for subject in subjects]
    found = sorted(
        path for path in input_dir.iterdir()
        if path.is_dir() and (path / "features").is_dir()
    )
    if not found:
        raise RuntimeError(f"No <subject>/features directories found in {input_dir}")
    return found


def build_parser():
    parser = argparse.ArgumentParser(
        description="BIC-MAC multimodal-to-CT inference (single model or ensemble)."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input_dir",
        help="Directory containing <subject>/features/*.nii.gz.",
    )
    source.add_argument(
        "--features_dir",
        help="A single subject's features directory (use with --output_ct).",
    )
    parser.add_argument("--output_dir", help="Output root for --input_dir mode.")
    parser.add_argument("--output_ct", help="Output file for --features_dir mode.")
    parser.add_argument("--subjects", nargs="+", help="Optional subset of subject IDs.")
    parser.add_argument(
        "--config",
        required=True,
        help="Model config, e.g. configs/final/stage2_conditioning.yaml.",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="One or more checkpoints; several are averaged in FP32 (ensemble).",
    )
    parser.add_argument("--device", default=None, help="cuda, cuda:0, cpu, ...")
    parser.add_argument(
        "--overlap",
        type=float,
        default=None,
        help="Sliding-window overlap (default: config val_cfg.overlap_ratio=0.5).",
    )
    parser.add_argument(
        "--sw_batch_size",
        type=int,
        default=None,
        help="Sliding-window batch size (default: config value).",
    )
    parser.add_argument(
        "--no_segmented",
        action="store_true",
        help="Disable PET-segmented inference and run plain whole-volume "
        "sliding-window inference (NOT the challenge setting).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.features_dir and not args.output_ct:
        raise SystemExit("--features_dir requires --output_ct")
    if args.input_dir and not args.output_dir:
        raise SystemExit("--input_dir requires --output_dir")

    predictor = prepare_predictor(args.config, args.checkpoints, args.device)
    options = dict(
        overlap=args.overlap,
        sw_batch_size=args.sw_batch_size,
        segmented=False if args.no_segmented else None,
        progress=not args.no_progress,
    )

    if args.features_dir:
        jobs = [(Path(args.features_dir), Path(args.output_ct), "subject")]
    else:
        jobs = [
            (subject / "features", Path(args.output_dir) / subject.name / "ct.nii.gz", subject.name)
            for subject in discover_subjects(args.input_dir, args.subjects)
        ]

    started = time.perf_counter()
    for features_dir, output_ct, name in jobs:
        if output_ct.exists() and not args.overwrite:
            print(f"[{name}] exists, skipping: {output_ct}", flush=True)
            continue
        subject_started = time.perf_counter()
        print(f"[{name}] CT inference", flush=True)
        predict_ct(predictor, features_dir, output_ct, **options)
        print(f"[{name}] time: {time.perf_counter() - subject_started:.2f} s", flush=True)
    print(f"total: {time.perf_counter() - started:.2f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
