from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import SimpleITK as sitk

try:
    from .common import (
        CaseResult,
        DEFAULT_BICMAC_ROOT,
        DEFAULT_EXTRA_ROOT,
        DEFAULT_ORIENTATION,
        build_reference_from_path,
        load_oriented_resampled,
        outputs_exist,
        parse_spacing,
        print_summary,
        resolve_target_spacing,
        resample_to_reference,
        sanitize_case_name,
        write_image,
        write_manifest,
    )
except ImportError:
    from common import (
        CaseResult,
        DEFAULT_BICMAC_ROOT,
        DEFAULT_EXTRA_ROOT,
        DEFAULT_ORIENTATION,
        build_reference_from_path,
        load_oriented_resampled,
        outputs_exist,
        parse_spacing,
        print_summary,
        resolve_target_spacing,
        resample_to_reference,
        sanitize_case_name,
        write_image,
        write_manifest,
    )


DEFAULT_INPUT_ROOT = None  # required: --input-root
DEFAULT_OUTPUT_ROOT = DEFAULT_EXTRA_ROOT / "quadra_hc"


@dataclass(frozen=True)
class QuadraCase:
    case_id: str
    subject_id: str
    session: str
    ct_path: Path
    pet_path: Path


def normalized_case_id(subject_name: str, session_name: str) -> str:
    match = re.search(r"QUADRA_HC_(\d+)", subject_name, flags=re.IGNORECASE)
    if match:
        base = f"sub{int(match.group(1)):03d}"
    else:
        base = sanitize_case_name(subject_name).lower()
    return f"{base}_{session_name.lower()}"


def find_quadra_cases(input_root: Path) -> list[QuadraCase]:
    imaging_root = input_root / "Imaging Data"
    if not imaging_root.exists():
        imaging_root = input_root

    cases = []
    for subject_dir in sorted(p for p in imaging_root.iterdir() if p.is_dir()):
        for session_name in ("Test", "Retest"):
            session_dir = subject_dir / session_name
            if not session_dir.exists():
                continue

            ct_files = sorted(session_dir.glob("*CT-AC.nii.gz"))
            pet_files = sorted(session_dir.glob("*PT-SUV.nii.gz"))
            if len(ct_files) != 1 or len(pet_files) != 1:
                print(
                    f"[missing] {subject_dir.name}/{session_name}: "
                    f"ct={len(ct_files)}, pet={len(pet_files)}"
                )
                continue

            case_id = normalized_case_id(subject_dir.name, session_name)
            cases.append(
                QuadraCase(
                    case_id=case_id,
                    subject_id=subject_dir.name,
                    session=session_name,
                    ct_path=ct_files[0],
                    pet_path=pet_files[0],
                )
            )

    return cases


def process_case(
    case: QuadraCase,
    output_root: Path,
    target_spacing: tuple[float, float, float],
    orientation: str,
    overwrite: bool,
    dry_run: bool,
) -> CaseResult:
    out_dir = output_root / sanitize_case_name(case.case_id)
    ct_out = out_dir / "ct.nii.gz"
    pet_out = out_dir / "pet.nii.gz"

    if dry_run:
        return CaseResult(case.case_id, "dry_run", out_dir)

    if not overwrite and outputs_exist([ct_out, pet_out]):
        return CaseResult(case.case_id, "skipped", out_dir, "outputs exist")

    ct_image, reference = build_reference_from_path(
        case.ct_path,
        target_spacing,
        orientation,
        sitk.sitkFloat32,
    )
    ct_resampled = resample_to_reference(
        ct_image,
        reference,
        sitk.sitkLinear,
        -1024.0,
        sitk.sitkFloat32,
    )
    pet_resampled = load_oriented_resampled(
        case.pet_path,
        reference,
        orientation,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )

    write_image(ct_resampled, ct_out)
    write_image(pet_resampled, pet_out)
    return CaseResult(case.case_id, "written", out_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare QUADRA_HC Test/Retest CT/PET pairs for BIC-MAC extra_data."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bicmac-root", type=Path, default=DEFAULT_BICMAC_ROOT)
    parser.add_argument(
        "--spacing",
        type=parse_spacing,
        default=None,
        help="Target spacing as x,y,z. Defaults to BIC-MAC CT median spacing.",
    )
    parser.add_argument(
        "--orientation",
        default=DEFAULT_ORIENTATION,
        help="Target orientation code, for example RAS, LPS, or LAS.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-manifest", action="store_true")
    return parser


def run(args: argparse.Namespace) -> list[CaseResult]:
    target_spacing = resolve_target_spacing(args.spacing, args.bicmac_root)
    cases = find_quadra_cases(args.input_root)
    if args.limit is not None:
        cases = cases[: args.limit]

    print(f"QUADRA_HC cases: {len(cases)}")
    print(f"Target spacing: {target_spacing}; orientation: {args.orientation.upper()}")

    results = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.case_id}", flush=True)
        try:
            result = process_case(
                case,
                args.output_root,
                target_spacing,
                args.orientation.upper(),
                args.overwrite,
                args.dry_run,
            )
        except Exception as exc:
            out_dir = args.output_root / sanitize_case_name(case.case_id)
            result = CaseResult(case.case_id, "failed", out_dir, str(exc))
            print(f"[failed] {case.case_id}: {exc}", flush=True)
        results.append(result)

    if not args.no_manifest and not args.dry_run:
        write_manifest(
            args.output_root / "manifest.csv",
            [
                {
                    "case_id": case.case_id,
                    "subject_id": case.subject_id,
                    "session": case.session,
                    "output_dir": str(args.output_root / sanitize_case_name(case.case_id)),
                    "ct_source": str(case.ct_path),
                    "pet_source": str(case.pet_path),
                    "status": result.status,
                    "message": result.message,
                }
                for case, result in zip(cases, results)
            ],
        )

    print_summary(results)
    return results


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
