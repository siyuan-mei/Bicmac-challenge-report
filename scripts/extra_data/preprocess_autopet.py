from __future__ import annotations

import argparse
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
        read_header,
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
        read_header,
        resolve_target_spacing,
        resample_to_reference,
        sanitize_case_name,
        write_image,
        write_manifest,
    )


DEFAULT_INPUT_ROOT = None  # required: --input-root
DEFAULT_OUTPUT_ROOT = DEFAULT_EXTRA_ROOT / "autopet"


@dataclass(frozen=True)
class AutoPetCase:
    case_id: str
    ct_path: Path
    pet_path: Path


def output_pair_valid(ct_path: Path, pet_path: Path) -> bool:
    if not outputs_exist([ct_path, pet_path]):
        return False
    try:
        ct_reader = read_header(ct_path)
        pet_reader = read_header(pet_path)
    except Exception:
        return False

    if ct_reader.GetSize() != pet_reader.GetSize():
        return False
    if any(
        abs(a - b) > 1e-4
        for a, b in zip(ct_reader.GetSpacing(), pet_reader.GetSpacing())
    ):
        return False
    if any(
        abs(a - b) > 1e-4
        for a, b in zip(ct_reader.GetDirection(), pet_reader.GetDirection())
    ):
        return False
    if any(
        abs(a - b) > 1e-3
        for a, b in zip(ct_reader.GetOrigin(), pet_reader.GetOrigin())
    ):
        return False

    return True


def find_nnunet_cases(input_root: Path) -> list[AutoPetCase]:
    images_dir = input_root / "imagesTr"
    if not images_dir.exists():
        images_dir = input_root

    cases = []
    for ct_path in sorted(images_dir.glob("*_0000.nii.gz")):
        case_id = ct_path.name[: -len("_0000.nii.gz")]
        pet_path = images_dir / f"{case_id}_0001.nii.gz"
        if not pet_path.exists():
            print(f"[missing] PET pair for {ct_path}")
            continue
        cases.append(AutoPetCase(case_id=case_id, ct_path=ct_path, pet_path=pet_path))

    return cases


def find_fdg_tree_cases(input_root: Path) -> list[AutoPetCase]:
    roots = []
    if input_root.name == "FDG-PET-CT-Lesions":
        roots.append(input_root)
    roots.extend(sorted(input_root.glob("FDG-PET-CT-Lesions")))

    cases = []
    seen: set[Path] = set()
    for root in roots:
        if root in seen or not root.exists():
            continue
        seen.add(root)

        for ct_path in sorted(root.glob("PETCT_*/*/CTres.nii.gz")):
            study_dir = ct_path.parent
            pet_path = study_dir / "SUV.nii.gz"
            if not pet_path.exists():
                print(f"[missing] SUV pair for {ct_path}")
                continue

            subject = study_dir.parent.name
            if subject.startswith("PETCT_"):
                subject = subject[len("PETCT_") :]
            case_id = f"fdg_{subject}_{study_dir.name}"
            cases.append(AutoPetCase(case_id=case_id, ct_path=ct_path, pet_path=pet_path))

    return cases


def find_autopet_cases(input_root: Path) -> list[AutoPetCase]:
    cases = find_nnunet_cases(input_root)
    cases.extend(find_fdg_tree_cases(input_root))

    unique = {}
    for case in cases:
        if case.case_id in unique:
            print(f"[duplicate] skipping repeated AutoPET case_id {case.case_id}")
            continue
        unique[case.case_id] = case

    return list(unique.values())


def process_case(
    case: AutoPetCase,
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

    if not overwrite and output_pair_valid(ct_out, pet_out):
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
        description="Prepare AutoPET CT/PET pairs for BIC-MAC extra_data."
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
    parser.add_argument(
        "--dataset-label",
        default="AutoPET",
        help="Label used in logs, for example FDG AutoPET or PSMA AutoPET.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-manifest", action="store_true")
    return parser


def run(args: argparse.Namespace) -> list[CaseResult]:
    target_spacing = resolve_target_spacing(args.spacing, args.bicmac_root)
    cases = find_autopet_cases(args.input_root)
    if args.limit is not None:
        cases = cases[: args.limit]

    print(f"{args.dataset_label} cases: {len(cases)}")
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
