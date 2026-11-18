#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


def dataset_root(path):
    path = Path(path).resolve()
    for candidate in (path / "bicmac_data", path / "bic-mac-data", path):
        if (candidate / "validation").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find the BIC-MAC validation split below {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Cache the eight AC-PET labels on the NAC-PET grid."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    root = dataset_root(args.data_root)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    subjects = sorted((root / "validation").glob("sub-*"))
    if len(subjects) != 8:
        raise RuntimeError(f"Expected 8 labeled validation cases, found {len(subjects)}")

    for subject in subjects:
        output = output_dir / f"{subject.name}.nii"
        if output.is_file():
            continue

        source = nib.load(subject / "pet-label" / "pet.nii.gz")
        target = nib.load(subject / "features" / "nacpet.nii.gz")
        aligned = resample_from_to(
            source,
            (target.shape, target.affine),
            order=1,
            mode="constant",
            cval=0.0,
        )
        data = np.nan_to_num(
            aligned.get_fdata(dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        header = target.header.copy()
        header.set_data_dtype(np.float32)
        temporary = output.with_suffix(".tmp.nii")
        nib.save(nib.Nifti1Image(data, target.affine, header), temporary)
        os.replace(temporary, output)
        print(f"aligned AC-PET: {subject.name} -> {output}")


if __name__ == "__main__":
    main()
