"""Shared helpers for the synthetic-data test-suite (no real data required)."""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# LAS affine with BIC-MAC voxel spacing.
AFFINE = np.diag([-1.5234375, 1.5234375, 2.0, 1.0])
AFFINE[:3, 3] = (100.0, -200.0, -500.0)


def small_net_overrides():
    """A tiny MedNeXtv3 with the same topology as the challenge model."""
    return {
        "features": [8, 16, 32, 64, 128],
        "depths": [1, 1, 1, 1, 1],
        "decoder_depths": [1, 1, 1, 1],
        "expansion": [2, 2, 2, 2, 2],
        "condition_channels": [8, 16, 32, 64],
    }


def write_subject(features_dir, shape=(48, 40, 64), seed=0):
    """Write a synthetic BIC-MAC subject (challenge file names, LAS affine)."""
    rng = np.random.default_rng(seed)
    features_dir = Path(features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)
    x, y, z = shape
    pet = np.zeros(shape, np.float32)
    pet[8:x - 8, 6:y - 6, 6:z - 6] = rng.uniform(5.0, 4000.0, (x - 16, y - 12, z - 12))
    mr_in = rng.integers(-5, 900, shape).astype(np.int16)
    mr_out = rng.integers(-5, 700, shape).astype(np.int16)
    topogram = rng.integers(-1024, 1500, (x, 1, z)).astype(np.int16)
    for name, data in (
        ("nacpet.nii.gz", pet),
        ("mri_combined_in_phase.nii.gz", mr_in),
        ("mri_combined_out_phase.nii.gz", mr_out),
        ("topogram.nii.gz", topogram),
    ):
        nib.save(nib.Nifti1Image(data, AFFINE), str(features_dir / name))
    return features_dir
