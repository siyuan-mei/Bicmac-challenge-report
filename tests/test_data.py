import tempfile
import unittest
from pathlib import Path

import numpy as np

import _helpers
from bicmac.data.dataset.bicmac import cross_validation_subjects
from bicmac.data.transforms import build_val_pipeline
from bicmac.inference import load_inputs


class PreprocessingParityTests(unittest.TestCase):
    """Inference preprocessing must equal the MONAI validation pipeline."""

    def test_inference_inputs_match_validation_pipeline(self):
        keys = ("pet", "mr_in", "mr_out", "topogram")
        with tempfile.TemporaryDirectory() as tmp:
            features = _helpers.write_subject(Path(tmp) / "sub-000" / "features")
            inputs, raw_pet, reference = load_inputs(features, keys)
            sample = {
                "dataset": "bicmac",
                "pet": str(features / "nacpet.nii.gz"),
                "mr_in": str(features / "mri_combined_in_phase.nii.gz"),
                "mr_out": str(features / "mri_combined_out_phase.nii.gz"),
                "topogram": str(features / "topogram.nii.gz"),
            }
            out = build_val_pipeline(label_keys=(), input_keys=keys, mask_keys=())(sample)
            for index, key in enumerate(keys):
                expected = np.asarray(out[key][0], dtype=np.float32)
                np.testing.assert_allclose(
                    inputs[0, index].numpy(), expected, atol=1e-5, err_msg=key
                )
            self.assertEqual(tuple(inputs.shape), (1, 4, *raw_pet.shape))
            np.testing.assert_allclose(reference.affine, _helpers.AFFINE)


class CrossValidationTests(unittest.TestCase):
    def test_folds_are_deterministic_and_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split, count, offset in (("train", 59, 8), ("validation", 8, 0)):
                for index in range(count):
                    (root / split / f"sub-{index + offset:03d}").mkdir(parents=True)
            original_val = sorted(str(p) for p in (root / "validation").iterdir())
            self.assertEqual(cross_validation_subjects(str(root), 1, "validation"), original_val)
            seen = set()
            for fold in range(1, 6):
                val = set(cross_validation_subjects(str(root), fold, "validation"))
                train = set(cross_validation_subjects(str(root), fold, "train"))
                self.assertEqual(len(val), 8)
                self.assertFalse(val & train)
                self.assertEqual(len(val | train), 67)
                self.assertFalse(val & seen)
                seen |= val
                self.assertEqual(val, set(cross_validation_subjects(str(root), fold, "validation")))


if __name__ == "__main__":
    unittest.main()
