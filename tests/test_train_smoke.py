"""CPU smoke test of the training entry point on a tiny synthetic dataset.

Runs one Stage-2 epoch (conditioning, PET-projection loss) and one Stage-3
epoch (AC-PET EMA distillation) initialised from the Stage-2 checkpoint.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

import _helpers

SHAPE = (40, 40, 48)


def write_labelled_subject(subject_dir, seed):
    rng = np.random.default_rng(seed)
    features = _helpers.write_subject(subject_dir / "features", shape=SHAPE, seed=seed)
    (features / "metadata.json").write_text(json.dumps({"sex": "U"}))
    labels = subject_dir / "ct-label"
    labels.mkdir(parents=True, exist_ok=True)
    ct = np.full(SHAPE, -1000, np.int16)
    ct[6:34, 6:34, 4:44] = rng.integers(-200, 1200, (28, 28, 40))
    body = np.zeros(SHAPE, np.uint8)
    body[6:34, 6:34, 4:30] = 1
    body[6:34, 6:34, 30:44] = 2
    organ = (body > 0).astype(np.uint8)
    for name, data in (("ct", ct), ("body_seg", body), ("organ_seg", organ),
                       ("prediction_mask", (body > 0).astype(np.uint8))):
        nib.save(nib.Nifti1Image(data, _helpers.AFFINE), str(labels / f"{name}.nii.gz"))
    pet_label = subject_dir / "pet-label"
    pet_label.mkdir(exist_ok=True)
    acpet = nib.load(str(features / "nacpet.nii.gz")).get_fdata().astype(np.float32) * 3.0
    nib.save(nib.Nifti1Image(acpet, _helpers.AFFINE), str(pet_label / "pet.nii.gz"))


def common_overrides():
    net = _helpers.small_net_overrides()
    values = [
        "crop_size=[32,32,32]",
        "train_cfg.accelerator=cpu",
        "train_cfg.devices=1",
        "train_cfg.precision=32-true",
        "train_cfg.use_wandb=false",
        "train_cfg.num_workers=0",
        "train_cfg.num_epochs=1",
        "train_cfg.save_freq=1",
        "train_cfg.save_visuals=false",
        "train_cfg.channels_last=false",
        "val_cfg.val_freq=1",
        "optim_cfg.warmup_epochs=0",
    ]
    for key, value in net.items():
        values.append(f"model_cfg.net.{key}=[{','.join(map(str, value))}]")
    args = []
    for value in values:
        args += ["--set", value]
    return args


class TrainingSmokeTest(unittest.TestCase):
    def run_train(self, args, env):
        result = subprocess.run(
            [sys.executable, str(_helpers.REPO_ROOT / "scripts" / "train.py"), *args],
            capture_output=True, text=True, env=env,
        )
        self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-3000:])
        return result

    def test_stage2_then_stage3(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = tmp / "data"
            for split, ids in (("train", (1, 2)), ("validation", (3,))):
                for index in ids:
                    write_labelled_subject(data / split / f"sub-{index:03d}", seed=index)
            env = dict(os.environ, OMP_NUM_THREADS="4")

            s2 = tmp / "runs" / "s2"
            self.run_train([
                "--config", "stages/finetune/spade", "--data-root", str(data),
                "--workdir", str(s2), "--run-name", "smoke_s2",
                "--set", "train_cfg.require_finetune_checkpoint=false",
                *common_overrides(),
            ], env)
            self.assertTrue((s2 / "TRAINING_COMPLETE").is_file())
            self.assertTrue((s2 / "checkpoints" / "last.ckpt").is_file())
            self.assertTrue(list((s2 / "checkpoints").glob("best*.ckpt")))

            s3 = tmp / "runs" / "s3"
            self.run_train([
                "--config", "stages/distillation/acpet", "--data-root", str(data),
                "--workdir", str(s3), "--run-name", "smoke_s3",
                "--finetune-from", str(s2 / "checkpoints" / "last.ckpt"),
                "--set", "data_cfg.val_cv_fold=null",
                *common_overrides(),
            ], env)
            self.assertTrue((s3 / "TRAINING_COMPLETE").is_file())


    def test_stage1_mixed_extra_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for index in (1, 2):
                write_labelled_subject(tmp / "bicmac_data" / "train" / f"sub-{index:03d}", index)
            write_labelled_subject(tmp / "bicmac_data" / "validation" / "sub-003", 3)
            rng = np.random.default_rng(7)
            for dataset, case in (("autopet", "fdg_case0"), ("autopet", "fdg_case1"),
                                  ("quadra_hc", "sub001_test")):
                case_dir = tmp / "extra_data" / dataset / case
                case_dir.mkdir(parents=True)
                ct = np.full(SHAPE, -1000, np.float32)
                ct[6:34, 6:34, 4:44] = rng.uniform(-200, 1200, (28, 28, 40))
                pet = np.where(ct > -500, rng.uniform(0, 20000, SHAPE), 0).astype(np.float32)
                for name, data in (("ct", ct), ("pet", pet),
                                   ("body_mask", (ct > -500).astype(np.uint8))):
                    nib.save(nib.Nifti1Image(data, _helpers.AFFINE), str(case_dir / f"{name}.nii.gz"))
            run = tmp / "runs" / "s1"
            self.run_train([
                "--config", "stages/pretrain/mednext_l-extra", "--data-root", str(tmp),
                "--workdir", str(run), "--run-name", "smoke_s1",
                "--set", "train_cfg.batch_size=2",
                *common_overrides(),
            ], dict(os.environ, OMP_NUM_THREADS="4"))
            self.assertTrue((run / "TRAINING_COMPLETE").is_file())


if __name__ == "__main__":
    unittest.main()
