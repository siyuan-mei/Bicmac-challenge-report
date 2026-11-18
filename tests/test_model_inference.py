import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from omegaconf import OmegaConf

import _helpers
from bicmac import config as configs
from bicmac.inference import build_model, prepare_predictor, predict_ct
from bicmac.models.background_aware_inferer import (
    BackgroundAwareSlidingWindowInferer,
    BackgroundContext,
)
from bicmac.models.register import build_net
import bicmac.models  # noqa: F401  (registers MedNeXtv3)

FINAL_S2 = configs.CONFIG_DIR / "final" / "stage2_conditioning.yaml"


def tiny_config(path):
    cfg = OmegaConf.load(FINAL_S2)
    cfg.model_cfg.net.update(_helpers.small_net_overrides())
    cfg.crop_size = [32, 32, 32]
    OmegaConf.save(cfg, path)
    return path


def tiny_checkpoint(config_path, path, seed):
    torch.manual_seed(seed)
    model = build_model(OmegaConf.load(config_path), _random_state(config_path, seed), "cpu")
    torch.save({"model": model.state_dict()}, path)
    return path


def _random_state(config_path, seed):
    from bicmac.models.encoder_decoder import EncoderDecoder

    cfg = OmegaConf.load(config_path)
    cfg.train_cfg.loss_weights = {"hu_mae": 1.0}
    torch.manual_seed(seed)
    model = EncoderDecoder(cfg)
    with torch.no_grad():  # make zero-initialised injector projections non-trivial
        for name, parameter in model.named_parameters():
            if ".injector." in name:
                parameter.add_(0.01 * torch.randn_like(parameter))
    state_path = Path(tempfile.mkdtemp()) / f"state_{seed}.ckpt"
    torch.save({"model": model.state_dict()}, state_path)
    return state_path


class ArchitectureTests(unittest.TestCase):
    def test_challenge_model_size(self):
        cfg = OmegaConf.load(FINAL_S2)
        net_cfg = OmegaConf.to_container(cfg.model_cfg.net)
        net_cfg.update(in_channels=4, out_channels=1, input_keys=list(cfg.data_cfg.input_keys))
        net = build_net(net_cfg)
        params = sum(p.numel() for p in net.parameters()) / 1e6
        self.assertAlmostEqual(params, 26.71, delta=0.01)  # value in the report

    def test_zero_initialised_injector_preserves_pet_prior(self):
        base = dict(_helpers.small_net_overrides(), type="MedNeXtv3", skip_fusion="cat",
                    layer_scale_init=None, kernel_sizes=[3] * 5, out_channels=1)
        torch.manual_seed(0)
        pet_net = build_net(dict(base, in_channels=1, input_fusion="concat")).eval()
        cond_net = build_net(dict(base, in_channels=4, input_fusion="injector",
                                  input_keys=["pet", "mr_in", "mr_out", "topogram"])).eval()
        result = cond_net.load_state_dict(pet_net.state_dict(), strict=False)
        self.assertEqual(result.unexpected_keys, [])
        self.assertTrue(all(key.startswith("injector.") for key in result.missing_keys))
        x = torch.rand(1, 4, 32, 32, 32)
        with torch.no_grad():
            torch.testing.assert_close(cond_net(x), pet_net(x[:, :1]), atol=1e-5, rtol=1e-5)


class SegmentedInfererTests(unittest.TestCase):
    def test_identity_network_is_reconstructed_inside_segments(self):
        shape = (64, 64, 96)
        pet = np.zeros(shape, np.float32)
        pet[16:48, 20:44, 8:88] = 100.0
        context = BackgroundContext.from_pet(pet, roi_size=(32, 32, 32), num_segments=3)
        self.assertEqual(len(context.segments), 3)
        inferer = BackgroundAwareSlidingWindowInferer(
            roi_size=(32, 32, 32), overlap=0.5, mode="gaussian", sw_device="cpu", device="cpu")
        volume = torch.rand(1, 1, *shape)
        # MONAI blends windows in place; a real network returns a new tensor.
        out = inferer(volume, lambda x: x.clone(), context)
        covered = torch.zeros(shape, dtype=torch.bool)
        for segment in context.segments:
            covered[tuple(slice(a, b) for a, b in segment.output_box)] = True
        torch.testing.assert_close(out[0, 0][covered], volume[0, 0][covered], atol=1e-5, rtol=1e-5)
        self.assertTrue(torch.all(out[0, 0][~covered] == 0.0))
        self.assertTrue(covered[16:48, 20:44, 8:88].all())


class InferenceTests(unittest.TestCase):
    def test_checkpoint_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = tiny_config(Path(tmp) / "cfg.yaml")
            state = torch.load(_random_state(config, 0), weights_only=True)["model"]
            state.pop(next(iter(state)))
            bad = Path(tmp) / "bad.ckpt"
            torch.save({"model": state}, bad)
            with self.assertRaises(RuntimeError):
                build_model(OmegaConf.load(config), bad, "cpu")

    def test_end_to_end_single_and_ensemble(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            config = tiny_config(tmp / "cfg.yaml")
            ckpts = [_random_state(config, seed) for seed in (1, 2)]
            features = _helpers.write_subject(tmp / "in" / "sub-001" / "features", shape=(48, 40, 64))
            outputs = []
            for name, selection in (("a", ckpts[:1]), ("b", ckpts[1:]), ("ab", ckpts)):
                predictor = prepare_predictor(config, selection, "cpu")
                out = predict_ct(predictor, features, tmp / name / "ct.nii.gz", progress=False)
                image = nib.load(str(out))
                self.assertEqual(image.shape, (48, 40, 64))
                self.assertEqual(image.get_data_dtype(), np.float32)
                np.testing.assert_allclose(image.affine, _helpers.AFFINE)
                np.testing.assert_allclose(image.header.get_zooms(), (1.5234375, 1.5234375, 2.0))
                data = image.get_fdata(dtype=np.float32)
                self.assertGreaterEqual(data.min(), -1024.0)
                self.assertLessEqual(data.max(), 3071.0)
                outputs.append(data)
            a, b, ab = outputs
            unclipped = (np.abs(a) < 1000) & (np.abs(b) < 1000)
            np.testing.assert_allclose(ab[unclipped], 0.5 * (a + b)[unclipped], atol=0.5)

    def test_cli(self):
        scripts = _helpers.REPO_ROOT / "scripts"
        for script in ("predict.py", "train.py", "fuse_checkpoints.py", "prepare_acpet.py"):
            result = subprocess.run([sys.executable, str(scripts / script), "--help"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            config = tiny_config(tmp / "cfg.yaml")
            ckpt = _random_state(config, 3)
            _helpers.write_subject(tmp / "in" / "sub-007" / "features")
            result = subprocess.run(
                [sys.executable, str(scripts / "predict.py"), "--input_dir", str(tmp / "in"),
                 "--output_dir", str(tmp / "out"), "--config", str(config),
                 "--checkpoints", str(ckpt), "--device", "cpu", "--no_progress"],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])
            self.assertTrue((tmp / "out" / "sub-007" / "ct.nii.gz").is_file())


if __name__ == "__main__":
    unittest.main()
