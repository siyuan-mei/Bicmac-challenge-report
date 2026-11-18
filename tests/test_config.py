import unittest

from omegaconf import OmegaConf

import _helpers  # noqa: F401
from bicmac import config as configs

RUNTIME_KEYS = {
    "data_cfg.data_root", "data_cfg.cv_fold", "data_cfg.use_extra_dataset",
    "train_cfg.devices", "train_cfg.world_size", "optim_cfg.base_lr",
}


def flat(node, prefix=""):
    out = {}
    for key, value in node.items():
        if isinstance(value, dict):
            out.update(flat(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


class ConfigTests(unittest.TestCase):
    def test_stage2_recipe_matches_recorded_final_run(self):
        merged = flat(OmegaConf.to_container(
            configs.load_and_merge_config(["stages/finetune/spade"])))
        record = flat(OmegaConf.to_container(
            OmegaConf.load(configs.CONFIG_DIR / "final" / "stage2_conditioning.yaml")))
        differences = {
            key for key in set(merged) | set(record)
            if merged.get(key) != record.get(key)
        }
        self.assertEqual(differences - RUNTIME_KEYS, set())
        # The recorded LR is sqrt(4)-scaled from the fragment LR.
        self.assertAlmostEqual(record["optim_cfg.base_lr"], 2 * merged["optim_cfg.base_lr"])

    def test_stage1_recipe(self):
        cfg = configs.load_and_merge_config(["stages/pretrain/mednext_l-extra"])
        self.assertIsNone(cfg.optim_cfg.scheduler)
        self.assertEqual(dict(cfg.train_cfg.loss_weights), {"hu_mae": 1.0, "mu_mae_masked": 1.0})
        self.assertEqual(cfg.train_cfg.num_epochs, 600)
        self.assertEqual(list(cfg.data_cfg.input_keys), ["pet"])

    def test_stage3_recipe(self):
        cfg = configs.load_and_merge_config(["stages/distillation/acpet"])
        self.assertEqual(cfg.model_cfg.distillation.weight, 0.05)
        self.assertEqual(cfg.train_cfg.num_epochs, 50)
        self.assertEqual(list(cfg.data_cfg.input_keys), ["pet", "mr_in", "mr_out", "topogram"])


if __name__ == "__main__":
    unittest.main()
