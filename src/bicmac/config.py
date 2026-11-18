"""Hierarchical YAML configuration loading (``includes`` + left-to-right merge)."""
from collections.abc import Sequence
import os
import pathlib

from omegaconf import OmegaConf


# YAML fragments live in <repo>/configs; override with BICMAC_CONFIG_DIR.
CONFIG_DIR = pathlib.Path(
    os.environ.get(
        "BICMAC_CONFIG_DIR",
        pathlib.Path(__file__).resolve().parents[2] / "configs",
    )
).resolve()


def load_config(config_name: str):
    config_path = pathlib.Path(config_name)
    if config_path.suffix != ".yaml":
        config_path = config_path.with_suffix(".yaml")
    if not config_path.is_absolute() and not config_path.is_file():
        config_path = CONFIG_DIR / config_path
    return OmegaConf.load(config_path)


default_config = load_config("default_config")


def _load_with_includes(config_name: str, stack=()):
    name = str(config_name)
    if name in stack:
        chain = " -> ".join((*stack, name))
        raise ValueError(f"Circular config include: {chain}")

    config = load_config(name)
    includes = config.pop("includes", [])
    if isinstance(includes, str):
        includes = [includes]

    parents = [_load_with_includes(item, (*stack, name)) for item in includes]
    return OmegaConf.merge(*parents, config) if parents else config


def load_and_merge_config(config_names: str | Sequence[str]):
    """Merge defaults and one or more config fragments from left to right."""
    if isinstance(config_names, str):
        config_names = [config_names]

    merged = OmegaConf.create(default_config)
    for fragment in (_load_with_includes(name) for name in config_names):
        # Experiment loss maps are complete recipes, not additive overlays.
        for path in ("train_cfg.loss_weights", "train_cfg.loss_options"):
            value = OmegaConf.select(fragment, path)
            if value is not None:
                OmegaConf.update(merged, path, value, merge=False)
        merged = OmegaConf.merge(merged, fragment)
    return merged
