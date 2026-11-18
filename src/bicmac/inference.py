"""CT inference for the BIC-MAC challenge models.

This module reproduces the inference path of the challenge code:

* inputs are read with nibabel on their native voxel grid (no resampling and
  no reorientation; all BIC-MAC inputs share the NAC-PET grid),
* per-modality raw thresholds followed by per-volume 1st-98th percentile
  normalisation to [0, 1] (identical to ``build_val_pipeline``),
* the topogram (X, 1, Z) is broadcast along Y to the PET volume,
* PET-segmented Gaussian sliding-window inference
  (``BackgroundAwareSlidingWindowInferer``; the same path that is used for
  validation inside ``EncoderDecoder``),
* optional multi-checkpoint ensembles averaged online in FP32 inside every
  sliding window,
* inverse CT scaling ``HU = 3000 * y - 1000`` and clipping to [-1024, 3071],
* output written as float32 NIfTI with the affine and header of the NAC-PET.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import SlidingWindowInferer
from omegaconf import OmegaConf
from torch import nn

from .models.background_aware_inferer import (
    BackgroundAwareSlidingWindowInferer,
    BackgroundContext,
)
from .models.encoder_decoder import EncoderDecoder


# Input file names inside a subject's ``features`` directory (challenge layout).
INPUT_FILES = {
    "pet": "nacpet.nii.gz",
    "nacpet": "nacpet.nii.gz",
    "mr_in": "mri_combined_in_phase.nii.gz",
    "mri_in": "mri_combined_in_phase.nii.gz",
    "mr_out": "mri_combined_out_phase.nii.gz",
    "mri_out": "mri_combined_out_phase.nii.gz",
    "topogram": "topogram.nii.gz",
}

# CT intensity convention used during training (ScaleIntensityRanged
# a_min=-1000, a_max=2000 -> [0, 1]).
CT_SCALE = 3000.0
CT_OFFSET = -1000.0
HU_CLIP = (-1024.0, 3071.0)


class MeanEnsemble(nn.Module):
    """Sequential FP32 averaging; activation memory is independent of fold count."""

    def __init__(self, models: Sequence[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, inputs):
        mean = None
        for index, model in enumerate(self.models, start=1):
            prediction = model(inputs)
            if isinstance(prediction, (tuple, list)):
                prediction = prediction[0]
            prediction = prediction.float()
            mean = prediction if mean is None else mean + (prediction - mean) / index
        return mean


@dataclass
class Predictor:
    model: nn.Module
    config: object
    checkpoints: tuple
    device: torch.device


def load_config(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    config = OmegaConf.load(path)
    config.pop("includes", None)
    for key in ("model_cfg", "data_cfg", "val_cfg"):
        if key not in config:
            raise KeyError(f"Config {path} does not contain {key}.")
    if "net" not in config.model_cfg or "input_keys" not in config.data_cfg:
        raise KeyError("Config requires model_cfg.net and data_cfg.input_keys.")
    if "crop_size" in config and "patch_size" not in config.val_cfg:
        config.val_cfg.patch_size = list(config.crop_size)
    return config


def checkpoint_state(path):
    """Return the model state dict of a training or collected checkpoint."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # Full training checkpoints (last.ckpt) also store optimizer and
        # scheduler state that may not be loadable with weights_only=True.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint must be a mapping: {path}")
    for key in ("state_dict", "model"):
        if isinstance(checkpoint.get(key), Mapping):
            return dict(checkpoint[key])
    if any(torch.is_tensor(value) for value in checkpoint.values()):
        return dict(checkpoint)
    raise TypeError(f"No model state_dict found in {path}")


def _strip_outer_prefixes(state):
    prefixes = ("_forward_module.", "module.", "model.")
    result = {}
    for key, value in state.items():
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
                    break
        result[key] = value
    return result


def _align_state(state, model):
    state = _strip_outer_prefixes(state)
    candidates = [
        state,
        {key.removeprefix("net."): value for key, value in state.items()},
        {f"net.{key}": value for key, value in state.items()},
    ]
    expected = set(model.state_dict())
    return max(candidates, key=lambda candidate: len(expected.intersection(candidate)))


def build_model(config, checkpoint, device):
    """Instantiate ``EncoderDecoder`` from ``config`` and load ``checkpoint``.

    Every ``net.*`` tensor must match; a mismatch raises instead of silently
    running a partially initialised network.
    """
    runtime_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    runtime_config.pop("includes", None)
    runtime_config.setdefault("train_cfg", {})
    # Losses are irrelevant at inference; use a cheap, dependency-free one.
    runtime_config.train_cfg.loss_weights = {"hu_mae": 1.0}
    runtime_config.train_cfg.loss_options = {}
    runtime_config.train_cfg.compile = False
    runtime_config.model_cfg.pop("distillation", None)
    model = EncoderDecoder(runtime_config)

    state = _align_state(checkpoint_state(checkpoint), model)
    incompatible = model.load_state_dict(state, strict=False)
    missing = [k for k in incompatible.missing_keys if k.startswith("net.")]
    unexpected = [k for k in incompatible.unexpected_keys if k.startswith("net.")]
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/model mismatch for {checkpoint}: "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}"
        )
    return model.to(device).eval()


def prepare_predictor(config_path, checkpoint_paths, device=None):
    config = load_config(config_path)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoints = [Path(path).resolve() for path in checkpoint_paths]
    if not checkpoints:
        raise ValueError("At least one checkpoint is required.")
    models = [build_model(config, checkpoint, device) for checkpoint in checkpoints]
    model = models[0] if len(models) == 1 else MeanEnsemble(models).to(device).eval()
    print(f"model: {len(models)} checkpoint(s); device={device}", flush=True)
    print("checkpoints:", *(str(path) for path in checkpoints), sep="\n  ", flush=True)
    return Predictor(model, config, tuple(checkpoints), device)


# ---------------------------------------------------------------------------
# Input preprocessing
# ---------------------------------------------------------------------------


def _load_image(path):
    image = nib.load(str(path))
    data = np.nan_to_num(
        np.asanyarray(image.dataobj, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return data, image


def normalize_input(key, data):
    """Raw threshold + per-volume 1st-98th percentile scaling to [0, 1].

    Matches ``InputThresholdD(pet_floor=10, mr_floor=0, topogram_air=-50 ->
    -1024)`` followed by ``ScaleIntensityRangePercentilesd(1, 98, clip=True)``.
    """
    if key in {"pet", "nacpet"}:
        data = np.where(data <= 10.0, 0.0, data)
    elif key in {"mr_in", "mri_in", "mr_out", "mri_out"}:
        data = np.where(data <= 0.0, 0.0, data)
    elif key == "topogram":
        data = np.where(data <= -50.0, -1024.0, data)
    lower, upper = np.percentile(data, (1, 98))
    if not np.isfinite(lower + upper) or upper <= lower:
        return np.zeros_like(data, dtype=np.float32)
    return np.clip((data - lower) / (upper - lower), 0.0, 1.0).astype(np.float32)


def expand_topogram(data, shape):
    """Broadcast an (X, Z) or (X, 1, Z) topogram along Y to ``shape``."""
    if data.ndim == 2:
        data = data[:, None, :]
    if data.ndim != 3 or data.shape[0] != shape[0] or data.shape[2] != shape[2]:
        raise ValueError(f"Topogram shape {data.shape} does not match PET {shape}.")
    if data.shape[1] == 1:
        return np.broadcast_to(data, shape)
    if data.shape != shape:
        raise ValueError(f"Topogram shape {data.shape} does not match PET {shape}.")
    return data


def load_inputs(features_dir, input_keys):
    """Return (model input [1, C, X, Y, Z], raw NAC-PET, NAC-PET nibabel image)."""
    input_keys = tuple(str(key) for key in input_keys)
    unknown = set(input_keys) - set(INPUT_FILES)
    if unknown:
        raise ValueError(f"Unsupported inputs: {sorted(unknown)}")
    pet_key = next((key for key in ("pet", "nacpet") if key in input_keys), None)
    if pet_key is None:
        raise ValueError("Inference requires the NAC-PET input.")
    features_dir = Path(features_dir)
    pet, reference = _load_image(features_dir / INPUT_FILES[pet_key])
    if pet.ndim != 3:
        raise ValueError(f"NAC-PET must be 3D, got {pet.shape}.")

    channels = []
    for key in input_keys:
        if key in {"pet", "nacpet"}:
            data = pet
        else:
            path = features_dir / INPUT_FILES[key]
            if not path.is_file():
                raise FileNotFoundError(path)
            data = _load_image(path)[0]
        data = normalize_input(key, data)
        if key == "topogram":
            data = expand_topogram(data, pet.shape)
        elif data.shape != pet.shape:
            raise ValueError(f"{key} shape {data.shape} does not match PET {pet.shape}.")
        channels.append(data)
    inputs = torch.from_numpy(np.stack(channels)).unsqueeze(0)
    return inputs, pet, reference


# ---------------------------------------------------------------------------
# Sliding-window inference
# ---------------------------------------------------------------------------


def inference_settings(config, overlap=None, sw_batch_size=None, segmented=None):
    val_cfg = OmegaConf.create(OmegaConf.to_container(config.val_cfg, resolve=True))
    if overlap is not None:
        val_cfg.overlap_ratio = float(overlap)
    if sw_batch_size is not None:
        val_cfg.sw_batch_size = int(sw_batch_size)
    options = EncoderDecoder._build_inference_options(val_cfg)
    segment_settings = EncoderDecoder._build_segmented_settings(val_cfg)
    if segmented is not None:
        segment_settings["enabled"] = bool(segmented)
    return options, segment_settings


def predict_normalized(predictor, inputs, raw_pet, options, segment_settings, progress=True):
    device = predictor.device
    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    with torch.inference_mode(), torch.amp.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=device.type == "cuda",
    ):
        if not segment_settings["enabled"]:
            inferer = SlidingWindowInferer(
                **options,
                sw_device=device,
                device=torch.device("cpu"),
                progress=progress,
            )
            return inferer(inputs, predictor.model)

        context_options = dict(segment_settings)
        context_options.pop("enabled", None)
        context = BackgroundContext.from_pet(
            raw_pet,
            roi_size=options["roi_size"],
            **context_options,
        )
        inferer = BackgroundAwareSlidingWindowInferer(
            **options,
            sw_device=device,
            device=torch.device("cpu"),
            progress=progress,
        )
        return inferer(inputs, predictor.model, context)


def to_hu(normalized):
    prediction = normalized[0, 0].float().cpu().numpy() * CT_SCALE + CT_OFFSET
    return np.clip(prediction, *HU_CLIP).astype(np.float32)


def save_like(volume, reference, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(volume, reference.affine, header), str(output_path))
    return output_path


def predict_ct(
    predictor,
    features_dir,
    output_ct,
    overlap=None,
    sw_batch_size=None,
    segmented=None,
    progress=True,
):
    options, segment_settings = inference_settings(
        predictor.config, overlap, sw_batch_size, segmented
    )
    print(f"inference: {options}; segmented={segment_settings['enabled']}", flush=True)
    inputs, raw_pet, reference = load_inputs(
        features_dir, predictor.config.data_cfg.input_keys
    )
    normalized = predict_normalized(
        predictor, inputs, raw_pet, options, segment_settings, progress
    )
    output_ct = save_like(to_hu(normalized), reference, output_ct)
    print(f"saved: {output_ct}", flush=True)
    return output_ct
