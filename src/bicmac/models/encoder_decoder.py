import math
from pathlib import Path

import lightning as L
import nibabel as nib
import numpy as np
import torch
from omegaconf import OmegaConf
from ..eval_metrics.image_metrics import evaluate
from .core_losses import CombinedLoss
from monai.inferers import SlidingWindowInferer
from .background_aware_inferer import (
    BackgroundAwareSlidingWindowInferer,
    BackgroundContext,
)
from .register import build_net


def adamw_groups(named_parameters, name, lr, lr_min, weight_decay):
    """Split one logical parameter group into decay and no-decay tensors."""
    named_parameters = list(named_parameters)
    decay = [
        parameter for parameter_name, parameter in named_parameters
        if parameter.ndim > 1 and not parameter_name.endswith(".bias")
    ]
    no_decay = [
        parameter for parameter_name, parameter in named_parameters
        if parameter.ndim <= 1 or parameter_name.endswith(".bias")
    ]
    groups = []
    for suffix, parameters, decay_value in (
        ("", decay, weight_decay),
        ("_no_decay", no_decay, 0.0),
    ):
        if parameters:
            groups.append({
                "name": f"{name}{suffix}",
                "params": parameters,
                "lr": float(lr),
                "lr_min": float(lr_min),
                "weight_decay": float(decay_value),
            })
    return groups


class EncoderDecoder(L.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.input_keys = tuple(config.data_cfg.input_keys)
        self.channels_last = bool(config.train_cfg.get("channels_last", False))

        net_cfg = OmegaConf.to_container(config.model_cfg.net, resolve=True)
        net_cfg.pop("input_channels", None)
        net_cfg.pop("n_classes", None)
        net_cfg.pop("num_classes", None)
        net_cfg["in_channels"] = len(self.input_keys)
        net_cfg["out_channels"] = 1
        if net_cfg.get("input_fusion") == "injector":
            net_cfg["input_keys"] = list(self.input_keys)
        self.net = build_net(net_cfg)
        if self.net is None:
            raise ValueError("model_cfg.net.type must name a registered model.")

        if bool(config.train_cfg.get("compile", False)):
            self.net = torch.compile(self.net)

        self.criterion = CombinedLoss(
            config.train_cfg.loss_weights,
            config.train_cfg.get("loss_options", {}),
        )

        self._current_epoch = 0

        # Keep validation/test inference numerically aligned with the production
        # submission path. The inferer itself is constructed per call because
        # sw_device follows the active Fabric/Lightning device, while the final
        # stitching buffer is kept on CPU exactly as in production.
        self.inference_options = self._build_inference_options(config.val_cfg)
        self.segmented_settings = self._build_segmented_settings(config.val_cfg)

        # Validation metrics use the same attenuation-domain parameters as the
        # smoothed-mu training loss whenever they are configured there.
        loss_options = config.train_cfg.get("loss_options", None)
        loss_options = (
            OmegaConf.to_container(loss_options, resolve=True)
            if OmegaConf.is_config(loss_options)
            else loss_options
        ) or {}
        smooth_options = {}
        for key in (
            "smoothed_mu_mae",
            "smooth_mu_mae",
            "mu_smoothed_mae",
            "recon_mu_mae",
        ):
            if key in loss_options:
                smooth_options = dict(loss_options[key] or {})
                if key != "recon_mu_mae":
                    break
        self.metric_spacing_mm = tuple(
            float(v) for v in smooth_options.get(
                "spacing_mm",
                (1.5234375, 1.5234375, 2.0),
            )
        )
        self.metric_smooth_fwhm_mm = float(smooth_options.get("fwhm_mm", 4.0))
        self.metric_gaussian_truncate = float(smooth_options.get("truncate", 4.0))

    @staticmethod
    def _build_inference_options(val_cfg):
        roi_size = tuple(int(v) for v in val_cfg.patch_size)
        if len(roi_size) != 3:
            raise ValueError("val_cfg.patch_size must contain three values.")
        overlap = float(val_cfg.overlap_ratio)
        if not 0.0 <= overlap < 1.0:
            raise ValueError("val_cfg.overlap_ratio must satisfy 0 <= overlap < 1.")
        return {
            "roi_size": roi_size,
            "sw_batch_size": int(val_cfg.get("sw_batch_size", 1)),
            "overlap": overlap,
            "mode": str(val_cfg.get("infer_mode", "constant")),
            "sigma_scale": float(val_cfg.get("sigma_scale", 0.125)),
            "cache_roi_weight_map": bool(
                val_cfg.get("cache_roi_weight_map", False)
            ),
        }

    @staticmethod
    def _build_segmented_settings(val_cfg):
        settings = val_cfg.get("pet_segmented_inference", {})
        fov_diameter = settings.get("fov_diameter", 452)
        return {
            "enabled": bool(settings.get("enabled", True)),
            "pet_threshold": float(settings.get("pet_threshold", 10.0)),
            "num_segments": int(settings.get("num_segments", 3)),
            "segment_xy_margins": tuple(
                int(v) for v in settings.get(
                    "segment_xy_margins",
                    (48, 64, 48),
                )
            ),
            "segment_overlap_z": int(settings.get("segment_overlap_z", 24)),
            "model_context_xy": int(settings.get("model_context_xy", 16)),
            "model_context_z": int(settings.get("model_context_z", 16)),
            "global_z_margin": int(settings.get("global_z_margin", 16)),
            "alignment": int(settings.get("alignment", 16)),
            "bbox_quantiles": tuple(
                float(v) for v in settings.get(
                    "bbox_quantiles",
                    (0.0005, 0.9995),
                )
            ),
            "min_z_plane_voxels": int(settings.get("min_z_plane_voxels", 32)),
            "min_segment_voxels": int(settings.get("min_segment_voxels", 128)),
            "background_value": float(settings.get("background_value", 0.0)),
            # Kept for exact config compatibility with production. The current
            # BackgroundAwareSlidingWindowInferer intentionally ignores the
            # final FOV mask.
            "fov_diameter": (
                None if fov_diameter is None else float(fov_diameter)
            ),
        }

    @staticmethod
    def _first_value(value):
        while isinstance(value, (list, tuple)):
            if not value:
                return None
            value = value[0]
        if isinstance(value, np.ndarray) and value.ndim == 0:
            value = value.item()
        return value

    def _raw_pet_path_from_batch(self, batch):
        candidates = []

        if "features_dir" in batch:
            features_dir = self._first_value(batch["features_dir"])
            if features_dir is not None:
                features_dir = Path(str(features_dir))
                candidates.extend(
                    [
                        features_dir / "nacpet.nii.gz",
                        features_dir / "pet.nii.gz",
                    ]
                )

        pet_key = next(
            (key for key in ("pet", "nacpet") if key in batch),
            None,
        )
        if pet_key is not None:
            pet = batch[pet_key]
            meta = getattr(pet, "meta", None)
            if meta is not None:
                filename = self._first_value(meta.get("filename_or_obj", None))
                if filename is not None:
                    candidates.append(Path(str(filename)))

        for path in candidates:
            if path.is_file():
                return path

        rendered = ", ".join(str(path) for path in candidates) or "<none>"
        raise RuntimeError(
            "PET-segmented validation/test inference requires the raw NAC-PET "
            "used by production to build BackgroundContext. Could not resolve "
            f"a raw PET path from the batch. Candidates: {rendered}. "
            "Keep features_dir or MONAI filename_or_obj metadata in validation/test."
        )

    def _load_raw_pet_for_context(self, batch, expected_shape):
        path = self._raw_pet_path_from_batch(batch)
        raw_pet = np.asarray(
            nib.load(str(path)).get_fdata(dtype=np.float32),
            dtype=np.float32,
        )
        raw_pet = np.nan_to_num(raw_pet, nan=0.0, posinf=0.0, neginf=0.0)
        if tuple(raw_pet.shape) != tuple(expected_shape):
            raise ValueError(
                f"Raw PET shape {raw_pet.shape} from {path} does not match "
                f"model input shape {tuple(expected_shape)}. Validation/test "
                "must use the same spatial grid as production inference."
            )
        return raw_pet

    def _predict_normalized(self, batch):
        x = self.get_x(batch)

        if not self.segmented_settings["enabled"]:
            inferer = SlidingWindowInferer(
                **self.inference_options,
                sw_device=x.device,
                device=torch.device("cpu"),
                progress=False,
            )
            return inferer(x, self.forward)

        if x.shape[0] != 1:
            raise ValueError(
                "Production-aligned PET-segmented inference is subject-wise and "
                f"requires validation/test batch_size=1, got {x.shape[0]}."
            )

        raw_pet = self._load_raw_pet_for_context(batch, x.shape[2:])
        context_options = dict(self.segmented_settings)
        context_options.pop("enabled", None)
        context = BackgroundContext.from_pet(
            raw_pet,
            roi_size=self.inference_options["roi_size"],
            **context_options,
        )
        inferer = BackgroundAwareSlidingWindowInferer(
            **self.inference_options,
            sw_device=x.device,
            device=torch.device("cpu"),
            progress=False,
        )
        return inferer(x, self.forward, context)

    def forward(self, inputs):
        output = self.net(inputs)
        return output[0] if isinstance(output, (tuple, list)) else output

    def compute_loss(
        self,
        pred,
        target,
        prediction_mask=None,
        organ_seg=None,
        body_seg=None,
    ):
        loss, loss_items = self.criterion(
            y_pred=pred,
            y=target,
            prediction_mask=prediction_mask,
            organ_seg=organ_seg,
            body_seg=body_seg,
        )

        loss_items["total_loss"] = loss
        loss_items.pop("total", None)

        return loss_items

    def get_x(self, batch):
        missing = [key for key in self.input_keys if key not in batch]
        if missing:
            raise KeyError(f"Missing model inputs {missing}; batch keys={tuple(batch)}")

        x = torch.cat([batch[key] for key in self.input_keys], dim=1)
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last_3d)
        return x

    def training_step(self, batch):
        x = self.get_x(batch)
        prediction_mask = batch.get("prediction_mask", None)
        organ_seg = batch.get("organ_seg", None)
        body_seg = batch.get("body_seg", None)
        y = batch["ct"]

        prediction = self.forward(x)

        loss_items = self.compute_loss(
            pred=prediction,
            target=y,
            prediction_mask=prediction_mask,
            organ_seg=organ_seg,
            body_seg=body_seg,
        )

        batch.update({"prediction": prediction})

        return batch, loss_items

    def _predict_hu(self, batch, output_device=None):
        prediction = self._predict_normalized(batch)
        prediction = prediction.float().mul(3000.0).sub(1000.0)
        prediction = torch.clamp(prediction, min=-1024.0, max=3071.0)
        if output_device is not None:
            prediction = prediction.to(output_device)
        return prediction

    def validation_step(self, batch):
        # Use the exact same PET-segmented sliding-window path as production,
        # then move only the assembled prediction back to the target device for
        # validation metrics. No additional foreground/FOV mask is applied.
        prediction = self._predict_hu(
            batch,
            output_device=batch["ct"].device,
        )
        batch["prediction"] = prediction

        val_metrics = evaluate(
            prediction,
            batch["ct"],
            body_mask=batch.get("body_seg"),
            organ_mask=batch.get("organ_seg"),
            prediction_mask=batch.get("prediction_mask"),
            spacing_mm=self.metric_spacing_mm,
            smooth_fwhm_mm=self.metric_smooth_fwhm_mm,
            gaussian_truncate=self.metric_gaussian_truncate,
        )
        return batch, val_metrics

    def test_step(self, batch):
        # Preserve the previous model/test API: the returned prediction lives on
        # the same device as the model inputs, while stitching itself still
        # happens on CPU exactly as in production.
        output_device = batch[self.input_keys[0]].device
        batch["prediction"] = self._predict_hu(
            batch,
            output_device=output_device,
        )
        return batch

    def configure_optimizers_schedulers(self, num_steps):
        if self.config.optim_cfg.get("optimizer", "adamw").lower() != "adamw":
            raise ValueError("The streamlined training path supports AdamW only.")

        injector_parameters = [
            (name, parameter)
            for name, parameter in self.named_parameters()
            if ".injector." in name and parameter.requires_grad
        ]
        parameter_groups = []
        if injector_parameters:
            injector_names = {name for name, _ in injector_parameters}
            backbone_parameters = [
                (name, parameter)
                for name, parameter in self.named_parameters()
                if name not in injector_names and parameter.requires_grad
            ]
            peak_lr = float(self.config.optim_cfg.base_lr)
            min_lr = float(self.config.optim_cfg.lr_min)
            weight_decay = float(self.config.optim_cfg.get("weight_decay", 0.05))
            mode = str(self.config.optim_cfg.get("backbone_lr_mode", "same"))
            if mode not in {"same", "scaled", "levelwise"}:
                raise ValueError(f"Unsupported backbone_lr_mode: {mode}")

            parameter_groups.extend(adamw_groups(
                injector_parameters, "injector", peak_lr, min_lr, weight_decay
            ))
            if mode == "same":
                backbone_sets = [("backbone", backbone_parameters, 1.0)]
            elif mode == "scaled":
                scale = float(self.config.optim_cfg.get("backbone_lr_scale", 0.1))
                backbone_sets = [("backbone", backbone_parameters, scale)]
            else:
                max_level = len(self.config.model_cfg.net.features) - 1
                layer_decay = float(self.config.optim_cfg.get("layer_decay", 0.9))
                by_level = {level: [] for level in range(max_level + 2)}
                for name, parameter in backbone_parameters:
                    by_level[self._backbone_level(name, max_level)].append((name, parameter))
                backbone_sets = [
                    (
                        f"backbone_l{level}",
                        values,
                        1.0 if level > max_level else layer_decay ** (max_level - level),
                    )
                    for level, values in by_level.items() if values
                ]

            for name, parameters, scale in backbone_sets:
                parameter_groups.extend(adamw_groups(
                    parameters,
                    name,
                    peak_lr * scale,
                    min_lr * scale,
                    weight_decay,
                ))

        optimizer = torch.optim.AdamW(
            parameter_groups or self.parameters(),
            lr=float(self.config.optim_cfg.base_lr),
            betas=tuple(self.config.optim_cfg.get("betas", (0.9, 0.999))),
            eps=float(self.config.optim_cfg.get("eps", 1.0e-8)),
            weight_decay=float(self.config.optim_cfg.get("weight_decay", 0.02)),
        )
        scheduler_cfg = param_scheduler(
            self.config.optim_cfg,
            optimizer,
            num_steps,
            self.config.train_cfg.num_epochs,
        )
        return optimizer, scheduler_cfg

    @staticmethod
    def _backbone_level(name, max_level):
        name = name.removeprefix("net.")
        for prefix in ("encoder_stages.", "downs."):
            if name.startswith(prefix):
                index = int(name.split(".")[1])
                return min(index + (prefix == "downs."), max_level)
        for prefix in ("ups.", "skip_fuses.", "decoder_stages."):
            if name.startswith(prefix):
                index = int(name.split(".")[1])
                return max(max_level - 1 - index, 0)
        return max_level + 1 if name.startswith("out_conv.") else 0

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    @current_epoch.setter
    def current_epoch(self, epoch: int) -> None:
        self._current_epoch = int(epoch)



def param_scheduler(optim_cfg, optimizer, num_steps, num_epochs):
    start_factor = float(getattr(optim_cfg, "start_factor", 0.001))
    lr_min = float(optim_cfg.lr_min)
    scheduler_type = optim_cfg.scheduler
    warmup_epochs = int(getattr(optim_cfg, "warmup_epochs", 0))
    total_t = max(num_steps * num_epochs, 1)
    warmup_t = min(warmup_epochs * num_steps, total_t)
    if scheduler_type is not None:
        if scheduler_type not in {"cosine", "warmup"}:
            raise ValueError("Supported schedulers are cosine and warmup.")

        schedules = []
        for group in optimizer.param_groups:
            base_lr = float(group["lr"])
            group_min = float(group.get("lr_min", lr_min))
            min_factor = min(group_min / max(base_lr, 1.0e-12), 1.0)

            def schedule(step, min_factor=min_factor):
                step = min(max(int(step), 0), total_t)
                if warmup_t > 0 and step < warmup_t:
                    return start_factor + (1.0 - start_factor) * step / warmup_t
                if scheduler_type == "warmup":
                    return 1.0
                progress = (step - warmup_t) / max(total_t - warmup_t, 1)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_factor + (1.0 - min_factor) * cosine

            schedules.append(schedule)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedules)
        scheduler_cfg = {
            "type": scheduler_type,
            "scheduler": scheduler,
            "monitor": None,
            "interval": "step",
            "frequency": 1,
        }

    else:
        scheduler_cfg = None
    return scheduler_cfg


@torch.no_grad()
def make_cylindrical_fov_mask(
    image,
    fov_diameter=452,
    border_xy=30,
):
    """
    Create cylindrical FOV mask in XY plane.

    image:
        [B, C, X, Y, Z]

    returns:
        [B, 1, X, Y, Z] bool

    For 512 x 512:
        fov_diameter=452 means 30 voxels are left outside on both sides.
    """
    if hasattr(image, "as_tensor"):
        image = image.as_tensor()

    if image.ndim != 5:
        raise ValueError(f"Expected image [B,C,X,Y,Z], got {tuple(image.shape)}")

    b, _, x, y, z = image.shape
    device = image.device

    if fov_diameter is None:
        radius = 0.5 * (min(x, y) - 2 * int(border_xy))
    else:
        radius = 0.5 * float(fov_diameter)

    max_radius = 0.5 * min(x - 1, y - 1)
    radius = min(radius, max_radius)

    cx = 0.5 * (x - 1)
    cy = 0.5 * (y - 1)

    xx = torch.arange(x, device=device, dtype=torch.float32)[:, None]
    yy = torch.arange(y, device=device, dtype=torch.float32)[None, :]

    circle_xy = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    mask = circle_xy[None, None, :, :, None].expand(b, 1, x, y, z)

    return mask.bool()
