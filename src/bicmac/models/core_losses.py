import math
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn


def _tensor(x):
    if hasattr(x, "as_tensor"):
        x = x.as_tensor()
    return torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)


def norm_to_hu(x):
    return x * 3000.0 - 1000.0


def hu_to_mu(x):
    """Carney bilinear HU-to-mu conversion at 511 keV (mu in cm^-1)."""
    shifted = x + 1000.0
    return torch.where(
        shifted < 1047.0,
        9.6e-5 * shifted,
        5.10e-5 * shifted + 4.71e-2,
    ).clamp_min(0.0)


def _mask_like(mask, reference):
    if mask is None:
        return None
    mask = _tensor(mask).to(reference.device)
    if mask.ndim == reference.ndim - 1:
        mask = mask.unsqueeze(1)
    if mask.ndim != reference.ndim:
        raise ValueError(
            f"Mask dimensionality mismatch: reference={tuple(reference.shape)}, "
            f"mask={tuple(mask.shape)}"
        )
    if mask.shape[2:] != reference.shape[2:]:
        mask = F.interpolate(mask, size=reference.shape[2:], mode="nearest")
    if mask.shape[1] == 1 and reference.shape[1] != 1:
        mask = mask.expand(-1, reference.shape[1], *mask.shape[2:])
    elif mask.shape[1] != reference.shape[1]:
        raise ValueError(
            f"Mask channel mismatch: reference={tuple(reference.shape)}, "
            f"mask={tuple(mask.shape)}"
        )
    return (mask > 0).to(reference.dtype)


def _mean(loss_map, mask, reference):
    if mask is None:
        return loss_map.mean()
    numerator = (loss_map * mask).flatten(1).sum(dim=1)
    denominator = mask.flatten(1).sum(dim=1)
    valid = denominator > 0
    per_sample = numerator / denominator.clamp_min(1.0)
    return (per_sample * valid).sum() / valid.sum().clamp_min(1)


def _weighted_sample_mean(loss_map, weight):
    """Weighted mean per sample, followed by an equal mean over valid samples."""
    numerator = (loss_map * weight).flatten(1).sum(dim=1)
    denominator = weight.flatten(1).sum(dim=1)
    valid = denominator > 0
    per_sample = numerator / denominator.clamp_min(1.0e-6)
    return (per_sample * valid).sum() / valid.sum().clamp_min(1)


def _gaussian_blur3d(x, spacing_mm, fwhm_mm):
    """
    Separable 3D Gaussian blur for tensors laid out as [B, C, X, Y, Z].

    The official reconstruction applies a 4 mm FWHM Gaussian to the complete
    mu-map. Training is patch based, so zero padding is used at artificial patch
    boundaries rather than reflecting errors that are not actually observed
    outside the current patch.
    """
    fwhm_mm = float(fwhm_mm)
    if fwhm_mm <= 0.0:
        return x
    if len(spacing_mm) != 3:
        raise ValueError("spacing_mm must contain X, Y, and Z spacing.")

    sigma_mm = fwhm_mm / math.sqrt(8.0 * math.log(2.0))
    output = x
    for axis, spacing in enumerate(spacing_mm):
        spacing = float(spacing)
        if spacing <= 0.0:
            raise ValueError("All spacing values must be positive.")
        sigma_voxel = sigma_mm / spacing
        if sigma_voxel < 1.0e-6:
            continue
        radius = max(1, int(math.ceil(3.0 * sigma_voxel)))
        coordinate = torch.arange(
            -radius,
            radius + 1,
            device=output.device,
            dtype=torch.float32,
        )
        kernel = torch.exp(-0.5 * (coordinate / sigma_voxel).square())
        kernel = (kernel / kernel.sum()).to(output.dtype)

        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = kernel.numel()
        weight = kernel.view(*shape).repeat(output.shape[1], 1, 1, 1, 1)
        padding = [0, 0, 0]
        padding[axis] = radius
        output = F.conv3d(
            output,
            weight,
            padding=tuple(padding),
            groups=output.shape[1],
        )
    return output


def _official_gaussian_radii(spacing_mm, fwhm_mm=4.0, truncate=4.0):
    """
    Kernel radii matching scipy.ndimage.gaussian_filter's default radius rule.

    BIC-MAC uses sigma_i = FWHM / (2.355 * spacing_i) and scipy's default
    truncate=4.0. scipy chooses radius = int(truncate * sigma + 0.5).
    """
    if len(spacing_mm) != 3:
        raise ValueError("spacing_mm must contain X, Y, and Z spacing.")
    if float(fwhm_mm) <= 0.0:
        return (0, 0, 0)
    if float(truncate) <= 0.0:
        raise ValueError("truncate must be positive.")

    radii = []
    for spacing in spacing_mm:
        spacing = float(spacing)
        if spacing <= 0.0:
            raise ValueError("All spacing values must be positive.")
        sigma_voxel = float(fwhm_mm) / (2.355 * spacing)
        radii.append(max(0, int(float(truncate) * sigma_voxel + 0.5)))
    return tuple(radii)


def _gaussian_blur3d_valid(
    x,
    spacing_mm=(1.5234375, 1.5234375, 2.0),
    fwhm_mm=4.0,
    truncate=4.0,
):
    """
    Official-style 4-mm Gaussian on the valid interior of a training patch.

    The official reconstruction smooths the complete mu-map with SciPy
    gaussian_filter. A random training patch does not contain the true values
    outside its borders, so padding that patch would create an artificial
    attenuation boundary. This implementation therefore performs separable
    convolution with no padding and returns only voxels whose full Gaussian
    support is observed inside the patch.

    Returns
    -------
    output : Tensor
        Smoothed tensor with the artificial patch border removed.
    radii : tuple[int, int, int]
        Removed support radius along X, Y and Z.
    """
    if x.ndim != 5:
        raise ValueError(
            f"_gaussian_blur3d_valid expects [B,C,X,Y,Z], got {tuple(x.shape)}."
        )

    fwhm_mm = float(fwhm_mm)
    if fwhm_mm <= 0.0:
        return x, (0, 0, 0)

    radii = _official_gaussian_radii(
        spacing_mm=spacing_mm,
        fwhm_mm=fwhm_mm,
        truncate=truncate,
    )
    output = x
    for axis, (spacing, radius) in enumerate(zip(spacing_mm, radii)):
        if radius <= 0:
            continue
        if output.shape[2 + axis] <= 2 * radius:
            raise ValueError(
                "Patch is too small for valid Gaussian smoothing: "
                f"shape={tuple(x.shape)}, axis={axis}, radius={radius}."
            )

        sigma_voxel = fwhm_mm / (2.355 * float(spacing))
        coordinate = torch.arange(
            -radius,
            radius + 1,
            device=output.device,
            dtype=torch.float32,
        )
        kernel = torch.exp(-0.5 * (coordinate / sigma_voxel).square())
        kernel = (kernel / kernel.sum()).to(dtype=output.dtype)

        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = kernel.numel()
        weight = kernel.view(*shape).repeat(output.shape[1], 1, 1, 1, 1)
        output = F.conv3d(
            output,
            weight,
            padding=0,
            groups=output.shape[1],
        )
    return output, radii


def _valid_gaussian_support(mask, radii):
    """
    Return voxels in the valid blurred map that can depend on prediction_mask.

    If the reconstruction replaces CT outside prediction_mask with ground truth
    before smoothing, prediction error can still spill across the mask boundary
    by the Gaussian support. The support mask therefore dilates responsibility
    by exactly the Gaussian kernel footprint while simultaneously removing the
    artificial training-patch border.
    """
    if mask is None:
        return None
    if mask.ndim != 5:
        raise ValueError(
            f"_valid_gaussian_support expects [B,C,X,Y,Z], got {tuple(mask.shape)}."
        )
    kernel_size = tuple(2 * int(radius) + 1 for radius in radii)
    if any(size > dim for size, dim in zip(kernel_size, mask.shape[2:])):
        raise ValueError(
            f"Mask shape {tuple(mask.shape)} is too small for kernel {kernel_size}."
        )
    return (
        F.max_pool3d(
            mask.float(),
            kernel_size=kernel_size,
            stride=1,
            padding=0,
        )
        > 0
    ).to(mask.dtype)


def _prepare_labels(labels, reference, name):
    labels = _tensor(labels).to(reference.device)
    if labels.ndim == reference.ndim - 1:
        labels = labels.unsqueeze(1)
    if labels.ndim != reference.ndim:
        raise ValueError(
            f"{name} dimensionality mismatch: reference={tuple(reference.shape)}, "
            f"{name}={tuple(labels.shape)}"
        )
    if labels.shape[2:] != reference.shape[2:]:
        labels = F.interpolate(labels, size=reference.shape[2:], mode="nearest")
    if labels.shape[1] != 1:
        raise ValueError(f"{name} must contain one label channel, got {labels.shape}.")
    return labels.round().long()


def _body_mask_like(body_seg, reference):
    if body_seg is None:
        return None
    labels = _prepare_labels(body_seg, reference, "body_seg")
    return (labels > 0).to(reference.dtype)


def _training_mask(prediction_mask, body_seg, reference):
    """Body-only responsibility mask used by CT-domain losses."""
    body = _body_mask_like(body_seg, reference)
    responsibility = _mask_like(prediction_mask, reference)
    if body is None:
        return responsibility
    if responsibility is None:
        return body
    return body * responsibility


class ReconstructionMuMAE(nn.Module):
    """
    Raw 511-keV mu MAE over the reconstruction responsibility region.

    Unlike ``mu_mae_masked``, this loss intentionally does NOT intersect the
    prediction mask with ``body_seg``. BIC-MAC replaces face/bed/far background
    outside ``prediction_mask`` with ground-truth CT before reconstruction, so
    the whole prediction_mask, including the near-body air rim, is relevant.
    """

    def __init__(self, require_prediction_mask=True):
        super().__init__()
        self.require_prediction_mask = bool(require_prediction_mask)

    def forward(
        self,
        y_pred,
        y,
        prediction_mask=None,
        organ_seg=None,
        body_seg=None,
    ):
        y_pred = _tensor(y_pred)
        y = _tensor(y)
        if y_pred.shape != y.shape:
            raise ValueError(
                f"ReconstructionMuMAE shape mismatch: {y_pred.shape} vs {y.shape}."
            )

        responsibility = _mask_like(prediction_mask, y_pred)
        if responsibility is None and self.require_prediction_mask:
            raise KeyError("ReconstructionMuMAE requires prediction_mask.")

        pred_mu = hu_to_mu(norm_to_hu(y_pred))
        target_mu = hu_to_mu(norm_to_hu(y))
        return _mean((pred_mu - target_mu).abs(), responsibility, pred_mu)


class SmoothedMuMAE(nn.Module):
    """
    Reconstruction-aligned MAE in the official 4-mm smoothed mu-map domain.

    Training tensors are normalized CT. The loss performs:
        normalized CT -> HU -> prediction-mask GT replacement
        -> Carney HU-to-mu -> official 4-mm Gaussian -> MAE

    Important details:
      * ``prediction_mask`` is used as the reconstruction responsibility mask;
        ``body_seg`` and ``organ_seg`` are intentionally ignored.
      * Outside ``prediction_mask``, predicted HU is replaced by target HU
        BEFORE HU-to-mu conversion and Gaussian smoothing, matching the official
        face/bed/background replacement order.
      * Gaussian convolution is valid on training patches. Artificial patch
        padding therefore cannot create fake attenuation boundaries.
      * The MAE is evaluated on the exact Gaussian influence support of
        ``prediction_mask`` after the valid crop, so boundary spill-over is
        retained without dilution by unrelated GT-replaced voxels.
    """

    def __init__(
        self,
        spacing_mm=(1.5234375, 1.5234375, 2.0),
        fwhm_mm=4.0,
        truncate=4.0,
        require_prediction_mask=True,
        hybrid_outside_prediction_mask=True,
    ):
        super().__init__()
        self.spacing_mm = tuple(float(v) for v in spacing_mm)
        if len(self.spacing_mm) != 3 or any(v <= 0.0 for v in self.spacing_mm):
            raise ValueError("spacing_mm must contain three positive values.")
        self.fwhm_mm = float(fwhm_mm)
        self.truncate = float(truncate)
        if self.fwhm_mm <= 0.0:
            raise ValueError("fwhm_mm must be positive.")
        if self.truncate <= 0.0:
            raise ValueError("truncate must be positive.")
        self.require_prediction_mask = bool(require_prediction_mask)
        self.hybrid_outside_prediction_mask = bool(
            hybrid_outside_prediction_mask
        )

    def forward(
        self,
        y_pred,
        y,
        prediction_mask=None,
        organ_seg=None,
        body_seg=None,
    ):
        y_pred = _tensor(y_pred)
        y = _tensor(y)
        if y_pred.ndim != 5 or y.shape != y_pred.shape:
            raise ValueError(
                "SmoothedMuMAE expects matching [B,C,X,Y,Z] tensors, "
                f"got pred={tuple(y_pred.shape)}, target={tuple(y.shape)}."
            )
        if y_pred.shape[1] != 1:
            raise ValueError("SmoothedMuMAE expects a single CT channel.")

        responsibility = _mask_like(prediction_mask, y_pred)
        if responsibility is None and self.require_prediction_mask:
            raise KeyError("SmoothedMuMAE requires prediction_mask.")

        pred_hu = norm_to_hu(y_pred)
        target_hu = norm_to_hu(y)

        if responsibility is not None and self.hybrid_outside_prediction_mask:
            # Exact binary replacement in HU before the nonlinear HU-to-mu map.
            pred_hu = target_hu + responsibility * (pred_hu - target_hu)

        pred_mu = hu_to_mu(pred_hu)
        target_mu = hu_to_mu(target_hu)

        pred_smoothed, radii = _gaussian_blur3d_valid(
            pred_mu,
            spacing_mm=self.spacing_mm,
            fwhm_mm=self.fwhm_mm,
            truncate=self.truncate,
        )
        target_smoothed, target_radii = _gaussian_blur3d_valid(
            target_mu,
            spacing_mm=self.spacing_mm,
            fwhm_mm=self.fwhm_mm,
            truncate=self.truncate,
        )
        if radii != target_radii:
            raise RuntimeError(
                f"Gaussian support mismatch: {radii} vs {target_radii}."
            )

        support = (
            _valid_gaussian_support(responsibility, radii)
            if responsibility is not None
            else None
        )
        return _mean(
            (pred_smoothed - target_smoothed).abs(),
            support,
            pred_smoothed,
        )


class VoxelMAE(nn.Module):
    """
    Stable voxel MAE in normalized CT or 511-keV mu space.

    When masked=True, supervision is restricted to body_seg > 0 and, when
    supplied, prediction_mask > 0. organ_seg is intentionally not used.
    """

    def __init__(
        self,
        masked=False,
        mu_space=False,
        challenge_mask=False,
        slice_thickness_mm=2.0,
        exclusion_cm=4.0,
    ):
        super().__init__()
        self.masked = bool(masked)
        self.mu_space = bool(mu_space)
        self.challenge_mask = bool(challenge_mask)
        # Retained only for config compatibility. Exact liver exclusion cannot
        # be constructed without organ labels and is therefore not approximated
        # from a cropped patch.
        self.slice_thickness_mm = float(slice_thickness_mm)
        self.exclusion_cm = float(exclusion_cm)

    def forward(
        self,
        y_pred,
        y,
        prediction_mask=None,
        organ_seg=None,
        body_seg=None,
    ):
        y_pred = _tensor(y_pred)
        y = _tensor(y)
        if self.mu_space:
            y_pred = hu_to_mu(norm_to_hu(y_pred))
            y = hu_to_mu(norm_to_hu(y))
        mask = (
            _training_mask(prediction_mask, body_seg, y_pred)
            if self.masked
            else None
        )
        return _mean((y_pred - y).abs(), mask, y_pred)


class CTStructureLoss(nn.Module):
    """Edge and multi-scale CT consistency in normalized CT or mu space."""

    def __init__(
        self,
        edge_weight=1.0,
        multiscale_weight=1.0,
        scales=(2, 4),
        mu_space=False,
    ):
        super().__init__()
        self.edge_weight = float(edge_weight)
        self.multiscale_weight = float(multiscale_weight)
        self.scales = tuple(int(scale) for scale in scales)
        self.mu_space = bool(mu_space)

    def forward(
        self,
        y_pred,
        y,
        prediction_mask=None,
        organ_seg=None,
        body_seg=None,
    ):
        y_pred = _tensor(y_pred)
        y = _tensor(y)
        if self.mu_space:
            y_pred = hu_to_mu(norm_to_hu(y_pred))
            y = hu_to_mu(norm_to_hu(y))
        mask = _training_mask(prediction_mask, body_seg, y_pred)

        edge = y_pred.sum() * 0.0
        edge_axes = 0
        for axis in range(2, y_pred.ndim):
            if y_pred.shape[axis] < 2:
                continue
            pred_grad = torch.diff(y_pred, dim=axis)
            target_grad = torch.diff(y, dim=axis)
            grad_mask = None
            if mask is not None:
                left = [slice(None)] * mask.ndim
                right = [slice(None)] * mask.ndim
                left[axis] = slice(1, None)
                right[axis] = slice(None, -1)
                grad_mask = mask[tuple(left)] * mask[tuple(right)]
            edge = edge + _mean((pred_grad - target_grad).abs(), grad_mask, pred_grad)
            edge_axes += 1
        edge = edge / max(edge_axes, 1)

        multiscale = y_pred.sum() * 0.0
        for scale in self.scales:
            kernel = tuple(scale if size >= scale else 1 for size in y_pred.shape[2:])
            pred_level = F.avg_pool3d(y_pred, kernel_size=kernel, stride=kernel)
            target_level = F.avg_pool3d(y, kernel_size=kernel, stride=kernel)
            mask_level = None
            if mask is not None:
                mask_level = F.avg_pool3d(mask, kernel_size=kernel, stride=kernel)
                mask_level = (mask_level > 0.5).to(y_pred.dtype)
            multiscale = multiscale + _mean(
                (pred_level - target_level).abs(),
                mask_level,
                pred_level,
            )
        multiscale = multiscale / max(len(self.scales), 1)
        return self.edge_weight * edge + self.multiscale_weight * multiscale


class PETProjectionLoss(nn.Module):
    """
    Generic body-aware PET attenuation projection surrogate.

    body_seg is the only anatomical prior:
        0 = background
        1 = body / trunk
        2 = head and extremities

    The implementation is intentionally generic:
      * background is exactly zero in the mu-map;
      * prediction_mask limits the region where prediction-vs-GT error exists;
      * the complete active-body attenuation error is projected along each ray;
      * body labels only select / balance ray groups, never rescale voxel mu;
      * the same ray-error map is used for global and class-balanced objectives.

    This keeps patch sampling and PET projection aligned around the same body
    classes without hard-coding organ-specific losses.
    """

    def __init__(
        self,
        spacing_mm=(1.5234375, 1.5234375, 2.0),
        angles=None,
        downsample_factor=1,
        blur_fwhm_mm=4.0,
        hu_min=-1000.0,
        hu_max=3000.0,
        max_log_acf=0.75,
        log_acf_weight=0.35,
        relative_acf_weight=0.45,
        tail_weight=0.20,
        tail_thresholds=(0.05, 0.10, 0.15),
        tail_temperature=0.01,
        global_weight=1.0,
        regional_weight=0.75,
        bias_weight=0.35,
        log_smooth_l1_beta=0.02,
        acf_smooth_l1_beta=0.02,
        min_body_path_cm=0.20,
        min_responsibility_path_cm=0.20,
        full_target_integral=2.50,
        attenuation_support_power=0.5,
        uniform_ray_weight=0.50,
        attenuation_ray_weight=0.50,
        region_labels=(1, 2),
        region_weights=(1.0, 1.25),
        min_region_path_cm=0.20,
        full_region_path_cm=8.0,
        region_support_power=0.5,
        min_region_rays=8,
        require_body_seg=True,
        require_prediction_mask=True,
    ):
        super().__init__()

        self.spacing_mm = tuple(float(v) for v in spacing_mm)
        if len(self.spacing_mm) != 3 or any(v <= 0.0 for v in self.spacing_mm):
            raise ValueError("spacing_mm must contain positive X, Y, Z spacing.")

        if angles is None:
            angles = tuple(180.0 * index / 32.0 for index in range(32))
        self.angles = tuple(float(v) for v in angles)
        if not self.angles:
            raise ValueError("angles cannot be empty.")

        self.downsample_factor = int(downsample_factor)
        if self.downsample_factor < 1:
            raise ValueError("downsample_factor must be >= 1.")

        self.blur_fwhm_mm = float(blur_fwhm_mm)
        self.hu_min = float(hu_min)
        self.hu_max = float(hu_max)
        if self.hu_max <= self.hu_min:
            raise ValueError("hu_max must be larger than hu_min.")

        self.max_log_acf = float(max_log_acf)
        if self.max_log_acf <= 0.0:
            raise ValueError("max_log_acf must be positive.")

        self.log_acf_weight = float(log_acf_weight)
        self.relative_acf_weight = float(relative_acf_weight)
        self.tail_weight = float(tail_weight)
        representation_weights = (
            self.log_acf_weight,
            self.relative_acf_weight,
            self.tail_weight,
        )
        if min(representation_weights) < 0.0 or sum(representation_weights) <= 0.0:
            raise ValueError("Projection representation weights are invalid.")

        self.tail_thresholds = tuple(float(v) for v in tail_thresholds)
        self.tail_temperature = float(tail_temperature)
        if self.tail_weight > 0.0:
            if not self.tail_thresholds or any(v <= 0.0 for v in self.tail_thresholds):
                raise ValueError("tail_thresholds must contain positive values.")
            if self.tail_temperature <= 0.0:
                raise ValueError("tail_temperature must be positive.")

        self.global_weight = float(global_weight)
        self.regional_weight = float(regional_weight)
        self.bias_weight = float(bias_weight)
        objective_weights = (
            self.global_weight,
            self.regional_weight,
            self.bias_weight,
        )
        if min(objective_weights) < 0.0 or sum(objective_weights) <= 0.0:
            raise ValueError("PET objective weights are invalid.")

        self.log_smooth_l1_beta = float(log_smooth_l1_beta)
        self.acf_smooth_l1_beta = float(acf_smooth_l1_beta)
        if self.log_smooth_l1_beta <= 0.0 or self.acf_smooth_l1_beta <= 0.0:
            raise ValueError("SmoothL1 beta values must be positive.")

        self.min_body_path_cm = float(min_body_path_cm)
        self.min_responsibility_path_cm = float(min_responsibility_path_cm)
        self.full_target_integral = float(full_target_integral)
        self.attenuation_support_power = float(attenuation_support_power)
        self.uniform_ray_weight = float(uniform_ray_weight)
        self.attenuation_ray_weight = float(attenuation_ray_weight)
        if self.min_body_path_cm < 0.0 or self.min_responsibility_path_cm < 0.0:
            raise ValueError("Minimum path lengths must be non-negative.")
        if self.full_target_integral <= 0.0:
            raise ValueError("full_target_integral must be positive.")
        if self.attenuation_support_power <= 0.0:
            raise ValueError("attenuation_support_power must be positive.")
        if self.uniform_ray_weight < 0.0 or self.attenuation_ray_weight < 0.0:
            raise ValueError("Ray weights must be non-negative.")
        if self.uniform_ray_weight + self.attenuation_ray_weight <= 0.0:
            raise ValueError("At least one ray weighting term must be active.")

        self.region_labels = tuple(int(v) for v in region_labels)
        self.region_weights = tuple(float(v) for v in region_weights)
        if len(self.region_labels) != len(self.region_weights):
            raise ValueError("region_labels and region_weights must have equal length.")
        if any(v <= 0.0 for v in self.region_weights):
            raise ValueError("region_weights must be positive.")

        self.min_region_path_cm = float(min_region_path_cm)
        self.full_region_path_cm = float(full_region_path_cm)
        self.region_support_power = float(region_support_power)
        self.min_region_rays = int(min_region_rays)
        if self.min_region_path_cm < 0.0:
            raise ValueError("min_region_path_cm must be non-negative.")
        if self.full_region_path_cm <= 0.0:
            raise ValueError("full_region_path_cm must be positive.")
        if self.region_support_power <= 0.0:
            raise ValueError("region_support_power must be positive.")
        if self.min_region_rays < 1:
            raise ValueError("min_region_rays must be >= 1.")

        self.require_body_seg = bool(require_body_seg)
        self.require_prediction_mask = bool(require_prediction_mask)

    @staticmethod
    def _pad_to_diagonal(x):
        height, width = x.shape[-2:]
        diagonal = int(math.ceil(math.sqrt(height * height + width * width)))
        pad_h = diagonal - height
        pad_w = diagonal - width
        return F.pad(
            x,
            (
                pad_w // 2,
                pad_w - pad_w // 2,
                pad_h // 2,
                pad_h - pad_h // 2,
            ),
        )

    @staticmethod
    def _rotation_grid(reference, angle):
        radians = math.radians(float(angle) % 180.0)
        theta = reference.new_tensor(
            [
                [math.cos(radians), -math.sin(radians), 0.0],
                [math.sin(radians), math.cos(radians), 0.0],
            ]
        )
        theta = theta.unsqueeze(0).expand(reference.shape[0], -1, -1)
        return F.affine_grid(theta, reference.shape, align_corners=False)

    @staticmethod
    def _rotate(x, angle, grid=None):
        normalized = float(angle) % 180.0
        if math.isclose(normalized, 0.0, abs_tol=1.0e-6):
            return x
        if math.isclose(normalized, 90.0, abs_tol=1.0e-6):
            return x.transpose(-2, -1)
        if grid is None:
            grid = PETProjectionLoss._rotation_grid(x, normalized)
        return F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

    @staticmethod
    def _to_slices(volume):
        # [B,C,X,Y,Z] -> [B*Z,C,X,Y]
        return volume.permute(0, 4, 1, 2, 3).flatten(0, 1)

    @staticmethod
    def _per_sample_weighted_mean(value, weight, min_rays=1):
        value = value.flatten(1)
        weight = weight.flatten(1)
        denominator = weight.sum(dim=1)
        ray_count = (weight > 0).sum(dim=1)
        valid = (denominator > 0.0) & (ray_count >= int(min_rays))
        mean = (value * weight).sum(dim=1) / denominator.clamp_min(1.0e-6)
        return mean, valid

    @staticmethod
    def _mean_over_valid(values, valid, zero):
        if bool(valid.any()):
            return values[valid].mean()
        return zero

    @staticmethod
    def _slice_bias(relative_acf, ray_weight, zero):
        # [B,Z,D]. Preserve local systematic bias by taking absolute value
        # after the signed detector mean for each axial level.
        numerator = (relative_acf * ray_weight).sum(dim=-1)
        denominator = ray_weight.sum(dim=-1)
        valid_z = denominator > 0.0
        signed_z = numerator / denominator.clamp_min(1.0e-6)
        per_sample_num = (signed_z.abs() * valid_z).sum(dim=-1)
        per_sample_den = valid_z.sum(dim=-1)
        valid_sample = per_sample_den > 0
        per_sample = per_sample_num / per_sample_den.clamp_min(1)
        if bool(valid_sample.any()):
            return per_sample[valid_sample].mean()
        return zero

    def _base_ray_weight(self, target_integral, body_path_cm, responsibility_path_cm):
        valid = (
            (body_path_cm >= self.min_body_path_cm)
            & (responsibility_path_cm >= self.min_responsibility_path_cm)
        )
        attenuation = (
            target_integral / self.full_target_integral
        ).clamp(0.0, 1.0).pow(self.attenuation_support_power)
        weight = (
            self.uniform_ray_weight
            + self.attenuation_ray_weight * attenuation
        ) / (self.uniform_ray_weight + self.attenuation_ray_weight)
        return torch.where(valid, weight, torch.zeros_like(weight))

    def _region_ray_weight(self, base_weight, region_path_cm):
        support = (
            region_path_cm / self.full_region_path_cm
        ).clamp(0.0, 1.0).pow(self.region_support_power)
        support = torch.where(
            region_path_cm >= self.min_region_path_cm,
            support,
            torch.zeros_like(support),
        )
        return base_weight * support

    def _ray_error(self, delta_log_acf):
        terms = []
        weights = []

        if self.log_acf_weight > 0.0:
            log_error = F.smooth_l1_loss(
                delta_log_acf,
                torch.zeros_like(delta_log_acf),
                beta=self.log_smooth_l1_beta,
                reduction="none",
            )
            terms.append(self.log_acf_weight * log_error)
            weights.append(self.log_acf_weight)

        relative_acf = torch.expm1(
            delta_log_acf.clamp(-self.max_log_acf, self.max_log_acf)
        )

        if self.relative_acf_weight > 0.0:
            acf_error = F.smooth_l1_loss(
                relative_acf,
                torch.zeros_like(relative_acf),
                beta=self.acf_smooth_l1_beta,
                reduction="none",
            )
            terms.append(self.relative_acf_weight * acf_error)
            weights.append(self.relative_acf_weight)

        if self.tail_weight > 0.0:
            abs_relative_acf = relative_acf.abs()
            tail_terms = [
                F.softplus(
                    (abs_relative_acf - threshold) / self.tail_temperature
                ) * self.tail_temperature
                for threshold in self.tail_thresholds
            ]
            tail_error = torch.stack(tail_terms, dim=0).mean(dim=0)
            terms.append(self.tail_weight * tail_error)
            weights.append(self.tail_weight)

        return sum(terms) / sum(weights), relative_acf

    def forward(
        self,
        y_pred,
        y,
        prediction_mask=None,
        organ_seg=None,
        body_seg=None,
    ):
        y_pred = _tensor(y_pred)
        y = _tensor(y)
        if y_pred.ndim != 5 or y.shape != y_pred.shape:
            raise ValueError(
                "PETProjectionLoss expects matching [B,C,X,Y,Z] tensors, "
                f"got pred={tuple(y_pred.shape)}, target={tuple(y.shape)}."
            )
        if y_pred.shape[1] != 1:
            raise ValueError("PETProjectionLoss expects a single CT channel.")

        zero = y_pred.sum() * 0.0

        if body_seg is None:
            if self.require_body_seg:
                raise KeyError("PETProjectionLoss requires body_seg.")
            body_labels = torch.ones_like(y_pred, dtype=torch.long)
        else:
            body_labels = _prepare_labels(body_seg, y_pred, "body_seg")

        body = (body_labels > 0).to(y_pred.dtype)

        responsibility = _mask_like(prediction_mask, y_pred)
        if responsibility is None:
            if self.require_prediction_mask:
                raise KeyError("PETProjectionLoss requires prediction_mask.")
            responsibility = torch.ones_like(y_pred)

        # Model responsibility inside physical body support.
        active = body * responsibility

        pred_hu = norm_to_hu(y_pred).clamp(self.hu_min, self.hu_max)
        target_hu = norm_to_hu(y).clamp(self.hu_min, self.hu_max)

        # Background is exactly zero. Only active-body prediction error can
        # affect the learned attenuation correction.
        pred_mu = hu_to_mu(pred_hu) * active
        target_mu_active = hu_to_mu(target_hu) * active

        delta_mu = _gaussian_blur3d(
            pred_mu - target_mu_active,
            spacing_mm=self.spacing_mm,
            fwhm_mm=self.blur_fwhm_mm,
        )

        with torch.no_grad():
            # Physical target attenuation uses the whole body in the patch,
            # while error remains restricted to model responsibility.
            target_mu_body = hu_to_mu(target_hu) * body
            target_mu_body = _gaussian_blur3d(
                target_mu_body,
                spacing_mm=self.spacing_mm,
                fwhm_mm=self.blur_fwhm_mm,
            )
            region_masks = [
                ((body_labels == label).to(y_pred.dtype) * active)
                for label in self.region_labels
            ]

        factor = self.downsample_factor
        if factor > 1:
            kernel = tuple(
                factor if size >= factor else 1
                for size in delta_mu.shape[2:]
            )
            delta_mu = F.avg_pool3d(delta_mu, kernel_size=kernel, stride=kernel)
            with torch.no_grad():
                target_mu_body = F.avg_pool3d(
                    target_mu_body, kernel_size=kernel, stride=kernel
                )
                body = F.avg_pool3d(body, kernel_size=kernel, stride=kernel)
                active = F.avg_pool3d(active, kernel_size=kernel, stride=kernel)
                region_masks = [
                    F.avg_pool3d(mask, kernel_size=kernel, stride=kernel)
                    for mask in region_masks
                ]
        else:
            kernel = (1, 1, 1)

        batch_size = delta_mu.shape[0]
        z_size = delta_mu.shape[-1]

        delta_slices = self._pad_to_diagonal(self._to_slices(delta_mu))
        with torch.no_grad():
            static = torch.cat(
                [target_mu_body, body, active, *region_masks],
                dim=1,
            )
            static_slices = self._pad_to_diagonal(self._to_slices(static))

        spacing_x_cm = self.spacing_mm[0] * kernel[0] / 10.0
        spacing_y_cm = self.spacing_mm[1] * kernel[1] / 10.0

        global_losses = []
        bias_losses = []
        regional_losses = []

        for angle in self.angles:
            normalized = float(angle) % 180.0
            if math.isclose(normalized, 0.0, abs_tol=1.0e-6) or math.isclose(
                normalized, 90.0, abs_tol=1.0e-6
            ):
                grid = None
            else:
                with torch.no_grad():
                    grid = self._rotation_grid(delta_slices.detach(), normalized)

            rotated_delta = self._rotate(delta_slices, normalized, grid)
            with torch.no_grad():
                rotated_static = self._rotate(static_slices, normalized, grid)

            radians = math.radians(normalized)
            ray_step_cm = math.sqrt(
                (spacing_x_cm * math.cos(radians)) ** 2
                + (spacing_y_cm * math.sin(radians)) ** 2
            )

            delta_log_acf = rotated_delta.sum(dim=-2) * ray_step_cm
            with torch.no_grad():
                static_projection = rotated_static.sum(dim=-2) * ray_step_cm

            detector_size = delta_log_acf.shape[-1]
            delta_log_acf = delta_log_acf.view(batch_size, z_size, detector_size)
            with torch.no_grad():
                static_projection = static_projection.view(
                    batch_size,
                    z_size,
                    static_projection.shape[1],
                    detector_size,
                )
                target_integral = static_projection[:, :, 0, :]
                body_path_cm = static_projection[:, :, 1, :]
                responsibility_path_cm = static_projection[:, :, 2, :]
                region_paths = [
                    static_projection[:, :, 3 + index, :]
                    for index in range(len(self.region_labels))
                ]
                base_ray_weight = self._base_ray_weight(
                    target_integral,
                    body_path_cm,
                    responsibility_path_cm,
                )

            ray_error, relative_acf = self._ray_error(delta_log_acf)

            if self.global_weight > 0.0:
                global_losses.append(
                    _weighted_sample_mean(ray_error, base_ray_weight)
                )

            if self.bias_weight > 0.0:
                bias_losses.append(
                    self._slice_bias(relative_acf, base_ray_weight, zero)
                )

            if self.regional_weight > 0.0 and region_paths:
                per_region = []
                per_region_weights = []
                for region_path_cm, configured_weight in zip(
                    region_paths,
                    self.region_weights,
                ):
                    with torch.no_grad():
                        region_ray_weight = self._region_ray_weight(
                            base_ray_weight,
                            region_path_cm,
                        )
                    sample_mean, valid = self._per_sample_weighted_mean(
                        ray_error,
                        region_ray_weight,
                        min_rays=self.min_region_rays,
                    )
                    region_loss = self._mean_over_valid(sample_mean, valid, zero)
                    if bool(valid.any()):
                        per_region.append(configured_weight * region_loss)
                        per_region_weights.append(configured_weight)

                if per_region:
                    regional_losses.append(
                        sum(per_region, zero) / sum(per_region_weights)
                    )

        global_loss = (
            torch.stack(global_losses).mean() if global_losses else zero
        )
        bias_loss = (
            torch.stack(bias_losses).mean() if bias_losses else zero
        )
        regional_loss = (
            torch.stack(regional_losses).mean() if regional_losses else zero
        )

        weighted_terms = []
        active_weights = []
        if self.global_weight > 0.0:
            weighted_terms.append(self.global_weight * global_loss)
            active_weights.append(self.global_weight)
        if self.regional_weight > 0.0 and regional_losses:
            weighted_terms.append(self.regional_weight * regional_loss)
            active_weights.append(self.regional_weight)
        if self.bias_weight > 0.0:
            weighted_terms.append(self.bias_weight * bias_loss)
            active_weights.append(self.bias_weight)

        return sum(weighted_terms, zero) / max(sum(active_weights), 1.0e-12)


class SegmentationGuidedMAE(nn.Module):
    """Equal-weight MAE over body_seg labels 1 and 2; organ_seg is unused."""

    def __init__(self, min_voxels=16, mu_space=False):
        super().__init__()
        self.min_voxels = int(min_voxels)
        self.mu_space = bool(mu_space)

    def forward(
        self,
        y_pred,
        y,
        prediction_mask=None,
        organ_seg=None,
        body_seg=None,
    ):
        y_pred = _tensor(y_pred)
        y = _tensor(y)
        if self.mu_space:
            y_pred = hu_to_mu(norm_to_hu(y_pred))
            y = hu_to_mu(norm_to_hu(y))
        if body_seg is None:
            return y_pred.sum() * 0.0

        labels = _prepare_labels(body_seg, y_pred, "body_seg")
        responsibility = _mask_like(prediction_mask, y_pred)
        difference = (y_pred - y).abs()
        sample_losses = []

        for sample_index in range(y_pred.shape[0]):
            region_losses = []
            for label in (1, 2):
                region = labels[sample_index] == label
                if responsibility is not None:
                    region = region & (responsibility[sample_index] > 0)
                if int(region.count_nonzero().item()) < self.min_voxels:
                    continue
                region_losses.append(difference[sample_index][region].mean())
            if region_losses:
                sample_losses.append(torch.stack(region_losses).mean())

        return (
            torch.stack(sample_losses).mean()
            if sample_losses
            else y_pred.sum() * 0.0
        )


def build_loss(name, options):
    if name in {"mae", "hu_mae"}:
        return VoxelMAE(**options)
    if name in {"mae_masked", "hu_mae_masked"}:
        return VoxelMAE(masked=True, **options)
    if name == "mu_mae":
        return VoxelMAE(mu_space=True, **options)
    if name == "mu_mae_masked":
        return VoxelMAE(masked=True, mu_space=True, **options)
    if name in {"recon_mu_mae", "reconstruction_mu_mae"}:
        return ReconstructionMuMAE(**options)
    if name in {
        "smoothed_mu_mae",
        "smooth_mu_mae",
        "mu_smoothed_mae",
        "recon_smoothed_mu_mae",
    }:
        return SmoothedMuMAE(**options)
    if name in {"ct_loss", "ct_structure"}:
        return CTStructureLoss(**options)
    if name in {"pet_projection", "lip", "ctp", "ct_projection"}:
        return PETProjectionLoss(**options)
    if name == "seg_guided":
        return SegmentationGuidedMAE(**options)
    raise ValueError(f"Unsupported loss: {name}")


CORE_LOSSES = {
    "mae",
    "mae_masked",
    "hu_mae",
    "hu_mae_masked",
    "mu_mae",
    "mu_mae_masked",
    "recon_mu_mae",
    "reconstruction_mu_mae",
    "smoothed_mu_mae",
    "smooth_mu_mae",
    "mu_smoothed_mae",
    "recon_smoothed_mu_mae",
    "ct_loss",
    "ct_structure",
    "ctp",
    "ct_projection",
    "pet_projection",
    "lip",
    "seg_guided",
}


class CombinedLoss(nn.Module):
    """Metric-aligned CT and attenuation-map voxel losses."""

    def __init__(self, loss_weights: Dict[str, float], loss_options=None):
        super().__init__()
        configured = {
            str(name): float(weight) for name, weight in loss_weights.items()
        }
        unknown = set(configured) - CORE_LOSSES
        if unknown:
            raise ValueError(f"Unsupported losses: {sorted(unknown)}")
        negative = {name: weight for name, weight in configured.items() if weight < 0.0}
        if negative:
            raise ValueError(f"Loss weights must be non-negative: {negative}")

        # A zero-weight legacy loss must not be instantiated or executed. This
        # is especially important for the expensive patch projection surrogate.
        self.loss_weights = {
            name: weight for name, weight in configured.items() if weight > 0.0
        }
        if not self.loss_weights:
            raise ValueError("At least one loss weight must be positive.")

        loss_options = loss_options or {}
        self.losses = nn.ModuleDict({
            name: build_loss(name, dict(loss_options.get(name, {})))
            for name in self.loss_weights
        })

    def forward(
        self,
        y_pred,
        y,
        prediction_mask=None,
        organ_seg=None,
        body_seg=None,
    ):
        total = y_pred.sum() * 0.0
        items = {}
        for name, loss_fn in self.losses.items():
            value = loss_fn(
                y_pred,
                y,
                prediction_mask=prediction_mask,
                organ_seg=organ_seg,
                body_seg=body_seg,
            )
            total = total + self.loss_weights[name] * value
            items[name] = value.detach()
        items["total_loss"] = total.detach()
        return total, items