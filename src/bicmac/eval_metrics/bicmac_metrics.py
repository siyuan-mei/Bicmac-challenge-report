import torch
import torch.nn.functional as F


LIVER_LABEL = 5
BRAIN_LABEL = 90

ORGAN_LABELS = {
    "brain": 90,
    "liver": 5,
    "spleen": 1,
    "heart": 51,
    "pancreas": 7,
}

DEFAULT_SPACING_MM = (1.5234375, 1.5234375, 2.0)
DEFAULT_SMOOTH_FWHM_MM = 4.0
DEFAULT_GAUSSIAN_TRUNCATE = 4.0


def hu_to_mu(ct_hu):
    """BIC-MAC/Carney bilinear HU-to-mu conversion at 511 keV (cm^-1)."""
    hu1000 = ct_hu.float() + 1000.0
    mu = torch.where(
        hu1000 < 1047.0,
        9.6e-5 * hu1000,
        5.10e-5 * hu1000 + 4.71e-2,
    )
    return torch.clamp(mu, min=0.0)


def masked_mean(x, mask):
    mask = mask.to(device=x.device, dtype=x.dtype)
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def _official_gaussian_radii(
    spacing_mm=DEFAULT_SPACING_MM,
    fwhm_mm=DEFAULT_SMOOTH_FWHM_MM,
    truncate=DEFAULT_GAUSSIAN_TRUNCATE,
):
    """Match scipy.ndimage.gaussian_filter's default radius rule."""
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


def _reflect_pad_for_axis(x, axis, radius):
    """SciPy ``mode=reflect`` padding for one axis of [B,C,X,Y,Z].

    SciPy's default Gaussian boundary mode is half-sample symmetric and repeats
    the edge sample (for example ``... b a | a b ...``). PyTorch's built-in
    ``F.pad(..., mode="reflect")`` uses the other reflection convention and
    therefore does not match SciPy at volume boundaries. Build the exact SciPy
    reflection explicitly with index_select.
    """
    if radius <= 0:
        return x
    dim = 2 + int(axis)
    size = x.shape[dim]
    if size <= radius:
        raise ValueError(
            f"Cannot scipy-reflect-pad axis {axis} by {radius}: shape={tuple(x.shape)}."
        )
    left_index = torch.arange(
        radius - 1,
        -1,
        -1,
        device=x.device,
        dtype=torch.long,
    )
    right_index = torch.arange(
        size - 1,
        size - radius - 1,
        -1,
        device=x.device,
        dtype=torch.long,
    )
    left = x.index_select(dim, left_index)
    right = x.index_select(dim, right_index)
    return torch.cat((left, x, right), dim=dim)


def gaussian_blur_mu_official(
    mu,
    spacing_mm=DEFAULT_SPACING_MM,
    fwhm_mm=DEFAULT_SMOOTH_FWHM_MM,
    truncate=DEFAULT_GAUSSIAN_TRUNCATE,
):
    """
    Full-volume 3D Gaussian matching the official pre-STIR mu smoothing.

    Input layout is [B,C,X,Y,Z]. The implementation is separable, uses
    sigma_i = FWHM / (2.355 * spacing_i), scipy-compatible truncate=4 radius,
    and reflect boundaries. This is for validation/full volumes, not random
    training patches.
    """
    if mu.ndim != 5:
        raise ValueError(f"Expected [B,C,X,Y,Z], got {tuple(mu.shape)}.")
    fwhm_mm = float(fwhm_mm)
    if fwhm_mm <= 0.0:
        return mu

    radii = _official_gaussian_radii(spacing_mm, fwhm_mm, truncate)
    output = mu.float()
    for axis, (spacing, radius) in enumerate(zip(spacing_mm, radii)):
        if radius <= 0:
            continue
        sigma_voxel = fwhm_mm / (2.355 * float(spacing))
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

        output = _reflect_pad_for_axis(output, axis, radius)
        output = F.conv3d(output, weight, padding=0, groups=output.shape[1])
    return output


def _gaussian_influence_support(mask, radii):
    """Voxels whose smoothed value can depend on a prediction-mask voxel.

    Use three separable 1-D max-pools instead of one large 3-D kernel. The
    resulting rectangular support is identical but is far cheaper for complete
    512x512xZ validation volumes.
    """
    support = (mask > 0).float()
    for axis, radius in enumerate(radii):
        radius = int(radius)
        if radius <= 0:
            continue
        kernel = [1, 1, 1]
        padding = [0, 0, 0]
        kernel[axis] = 2 * radius + 1
        padding[axis] = radius
        support = F.max_pool3d(
            support,
            kernel_size=tuple(kernel),
            stride=1,
            padding=tuple(padding),
        )
    return support > 0


def compute_reconstruction_mu_metrics(
    pred_ct,
    gt_ct,
    prediction_mask,
    spacing_mm=DEFAULT_SPACING_MM,
    fwhm_mm=DEFAULT_SMOOTH_FWHM_MM,
    truncate=DEFAULT_GAUSSIAN_TRUNCATE,
    epsilon=1.0e-12,
):
    """
    PET-reconstruction-aligned image-domain diagnostics.

    The official reconstruction replaces CT outside prediction_mask with GT CT
    before HU->mu and 4-mm smoothing. Because Gaussian smoothing is linear, the
    smoothed prediction-vs-GT difference can be computed exactly as

        G((mu_pred - mu_gt) * prediction_mask).

    This avoids allocating two full smoothed mu volumes while preserving the
    exact full-volume difference that is sent toward STIR forward projection.

    Returns scalar tensors:
      recon_mu_mae:
          Raw mu MAE in the reconstruction responsibility region.
      recon_mu_bias_signed / recon_mu_bias_abs:
          Signed / absolute mean raw mu bias in that region.
      smooth_mu_mae:
          MAE of the 4-mm-smoothed effective mu difference over the Gaussian
          influence support of prediction_mask.
      smooth_mu_bias_signed / smooth_mu_bias_abs:
          Signed / absolute mean of that smoothed difference.
      smooth_mu_coherence:
          |signed bias| / MAE. Close to 1 means strongly coherent same-sign
          attenuation error; close to 0 means substantial sign cancellation.
      smooth_mu_rmse:
          RMSE of the smoothed effective mu difference.
    """
    if prediction_mask is None:
        raise ValueError("prediction_mask is required for reconstruction-aligned metrics.")
    if pred_ct.shape != gt_ct.shape:
        raise ValueError(
            f"pred_ct and gt_ct must match, got {pred_ct.shape} and {gt_ct.shape}."
        )

    responsibility = prediction_mask > 0
    if responsibility.ndim == pred_ct.ndim - 1:
        responsibility = responsibility.unsqueeze(1)
    if responsibility.shape != pred_ct.shape:
        if responsibility.shape[1] == 1 and pred_ct.shape[1] == 1:
            pass
        else:
            raise ValueError(
                "prediction_mask shape must match CT shape; "
                f"got mask={tuple(responsibility.shape)}, ct={tuple(pred_ct.shape)}."
            )

    pred_mu = hu_to_mu(pred_ct)
    gt_mu = hu_to_mu(gt_ct)
    delta_mu = (pred_mu - gt_mu) * responsibility.to(pred_mu.dtype)

    raw_mae = masked_mean(delta_mu.abs(), responsibility)
    raw_bias_signed = masked_mean(delta_mu, responsibility)
    raw_bias_abs = raw_bias_signed.abs()

    smooth_delta = gaussian_blur_mu_official(
        delta_mu,
        spacing_mm=spacing_mm,
        fwhm_mm=fwhm_mm,
        truncate=truncate,
    )
    radii = _official_gaussian_radii(spacing_mm, fwhm_mm, truncate)
    influence = _gaussian_influence_support(responsibility, radii)

    smooth_mae = masked_mean(smooth_delta.abs(), influence)
    smooth_bias_signed = masked_mean(smooth_delta, influence)
    smooth_bias_abs = smooth_bias_signed.abs()
    smooth_rmse = torch.sqrt(masked_mean(smooth_delta.square(), influence))
    smooth_coherence = smooth_bias_abs / smooth_mae.clamp_min(float(epsilon))

    return {
        "recon_mu_mae": raw_mae,
        "recon_mu_bias_signed": raw_bias_signed,
        "recon_mu_bias_abs": raw_bias_abs,
        "smooth_mu_mae": smooth_mae,
        "smooth_mu_bias_signed": smooth_bias_signed,
        "smooth_mu_bias_abs": smooth_bias_abs,
        "smooth_mu_coherence": smooth_coherence,
        "smooth_mu_rmse": smooth_rmse,
    }


def compute_mu_mae(
    pred_ct,
    gt_ct,
    body_seg,
    organ_seg,
    slice_thickness_mm=2.0,
    exclusion_cm=4.0,
):
    """
    Whole-body challenge-style mu-MAE.

    Inputs:
        pred_ct:  [B, C, X, Y, Z], HU
        gt_ct:    [B, C, X, Y, Z], HU
        body_seg: [B, C, X, Y, Z]
        organ_seg:[B, C, X, Y, Z]

    Z axis is the last dimension.
    """
    pred_mu = hu_to_mu(pred_ct)
    gt_mu = hu_to_mu(gt_ct)

    body_mask = body_seg > 0
    liver_mask = organ_seg == LIVER_LABEL

    exclusion_slices = int(round((exclusion_cm * 10.0) / float(slice_thickness_mm)))

    z_size = pred_ct.shape[-1]
    z = torch.arange(z_size, device=pred_ct.device)

    liver_by_z = liver_mask.any(dim=(1, 2, 3))  # [B, Z]

    superior_z = torch.where(
        liver_by_z,
        z.view(1, z_size),
        z.new_full((pred_ct.shape[0], z_size), -1),
    ).amax(dim=1)

    has_liver = superior_z >= 0

    z_min = (superior_z - exclusion_slices).clamp_min(0)
    z_max = (superior_z + exclusion_slices).clamp_max(z_size)

    z_grid = z.view(1, 1, 1, 1, z_size)

    exclusion_mask = (
        has_liver.view(-1, 1, 1, 1, 1)
        & (z_grid >= z_min.view(-1, 1, 1, 1, 1))
        & (z_grid < z_max.view(-1, 1, 1, 1, 1))
    )

    eval_mask = body_mask & (~exclusion_mask)

    return masked_mean(torch.abs(pred_mu - gt_mu), eval_mask)


def compute_mu_organ_bias(
    pred_ct,
    gt_ct,
    organ_seg,
    epsilon=1e-6,
):
    """CT organ-wise mean absolute relative error in 511-keV mu space."""
    pred_mu = hu_to_mu(pred_ct)
    gt_mu = hu_to_mu(gt_ct)

    bias_values = []

    for _, label in ORGAN_LABELS.items():
        mask = organ_seg == label

        if mask.sum() == 0:
            continue

        pred_mean = masked_mean(pred_mu, mask)
        gt_mean = masked_mean(gt_mu, mask)

        bias = 100.0 * torch.abs(pred_mean - gt_mean) / (torch.abs(gt_mean) + epsilon)
        bias_values.append(bias)

    if len(bias_values) == 0:
        raise ValueError("No valid organ mask found.")

    return torch.stack(bias_values).mean()


def compute_mu_brain_outlier_score(
    pred_ct,
    gt_ct,
    organ_seg,
    thresholds=(0.05, 0.10, 0.15),
    epsilon=1e-6,
):
    """CT brain outlier robustness score in 511-keV mu space."""
    pred_mu = hu_to_mu(pred_ct)
    gt_mu = hu_to_mu(gt_ct)

    brain_mask = organ_seg == BRAIN_LABEL
    valid_mask = brain_mask & (torch.abs(gt_mu) > epsilon)

    relative_error = torch.abs(pred_mu - gt_mu) / (torch.abs(gt_mu) + epsilon)

    k_values = []

    for threshold in thresholds:
        passed = (relative_error < threshold) & valid_mask

        numerator = passed.flatten(1).sum(dim=1)
        denominator = valid_mask.flatten(1).sum(dim=1)

        k = numerator / denominator.clamp_min(1)
        k = torch.where(denominator > 0, k, torch.zeros_like(k))

        k_values.append(k)

    k_values = torch.stack(k_values, dim=1)  # [B, num_thresholds]

    return 1.0 - k_values.mean()
