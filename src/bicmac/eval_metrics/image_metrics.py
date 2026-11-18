import torch
from monai.metrics.regression import (
    PSNRMetric,
    SSIMMetric,
    MultiScaleSSIMMetric,
)

from .bicmac_metrics import (
    compute_mu_mae,
    compute_mu_organ_bias,
    compute_mu_brain_outlier_score,
    compute_reconstruction_mu_metrics,
)


psnr_metric = PSNRMetric(max_val=1.0)
ssim_metric = SSIMMetric(spatial_dims=3)
ms_ssim_metric = MultiScaleSSIMMetric(spatial_dims=3, weights=[0.3, 0.5, 0.2])


def safe_metric(metric, y_pred, y):
    try:
        return metric(y_pred, y).mean().item()
    except ValueError:
        return float("nan")


def hu_to_norm(ct_img):
    ct_img = torch.clamp(ct_img, min=-1024.0, max=3071.0)
    return (ct_img + 1024.0) / 4095.0


def crop_by_mask(x, mask, margin=8):
    """
    x:    [B, C, X, Y, Z]
    mask: [B, C, X, Y, Z]
    """
    pos = torch.where(mask > 0)

    if pos[0].numel() == 0:
        return x

    x0 = max(int(pos[2].min()) - margin, 0)
    x1 = min(int(pos[2].max()) + 1 + margin, x.shape[2])

    y0 = max(int(pos[3].min()) - margin, 0)
    y1 = min(int(pos[3].max()) + 1 + margin, x.shape[3])

    z0 = max(int(pos[4].min()) - margin, 0)
    z1 = min(int(pos[4].max()) + 1 + margin, x.shape[4])

    return x[:, :, x0:x1, y0:y1, z0:z1]


@torch.no_grad()
def evaluate(
    y_pred,
    y,
    body_mask=None,
    organ_mask=None,
    prediction_mask=None,
    spacing_mm=(1.5234375, 1.5234375, 2.0),
    smooth_fwhm_mm=4.0,
    gaussian_truncate=4.0,
):
    """
    Validation metrics for CT quality and PET-reconstruction-aligned mu quality.

    Parameters
    ----------
    y_pred, y:
        [B,C,X,Y,Z] in HU.
    body_mask:
        Challenge body segmentation used by legacy/body CT metrics.
    organ_mask:
        Organ labels used by challenge-style CT organ/brain diagnostics.
    prediction_mask:
        Region for which the submitted sCT remains responsible after the
        official face/bed/far-background GT replacement. This is intentionally
        NOT intersected with body_mask for the reconstruction-aligned metrics.

    Important
    ---------
    The reconstruction metrics are computed from the untouched HU volumes
    before any body masking. This preserves the near-body air rim that can still
    affect attenuation correction.
    """
    pred_ct = y_pred.float()
    gt_ct = y.float()

    results = {}

    # PET-reconstruction-aligned image-domain metrics. These operate on the
    # actual prediction responsibility region, including near-body air, and on
    # the 4-mm-smoothed mu difference that is handed toward STIR projection.
    if prediction_mask is not None:
        recon = compute_reconstruction_mu_metrics(
            pred_ct,
            gt_ct,
            prediction_mask=prediction_mask,
            spacing_mm=spacing_mm,
            fwhm_mm=smooth_fwhm_mm,
            truncate=gaussian_truncate,
        )
        results.update(
            {
                # Primary new checkpoint-selection metric.
                "1b-smooth_mu_mae": recon["smooth_mu_mae"].item(),
                # Raw responsibility-region mu error, useful to monitor CT/mu.
                "1c-recon_mu_mae": recon["recon_mu_mae"].item(),
                # Systematic attenuation bias diagnostics. Use abs for ranking;
                # keep signed value to diagnose over/under attenuation.
                "1d-smooth_mu_bias_abs": recon["smooth_mu_bias_abs"].item(),
                "1e-smooth_mu_bias_signed": recon["smooth_mu_bias_signed"].item(),
                "1f-smooth_mu_coherence": recon["smooth_mu_coherence"].item(),
                "1g-smooth_mu_rmse": recon["smooth_mu_rmse"].item(),
                "1h-recon_mu_bias_abs": recon["recon_mu_bias_abs"].item(),
                "1i-recon_mu_bias_signed": recon["recon_mu_bias_signed"].item(),
            }
        )

    # Legacy/challenge-style CT metrics remain available. Keep a separate body-
    # masked copy only for these metrics; never feed this masked copy into the
    # reconstruction-aligned metrics above.
    if body_mask is None:
        body_mask_bool = torch.ones_like(gt_ct, dtype=torch.bool)
    else:
        body_mask_bool = body_mask > 0

    hu_err = pred_ct[body_mask_bool] - gt_ct[body_mask_bool]

    pred_body = torch.where(
        body_mask_bool,
        pred_ct,
        torch.full_like(pred_ct, -1024.0),
    )
    gt_body = torch.where(
        body_mask_bool,
        gt_ct,
        torch.full_like(gt_ct, -1024.0),
    )

    pred_norm = hu_to_norm(pred_body)
    gt_norm = hu_to_norm(gt_body)
    pred_crop = crop_by_mask(pred_norm, body_mask_bool)
    gt_crop = crop_by_mask(gt_norm, body_mask_bool)

    results.update(
        {
            "4-mae_hu": torch.abs(hu_err).mean().item(),
            "5-mse_hu": torch.square(hu_err).mean().item(),
            "6-psnr_hu": safe_metric(psnr_metric, pred_crop, gt_crop),
            "7-ssim_hu": safe_metric(ssim_metric, pred_crop, gt_crop),
            "8-ms_ssim_hu": safe_metric(ms_ssim_metric, pred_crop, gt_crop),
        }
    )

    if organ_mask is not None:
        results.update(
            {
                "1-mae_mu": compute_mu_mae(
                    pred_ct,
                    gt_ct,
                    body_mask_bool,
                    organ_mask,
                ).item(),
                "2-organ_bias_mu": compute_mu_organ_bias(
                    pred_ct,
                    gt_ct,
                    organ_mask,
                ).item(),
                "3-brain_outlier_mu": compute_mu_brain_outlier_score(
                    pred_ct,
                    gt_ct,
                    organ_mask,
                ).item(),
            }
        )

    return results
