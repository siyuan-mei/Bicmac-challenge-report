from __future__ import annotations

import csv
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi


DEFAULT_BICMAC_ROOT = Path("data/bicmac/bicmac_data")
DEFAULT_EXTRA_ROOT = Path("data/bicmac/extra_data")
DEFAULT_TARGET_SPACING = (1.5234375, 1.5234375, 2.0)
DEFAULT_ORIENTATION = "RAS"

_ORIENTATION_TO_AXIS_SIGN = {
    "L": (0, 1),
    "R": (0, -1),
    "P": (1, 1),
    "A": (1, -1),
    "S": (2, 1),
    "I": (2, -1),
}
_AXIS_SIGN_TO_ORIENTATION = {
    value: key for key, value in _ORIENTATION_TO_AXIS_SIGN.items()
}


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    status: str
    output_dir: Path
    message: str = ""


def parse_spacing(value: str | None) -> tuple[float, float, float] | None:
    if value is None:
        return None

    parts = re.split(r"[x, ]+", value.strip())
    parts = [p for p in parts if p]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 spacing values, got {value!r}")

    spacing = tuple(float(p) for p in parts)
    if any(v <= 0 for v in spacing):
        raise ValueError(f"Spacing values must be positive, got {spacing}")

    return spacing  # type: ignore[return-value]


def sanitize_case_name(name: str) -> str:
    clean = re.sub(r'[<>:"/\\|?*]+', "_", name.strip())
    clean = re.sub(r"\s+", " ", clean).strip(" .")
    return clean or "case"


def iter_bicmac_ct_paths(bicmac_root: Path) -> list[Path]:
    return (
        sorted(bicmac_root.glob("train/*/ct-label/ct.nii.gz"))
        + sorted(bicmac_root.glob("validation/*/ct-label/ct.nii.gz"))
    )


def read_header(path: Path) -> sitk.ImageFileReader:
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    return reader


def median_bicmac_spacing(
    bicmac_root: Path = DEFAULT_BICMAC_ROOT,
    fallback: Sequence[float] = DEFAULT_TARGET_SPACING,
) -> tuple[float, float, float]:
    paths = iter_bicmac_ct_paths(bicmac_root)
    if not paths:
        return tuple(float(v) for v in fallback)  # type: ignore[return-value]

    spacings = []
    for path in paths:
        spacings.append(read_header(path).GetSpacing())

    return tuple(
        float(statistics.median([spacing[axis] for spacing in spacings]))
        for axis in range(3)
    )


def resolve_target_spacing(
    explicit_spacing: tuple[float, float, float] | None,
    bicmac_root: Path,
) -> tuple[float, float, float]:
    if explicit_spacing is not None:
        return explicit_spacing
    return median_bicmac_spacing(bicmac_root)


def orientation_code(image: sitk.Image) -> str:
    return orientation_code_from_direction(image.GetDirection())


def orientation_code_from_direction(direction: Sequence[float]) -> str:
    if hasattr(sitk, "DICOMOrientImageFilter_GetOrientationFromDirectionCosines"):
        return sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(direction)

    matrix = np.asarray(direction, dtype=float).reshape(3, 3)
    labels = []
    used_axes: set[int] = set()
    for image_axis in range(3):
        column = matrix[:, image_axis]
        candidates = np.argsort(np.abs(column))[::-1]
        physical_axis = int(next(axis for axis in candidates if int(axis) not in used_axes))
        used_axes.add(physical_axis)
        sign = 1 if float(column[physical_axis]) >= 0 else -1
        labels.append(_AXIS_SIGN_TO_ORIENTATION[(physical_axis, sign)])

    return "".join(labels)


def orient_image(image: sitk.Image, orientation: str) -> sitk.Image:
    orientation = orientation.upper()
    if hasattr(sitk, "DICOMOrient"):
        return sitk.DICOMOrient(image, orientation)

    if image.GetDimension() != 3:
        raise ValueError(f"Expected a 3D image, got dimension {image.GetDimension()}")
    if len(orientation) != 3 or any(label not in _ORIENTATION_TO_AXIS_SIGN for label in orientation):
        raise ValueError(f"Unsupported orientation code {orientation!r}")

    current = orientation_code(image)
    if current == orientation:
        return image

    current_axis_map = {
        _ORIENTATION_TO_AXIS_SIGN[label][0]: (
            image_axis,
            _ORIENTATION_TO_AXIS_SIGN[label][1],
        )
        for image_axis, label in enumerate(current)
    }

    order = []
    flip_axes = []
    for label in orientation:
        physical_axis, desired_sign = _ORIENTATION_TO_AXIS_SIGN[label]
        image_axis, current_sign = current_axis_map[physical_axis]
        order.append(image_axis)
        flip_axes.append(current_sign != desired_sign)

    oriented = image
    if order != [0, 1, 2]:
        oriented = sitk.PermuteAxes(oriented, order)
    if any(flip_axes):
        oriented = sitk.Flip(oriented, flip_axes, False)

    desired_direction = np.zeros((3, 3), dtype=float)
    for image_axis, label in enumerate(orientation):
        physical_axis, sign = _ORIENTATION_TO_AXIS_SIGN[label]
        desired_direction[physical_axis, image_axis] = float(sign)
    oriented.SetDirection(tuple(float(v) for v in desired_direction.ravel()))

    return oriented


def compute_resampled_size(
    size: Sequence[int],
    spacing: Sequence[float],
    target_spacing: Sequence[float],
) -> tuple[int, int, int]:
    new_size = []
    for n_voxels, src_spacing, dst_spacing in zip(size, spacing, target_spacing):
        if n_voxels <= 1:
            new_size.append(1)
            continue
        extent = (int(n_voxels) - 1) * float(src_spacing)
        new_size.append(max(1, int(round(extent / float(dst_spacing))) + 1))
    return tuple(new_size)  # type: ignore[return-value]


def make_reference_image(
    image: sitk.Image,
    target_spacing: Sequence[float],
) -> sitk.Image:
    target_spacing = tuple(float(v) for v in target_spacing)
    target_size = compute_resampled_size(
        image.GetSize(),
        image.GetSpacing(),
        target_spacing,
    )

    reference = sitk.Image(target_size, sitk.sitkFloat32)
    reference.SetOrigin(image.GetOrigin())
    reference.SetDirection(image.GetDirection())
    reference.SetSpacing(target_spacing)
    return reference


def center_crop_xy(
    image: sitk.Image,
    start_x: int,
    start_y: int,
    size_x: int,
    size_y: int,
) -> sitk.Image:
    size = image.GetSize()
    return sitk.RegionOfInterest(
        image,
        [int(size_x), int(size_y), int(size[2])],
        [int(start_x), int(start_y), 0],
    )


def estimate_center_fov_crop(
    image: sitk.Image,
    padding_value: float | None = None,
    tolerance: float = 1.0,
    margin: int = 0,
) -> dict[str, float | int | bool]:
    array = sitk.GetArrayViewFromImage(image)
    height, width = int(array.shape[1]), int(array.shape[2])
    corner = max(4, min(20, int(round(min(height, width) * 0.03))))

    if padding_value is None:
        corners = np.concatenate(
            [
                np.asarray(array[:, :corner, :corner]).ravel(),
                np.asarray(array[:, :corner, -corner:]).ravel(),
                np.asarray(array[:, -corner:, :corner]).ravel(),
                np.asarray(array[:, -corner:, -corner:]).ravel(),
            ]
        )
        padding_value = float(np.nanmedian(corners))

    projection = np.any(np.abs(array - float(padding_value)) > float(tolerance), axis=0)
    ys, xs = np.where(projection)
    if xs.size == 0:
        side = min(width, height)
    else:
        center_x = (width - 1) / 2.0
        center_y = (height - 1) / 2.0
        radius = max(
            center_x - float(xs.min()),
            float(xs.max()) - center_x,
            center_y - float(ys.min()),
            float(ys.max()) - center_y,
        )
        side = int(2 * np.ceil(radius)) + int(2 * margin)
        side = max(1, min(side, width, height))

    start_x = int((width - side) // 2)
    start_y = int((height - side) // 2)
    return {
        "applied": bool(side < min(width, height)),
        "padding_value": float(padding_value),
        "tolerance": float(tolerance),
        "start_x": start_x,
        "start_y": start_y,
        "size_x": side,
        "size_y": side,
        "orig_size_x": width,
        "orig_size_y": height,
    }


def center_crop_to_fov(
    image: sitk.Image,
    padding_value: float | None = None,
    tolerance: float = 1.0,
    margin: int = 0,
) -> tuple[sitk.Image, dict[str, float | int | bool]]:
    crop = estimate_center_fov_crop(
        image,
        padding_value=padding_value,
        tolerance=tolerance,
        margin=margin,
    )
    if not crop["applied"]:
        return image, crop

    return (
        center_crop_xy(
            image,
            int(crop["start_x"]),
            int(crop["start_y"]),
            int(crop["size_x"]),
            int(crop["size_y"]),
        ),
        crop,
    )


def clip_image(
    image: sitk.Image,
    lower: float,
    upper: float,
    output_pixel_type: int = sitk.sitkFloat32,
) -> sitk.Image:
    return sitk.Cast(
        sitk.Clamp(image, lowerBound=float(lower), upperBound=float(upper)),
        output_pixel_type,
    )


def make_body_mask_from_ct(
    ct_image: sitk.Image,
    threshold: float = -500.0,
) -> sitk.Image:
    ct = sitk.GetArrayFromImage(ct_image)
    tissue = np.isfinite(ct) & (ct > float(threshold))

    labels, n_labels = ndi.label(tissue, structure=ndi.generate_binary_structure(3, 1))
    mask = np.zeros(tissue.shape, dtype=bool)
    if n_labels > 0:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        body_label = int(np.argmax(sizes))
        mask = labels == body_label

    filled = np.zeros(mask.shape, dtype=np.uint8)
    for z_index in range(mask.shape[0]):
        if mask[z_index].any():
            filled[z_index] = ndi.binary_fill_holes(mask[z_index]).astype(np.uint8)

    body = sitk.GetImageFromArray(filled)
    body.CopyInformation(ct_image)
    return body


def read_image(path: Path, pixel_type: int = sitk.sitkFloat32) -> sitk.Image:
    return sitk.ReadImage(str(path), pixel_type)


def resample_to_reference(
    image: sitk.Image,
    reference: sitk.Image,
    interpolator: int,
    default_value: float,
    output_pixel_type: int,
) -> sitk.Image:
    return sitk.Resample(
        image,
        reference,
        sitk.Transform(3, sitk.sitkIdentity),
        interpolator,
        float(default_value),
        output_pixel_type,
    )


def build_reference_from_path(
    path: Path,
    target_spacing: Sequence[float],
    orientation: str,
    pixel_type: int = sitk.sitkFloat32,
) -> tuple[sitk.Image, sitk.Image]:
    image = orient_image(read_image(path, pixel_type), orientation)
    reference = make_reference_image(image, target_spacing)
    return image, reference


def load_oriented_resampled(
    path: Path,
    reference: sitk.Image,
    orientation: str,
    interpolator: int,
    default_value: float,
    output_pixel_type: int,
    input_pixel_type: int = sitk.sitkFloat32,
) -> sitk.Image:
    image = orient_image(read_image(path, input_pixel_type), orientation)
    return resample_to_reference(
        image,
        reference,
        interpolator,
        default_value,
        output_pixel_type,
    )


def write_image(image: sitk.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(output_path), True)


def outputs_exist(paths: Iterable[Path]) -> bool:
    return all(path.exists() for path in paths)


def write_manifest(manifest_path: Path, rows: Iterable[dict[str, str]]) -> None:
    rows = list(rows)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(results: Sequence[CaseResult]) -> None:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    summary = ", ".join(f"{key}={counts[key]}" for key in sorted(counts))
    print(f"Summary: total={len(results)}" + (f", {summary}" if summary else ""))
