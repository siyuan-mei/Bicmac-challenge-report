from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from monai.inferers import SlidingWindowInferer


@dataclass(frozen=True)
class SegmentSpec:
    """One PET-derived axial segment and its output/inference boxes."""

    index: int
    core_z: tuple[int, int]
    output_box: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    inference_box: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    left_blend: bool
    right_blend: bool


@dataclass(frozen=True)
class BackgroundContext:
    """
    PET-derived segmented foreground context for production inference.

    The active PET extent is divided along Z into several contiguous segments.
    Every segment receives its own conservative XY bounding box. The output is
    initialized to ``background_value`` and only voxels covered by a segment are
    replaced by network predictions.
    """

    full_shape: tuple[int, int, int]
    segments: tuple[SegmentSpec, ...]
    background_value: float = 0.0
    segment_overlap_z: int = 24
    fov_diameter: float | None = 452.0
    used_full_volume_fallback: bool = False

    @staticmethod
    def _as_numpy_3d(pet: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(pet, torch.Tensor):
            pet = pet.detach().float().cpu().numpy()
        pet = np.asarray(pet, dtype=np.float32)
        if pet.ndim == 5:
            if pet.shape[0] != 1 or pet.shape[1] != 1:
                raise ValueError(f"Expected singleton B/C PET tensor, got {pet.shape}.")
            pet = pet[0, 0]
        elif pet.ndim == 4:
            if pet.shape[0] != 1:
                raise ValueError(f"Expected singleton PET channel, got {pet.shape}.")
            pet = pet[0]
        if pet.ndim != 3:
            raise ValueError(f"PET must be a 3D volume, got {pet.shape}.")
        return np.nan_to_num(pet, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _normalize_int_tuple(value: int | Sequence[int], length: int, name: str) -> tuple[int, ...]:
        if isinstance(value, (int, np.integer)):
            values = (int(value),) * length
        else:
            values = tuple(int(v) for v in value)
            if len(values) != length:
                raise ValueError(f"{name} must contain {length} values, got {values}.")
        if any(v < 0 for v in values):
            raise ValueError(f"{name} must be non-negative, got {values}.")
        return values

    @staticmethod
    def _aligned_interval(
        low: int,
        high: int,
        size: int,
        minimum_size: int,
        alignment: int,
    ) -> tuple[int, int]:
        low = max(0, int(low))
        high = min(int(size), int(high))
        if high <= low:
            low, high = 0, size

        minimum_size = min(max(int(minimum_size), 1), size)
        current = high - low
        if current < minimum_size:
            missing = minimum_size - current
            low -= missing // 2
            high += missing - missing // 2

        if alignment > 1:
            low = (low // alignment) * alignment
            high = int(math.ceil(high / alignment) * alignment)

        if low < 0:
            high -= low
            low = 0
        if high > size:
            low -= high - size
            high = size
        low = max(0, low)

        if high - low < minimum_size:
            if low == 0:
                high = min(size, minimum_size)
            else:
                low = max(0, size - minimum_size)
                high = size

        return int(low), int(high)

    @staticmethod
    def _quantile_interval(
        coordinates: np.ndarray,
        size: int,
        quantiles: tuple[float, float],
        margin: int,
        minimum_size: int,
        alignment: int,
    ) -> tuple[int, int]:
        if coordinates.size == 0:
            return BackgroundContext._aligned_interval(
                0, size, size, minimum_size, alignment
            )

        low_q, high_q = quantiles
        low = int(math.floor(float(np.quantile(coordinates, low_q)))) - margin
        high = int(math.ceil(float(np.quantile(coordinates, high_q)))) + margin + 1
        return BackgroundContext._aligned_interval(
            low, high, size, minimum_size, alignment
        )

    @classmethod
    def from_pet(
        cls,
        pet: np.ndarray | torch.Tensor,
        roi_size: Sequence[int],
        pet_threshold: float = 10.0,
        num_segments: int = 3,
        segment_xy_margins: int | Sequence[int] = (48, 64, 48),
        segment_overlap_z: int = 24,
        model_context_xy: int = 16,
        model_context_z: int = 16,
        global_z_margin: int = 16,
        alignment: int = 16,
        bbox_quantiles: Sequence[float] = (0.0005, 0.9995),
        min_z_plane_voxels: int = 32,
        min_segment_voxels: int = 128,
        background_value: float = 0.0,
        fov_diameter: float | None = 452.0,
    ) -> "BackgroundContext":
        pet = cls._as_numpy_3d(pet)
        full_shape = tuple(int(v) for v in pet.shape)
        roi_size = tuple(int(v) for v in roi_size)
        if len(roi_size) != 3 or any(v <= 0 for v in roi_size):
            raise ValueError(f"roi_size must contain three positive values, got {roi_size}.")
        if num_segments < 1:
            raise ValueError("num_segments must be >= 1.")
        if segment_overlap_z < 0:
            raise ValueError("segment_overlap_z must be non-negative.")
        if alignment < 1:
            raise ValueError("alignment must be >= 1.")

        quantiles = tuple(float(v) for v in bbox_quantiles)
        if len(quantiles) != 2 or not 0.0 <= quantiles[0] < quantiles[1] <= 1.0:
            raise ValueError(f"Invalid bbox_quantiles: {quantiles}.")

        if isinstance(segment_xy_margins, (int, np.integer)):
            xy_margins = (int(segment_xy_margins),) * num_segments
        else:
            xy_margins = tuple(int(v) for v in segment_xy_margins)
            if len(xy_margins) == 1:
                xy_margins = xy_margins * num_segments
            elif len(xy_margins) != num_segments:
                raise ValueError(
                    "segment_xy_margins must be a scalar or contain one value "
                    f"per segment, got {xy_margins}."
                )
        if any(v < 0 for v in xy_margins):
            raise ValueError(f"segment_xy_margins must be non-negative, got {xy_margins}.")

        active = pet > float(pet_threshold)
        z_counts = active.sum(axis=(0, 1))
        valid_z = np.flatnonzero(z_counts >= int(min_z_plane_voxels))
        if valid_z.size == 0:
            valid_z = np.flatnonzero(z_counts > 0)

        fallback = valid_z.size == 0
        if fallback:
            z_start, z_end = 0, full_shape[2]
            global_x = np.arange(full_shape[0], dtype=np.int64)
            global_y = np.arange(full_shape[1], dtype=np.int64)
        else:
            z_start = max(0, int(valid_z[0]) - int(global_z_margin))
            z_end = min(full_shape[2], int(valid_z[-1]) + int(global_z_margin) + 1)
            global_coords = np.nonzero(active[:, :, z_start:z_end])
            global_x = global_coords[0]
            global_y = global_coords[1]

        z_start, z_end = cls._aligned_interval(
            z_start,
            z_end,
            full_shape[2],
            min(roi_size[2], full_shape[2]),
            alignment,
        )

        boundaries = np.rint(
            np.linspace(z_start, z_end, num_segments + 1)
        ).astype(np.int64)
        boundaries[0] = z_start
        boundaries[-1] = z_end
        for index in range(1, len(boundaries)):
            boundaries[index] = max(boundaries[index], boundaries[index - 1] + 1)
        boundaries[-1] = z_end

        segments: list[SegmentSpec] = []
        for index in range(num_segments):
            core_start = int(boundaries[index])
            core_end = int(boundaries[index + 1])
            if core_end <= core_start:
                continue

            bbox_z_start = max(z_start, core_start - segment_overlap_z)
            bbox_z_end = min(z_end, core_end + segment_overlap_z)
            coords = np.nonzero(active[:, :, bbox_z_start:bbox_z_end])
            if coords[0].size < int(min_segment_voxels):
                xs, ys = global_x, global_y
            else:
                xs, ys = coords[0], coords[1]

            margin = xy_margins[index]
            x_out = cls._quantile_interval(
                xs,
                full_shape[0],
                quantiles,
                margin,
                min(roi_size[0], full_shape[0]),
                alignment,
            )
            y_out = cls._quantile_interval(
                ys,
                full_shape[1],
                quantiles,
                margin,
                min(roi_size[1], full_shape[1]),
                alignment,
            )

            z_out = (
                max(z_start, core_start - (segment_overlap_z if index > 0 else 0)),
                min(z_end, core_end + (segment_overlap_z if index < num_segments - 1 else 0)),
            )
            z_out = cls._aligned_interval(
                z_out[0],
                z_out[1],
                full_shape[2],
                min(roi_size[2], full_shape[2]),
                alignment,
            )

            x_in = cls._aligned_interval(
                x_out[0] - int(model_context_xy),
                x_out[1] + int(model_context_xy),
                full_shape[0],
                min(roi_size[0], full_shape[0]),
                alignment,
            )
            y_in = cls._aligned_interval(
                y_out[0] - int(model_context_xy),
                y_out[1] + int(model_context_xy),
                full_shape[1],
                min(roi_size[1], full_shape[1]),
                alignment,
            )
            z_in = cls._aligned_interval(
                z_out[0] - int(model_context_z),
                z_out[1] + int(model_context_z),
                full_shape[2],
                min(roi_size[2], full_shape[2]),
                alignment,
            )

            segments.append(
                SegmentSpec(
                    index=index,
                    core_z=(core_start, core_end),
                    output_box=(x_out, y_out, z_out),
                    inference_box=(x_in, y_in, z_in),
                    left_blend=index > 0,
                    right_blend=index < num_segments - 1,
                )
            )

        if not segments:
            segments = [
                SegmentSpec(
                    index=0,
                    core_z=(0, full_shape[2]),
                    output_box=((0, full_shape[0]), (0, full_shape[1]), (0, full_shape[2])),
                    inference_box=((0, full_shape[0]), (0, full_shape[1]), (0, full_shape[2])),
                    left_blend=False,
                    right_blend=False,
                )
            ]
            fallback = True

        return cls(
            full_shape=full_shape,
            segments=tuple(segments),
            background_value=float(background_value),
            segment_overlap_z=int(segment_overlap_z),
            fov_diameter=None if fov_diameter is None else float(fov_diameter),
            used_full_volume_fallback=bool(fallback),
        )

    def describe(self) -> str:
        lines = [
            "PET-segmented background context:",
            f"  full_shape={self.full_shape}",
            f"  fallback_full_volume={self.used_full_volume_fallback}",
        ]
        for segment in self.segments:
            lines.append(
                "  segment "
                f"{segment.index + 1}/{len(self.segments)}: "
                f"core_z={segment.core_z}, "
                f"output={segment.output_box}, "
                f"inference={segment.inference_box}"
            )
        return "\n".join(lines)


class BackgroundAwareSlidingWindowInferer:
    """
    Run standard MONAI sliding-window inference inside PET-derived segments.

    Segment overlap is blended only between actual network predictions. Voxels
    not covered by any PET-derived segment remain ``background_value``. No
    additional cylindrical/body foreground mask is applied after stitching:
    BIC-MAC replaces face/bed/far-background regions with ground-truth CT before
    attenuation reconstruction, while an extra post-hoc mask can accidentally
    erase valid near-body attenuation inside the official prediction region.

    ``BackgroundContext.fov_diameter`` is retained only for caller/API
    compatibility and is intentionally ignored here.
    """

    def __init__(
        self,
        roi_size: Sequence[int],
        sw_batch_size: int = 1,
        overlap: float = 0.25,
        mode: str = "constant",
        sigma_scale: float = 0.125,
        cache_roi_weight_map: bool = False,
        sw_device: torch.device | str | None = None,
        device: torch.device | str | None = "cpu",
        progress: bool = False,
    ):
        self.roi_size = tuple(int(v) for v in roi_size)
        self.output_device = torch.device(device or "cpu")
        self.inferer = SlidingWindowInferer(
            roi_size=self.roi_size,
            sw_batch_size=int(sw_batch_size),
            overlap=float(overlap),
            mode=str(mode),
            sigma_scale=float(sigma_scale),
            cache_roi_weight_map=bool(cache_roi_weight_map),
            sw_device=sw_device,
            device=self.output_device,
            progress=bool(progress),
        )

    @staticmethod
    def _slices(box: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]) -> tuple[slice, ...]:
        return tuple(slice(start, end) for start, end in box)

    @staticmethod
    def _relative_slices(
        inner: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
        outer: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    ) -> tuple[slice, ...]:
        slices = []
        for (inner_start, inner_end), (outer_start, outer_end) in zip(inner, outer):
            if inner_start < outer_start or inner_end > outer_end:
                raise ValueError(f"Inner box {inner} is not contained in outer box {outer}.")
            slices.append(slice(inner_start - outer_start, inner_end - outer_start))
        return tuple(slices)

    @staticmethod
    def _segment_z_weight(segment: SegmentSpec, overlap: int, device: torch.device) -> torch.Tensor:
        z_start, z_end = segment.output_box[2]
        core_start, core_end = segment.core_z
        z = torch.arange(z_start, z_end, dtype=torch.float32, device=device)
        weight = torch.ones_like(z)

        if segment.left_blend and overlap > 0:
            blend_start = core_start - overlap
            blend_end = core_start + overlap
            t = ((z - blend_start) / max(blend_end - blend_start, 1)).clamp(0.0, 1.0)
            left = 0.5 - 0.5 * torch.cos(math.pi * t)
            weight = weight * left

        if segment.right_blend and overlap > 0:
            blend_start = core_end - overlap
            blend_end = core_end + overlap
            t = ((z - blend_start) / max(blend_end - blend_start, 1)).clamp(0.0, 1.0)
            right = 0.5 + 0.5 * torch.cos(math.pi * t)
            weight = weight * right

        return weight.view(1, 1, 1, 1, -1)

    @staticmethod
    def _cylindrical_fov(
        shape: tuple[int, int, int],
        diameter: float,
        device: torch.device,
    ) -> torch.Tensor:
        x_size, y_size, _ = shape
        radius = min(float(diameter) * 0.5, 0.5 * min(x_size - 1, y_size - 1))
        cx = 0.5 * (x_size - 1)
        cy = 0.5 * (y_size - 1)
        xx = torch.arange(x_size, device=device, dtype=torch.float32)[:, None]
        yy = torch.arange(y_size, device=device, dtype=torch.float32)[None, :]
        return (((xx - cx) ** 2 + (yy - cy) ** 2) <= radius ** 2)[None, None, :, :, None]

    def __call__(
        self,
        inputs: torch.Tensor,
        network,
        context: BackgroundContext,
    ) -> torch.Tensor:
        if inputs.ndim != 5:
            raise ValueError(f"Expected inputs [B,C,X,Y,Z], got {tuple(inputs.shape)}.")
        if inputs.shape[0] != 1:
            raise ValueError(
                "BackgroundAwareSlidingWindowInferer currently expects batch size 1 "
                f"for subject-wise production inference, got {inputs.shape[0]}."
            )
        if tuple(inputs.shape[2:]) != context.full_shape:
            raise ValueError(
                f"Input shape {tuple(inputs.shape[2:])} does not match context {context.full_shape}."
            )

        output_sum = None
        weight_sum = None

        print(context.describe(), flush=True)

        for segment in context.segments:
            inference_slices = self._slices(segment.inference_box)
            crop = inputs[(slice(None), slice(None), *inference_slices)]
            prediction = self.inferer(crop, network)
            if isinstance(prediction, (tuple, list)):
                prediction = prediction[0]
            prediction = prediction.float().to(self.output_device)

            relative = self._relative_slices(segment.output_box, segment.inference_box)
            prediction = prediction[(slice(None), slice(None), *relative)]

            if output_sum is None:
                output_shape = (inputs.shape[0], prediction.shape[1], *context.full_shape)
                output_sum = torch.zeros(output_shape, dtype=torch.float32, device=self.output_device)
                weight_sum = torch.zeros(
                    (inputs.shape[0], 1, *context.full_shape),
                    dtype=torch.float32,
                    device=self.output_device,
                )

            weight = self._segment_z_weight(
                segment,
                context.segment_overlap_z,
                self.output_device,
            )
            output_slices = self._slices(segment.output_box)
            output_sum[(slice(None), slice(None), *output_slices)] += prediction * weight
            weight_sum[(slice(None), slice(None), *output_slices)] += weight

        if output_sum is None or weight_sum is None:
            return torch.full(
                (inputs.shape[0], 1, *context.full_shape),
                fill_value=context.background_value,
                dtype=torch.float32,
                device=self.output_device,
            )

        prediction = torch.where(
            weight_sum > 0,
            output_sum / weight_sum.clamp_min(1.0e-6),
            torch.full_like(output_sum, context.background_value),
        )

        # Intentionally no final FOV/body mask here. The official
        # reconstruction replaces face/bed/far-background regions itself.
        # Applying another foreground mask at this point can erase valid
        # attenuation within the prediction-controlled near-body rim.
        return prediction
