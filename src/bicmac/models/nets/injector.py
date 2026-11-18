import torch
import torch.nn.functional as F
from torch import nn


class ConditionPyramid3D(nn.Module):
    """
    Encode auxiliary conditions into a 3D feature pyramid.

    Output scales:
        2  -> 1/2
        4  -> 1/4
        8  -> 1/8
        16 -> 1/16
    """

    def __init__(
        self,
        in_channels,
        channels=(64, 128, 256, 512),
    ):
        super().__init__()

        channels = tuple(int(c) for c in channels)

        if len(channels) != 4:
            raise ValueError(
                "condition_channels must contain four values "
                "for scales 2, 4, 8, and 16."
            )

        def block(in_ch, out_ch, stride):
            return nn.Sequential(
                nn.Conv3d(
                    in_ch,
                    out_ch,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                ),
                nn.InstanceNorm3d(
                    out_ch,
                    affine=True,
                ),
                nn.GELU(),
            )

        # 1 -> 1/2
        self.stem = nn.Sequential(
            block(
                in_channels,
                channels[0],
                stride=2,
            ),
            block(
                channels[0],
                channels[0],
                stride=1,
            ),
            block(
                channels[0],
                channels[0],
                stride=1,
            ),
        )

        # 1/2 -> 1/4 -> 1/8 -> 1/16
        self.downs = nn.ModuleList([
            block(
                channels[i],
                channels[i + 1],
                stride=2,
            )
            for i in range(3)
        ])

    def forward(self, x):
        x2 = self.stem(x)
        x4 = self.downs[0](x2)
        x8 = self.downs[1](x4)
        x16 = self.downs[2](x8)

        return {
            2: x2,
            4: x4,
            8: x8,
            16: x16,
        }


class ZeroSPADE3D(nn.Module):
    """
    Identity-initialized spatial modulation.

    A single 1x1x1 convolution predicts:

        [scale_delta, shift, gate_delta]

    with C channels for each component.

    after_norm:
        y = x * (1 + scale_delta) + shift

    after_mlp:
        gate = 1 + gate_delta
        y = x * gate

    Zero initialization gives:

        scale_delta = 0
        shift       = 0
        gate_delta  = 0

    therefore:

        effective scale = 1
        effective gate  = 1

    so the pretrained backbone is exactly preserved at initialization.
    """

    def __init__(
        self,
        feature_channels,
        condition_channels,
        scale_limit=0.5,
        gate_limit=0.5,
    ):
        super().__init__()

        self.feature_channels = int(feature_channels)

        self.scale_limit = scale_limit
        self.gate_limit = gate_limit

        self.proj = nn.Conv3d(
            int(condition_channels),
            3 * self.feature_channels,
            kernel_size=1,
            bias=True,
        )

        self.reset_zero()

    def reset_zero(self):
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def _project(self, features, condition):
        if condition.shape[2:] != features.shape[2:]:
            condition = F.interpolate(
                condition,
                size=features.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

        scale_delta, shift, gate_delta = self.proj(
            condition
        ).chunk(3, dim=1)

        # Limit multiplicative modulation while preserving
        # exact identity initialization.
        # if self.scale_limit is not None:
        #     scale_delta = (
        #         float(self.scale_limit)
        #         * torch.tanh(scale_delta)
        #     )
        #
        # if self.gate_limit is not None:
        #     gate_delta = (
        #         float(self.gate_limit)
        #         * torch.tanh(gate_delta)
        #     )

        return scale_delta, shift, gate_delta

    def after_norm(
        self,
        features,
        condition,
        present=None,
    ):
        """
        features:
            [B, C, D, H, W]

        condition:
            [B, C_cond, d, h, w]

        present:
            [B, 1, 1, 1, 1]

        Returns:
            features * (1 + scale_delta) + shift
        """

        # No condition for this stage.
        if condition is None:
            return features

        scale_delta, shift, _ = self._project(
            features,
            condition,
        )

        if present is not None:
            present = present.to(
                device=features.device,
                dtype=features.dtype,
            )

            scale_delta = scale_delta * present
            shift = shift * present

        return (
            features * (1.0 + scale_delta)
            + shift
        )

    def after_mlp(
        self,
        features,
        condition,
        present=None,
    ):
        """
        Returns:
            features * gate

        where:
            gate = 1 + gate_delta
        """

        if condition is None:
            return features

        _, _, gate_delta = self._project(
            features,
            condition,
        )

        if present is not None:
            present = present.to(
                device=features.device,
                dtype=features.dtype,
            )

            gate_delta = gate_delta * present

        gate = 1.0 + gate_delta

        return features * gate


class SpatialConditionInjector(nn.Module):
    """
    Spatial condition injector compatible with the existing MedNeXtv3.

    Examples
    --------

    PET only:

        input_keys = ("pet",)
        primary_keys = ("pet",)

    In this case:
        encoder = None
        blocks = empty
        encode_inputs() returns:
            primary, ({}, None)

    and the MedNeXt backbone behaves exactly as the original model.


    PET + MR + topogram:

        input_keys = (
            "pet",
            "mr_in",
            "mr_out",
            "topogram",
        )

        primary_keys = ("pet",)

    Then:
        primary   = PET
        condition = MR_in + MR_out + topogram

    Conditions are injected at scales:
        2, 4, 8, 16

    Missing condition channels can simply be filled with zeros.
    """

    def __init__(
        self,
        input_keys,
        feature_channels_by_scale,
        depth_by_scale,
        primary_keys=("pet",),
        condition_channels=(64, 128, 256, 512),
        scale_limit=0.5,
        gate_limit=0.5,
    ):
        super().__init__()

        self.input_keys = tuple(input_keys)
        self.primary_keys = tuple(primary_keys)

        if len(self.input_keys) == 0:
            raise ValueError(
                "input_keys cannot be empty."
            )

        if len(self.primary_keys) == 0:
            raise ValueError(
                "primary_keys cannot be empty."
            )

        if not set(self.primary_keys).issubset(
            self.input_keys
        ):
            raise ValueError(
                "Every primary key must be present "
                "in input_keys."
            )

        # Keep the current MR design:
        # mr_in and mr_out must appear together.
        if (
            ("mr_in" in self.input_keys)
            != ("mr_out" in self.input_keys)
        ):
            raise ValueError(
                "MR conditioning requires both "
                "mr_in and mr_out."
            )

        self.primary_indices = tuple(
            self.input_keys.index(key)
            for key in self.primary_keys
        )

        self.condition_keys = tuple(
            key
            for key in self.input_keys
            if key not in self.primary_keys
        )

        self.condition_indices = tuple(
            self.input_keys.index(key)
            for key in self.condition_keys
        )

        self.has_condition = (
            len(self.condition_indices) > 0
        )

        # ---------------------------------------------------------
        # No auxiliary conditions
        # ---------------------------------------------------------

        if not self.has_condition:
            self.encoder = None
            self.blocks = nn.ModuleDict()
            return

        # ---------------------------------------------------------
        # Auxiliary condition branch
        # ---------------------------------------------------------

        condition_channels = tuple(
            int(c)
            for c in condition_channels
        )

        if len(condition_channels) != 4:
            raise ValueError(
                "condition_channels must contain four "
                "values for scales 2, 4, 8, and 16."
            )

        self.condition_scales = (2, 4, 8, 16)

        for scale in self.condition_scales:
            if scale not in feature_channels_by_scale:
                raise ValueError(
                    f"Backbone does not expose scale {scale}."
                )

            if scale not in depth_by_scale:
                raise ValueError(
                    f"depth_by_scale does not expose "
                    f"scale {scale}."
                )

        self.encoder = ConditionPyramid3D(
            in_channels=len(
                self.condition_indices
            ),
            channels=condition_channels,
        )

        self.blocks = nn.ModuleDict()

        for scale, cond_channels in zip(
            self.condition_scales,
            condition_channels,
        ):
            self.blocks[str(scale)] = nn.ModuleList([
                ZeroSPADE3D(
                    feature_channels=int(
                        feature_channels_by_scale[scale]
                    ),
                    condition_channels=int(
                        cond_channels
                    ),
                    scale_limit=scale_limit,
                    gate_limit=gate_limit,
                )
                for _ in range(
                    int(depth_by_scale[scale])
                )
            ])

    def encode_inputs(self, inputs):
        """
        Compatible with:

            x, state = injector.encode_inputs(x)

        where state is always:

            (contexts, present)

        so your MedNeXtv3 forward_features()
        does not need any modification.
        """

        if inputs.shape[1] != len(self.input_keys):
            raise ValueError(
                f"Expected {len(self.input_keys)} "
                f"input channels according to input_keys, "
                f"but received {inputs.shape[1]}."
            )

        # Primary modality / modalities
        primary = torch.cat(
            [
                inputs[:, index:index + 1]
                for index in self.primary_indices
            ],
            dim=1,
        )

        # PET-only / no auxiliary condition.
        if not self.has_condition:
            return primary, ({}, None)

        # Auxiliary modalities.
        condition = torch.cat(
            [
                inputs[:, index:index + 1]
                for index in self.condition_indices
            ],
            dim=1,
        )

        # One availability flag per sample:
        #
        # [B, 1, 1, 1, 1]
        #
        # If MR/topogram are all missing and represented
        # by zero tensors, present becomes 0.
        present = (
            condition
            .abs()
            .amax(
                dim=tuple(
                    range(1, condition.ndim)
                ),
                keepdim=True,
            )
            > 1.0e-6
        ).to(condition.dtype)

        contexts = self.encoder(condition)

        # Important:
        # InstanceNorm has affine=True, so a zero condition
        # could become non-zero after training.
        #
        # Explicitly masking contexts guarantees:
        #
        #     missing conditions -> zero context
        #
        # even after the condition encoder has learned.
        contexts = {
            scale: context * present
            for scale, context
            in contexts.items()
        }

        return primary, (
            contexts,
            present,
        )

    def modulators(self, scale):
        """
        Called directly by your MedNeXtv3:

            modulators =
                self.injector.modulators(2 ** i)

        scale 1 intentionally has no condition injection.
        """

        if not self.has_condition:
            return None

        key = str(int(scale))

        if key not in self.blocks:
            return None

        return self.blocks[key]

    def reset_zero(self):
        """
        Called after MedNeXtv3._init_weights().

        This is important because the parent model's generic
        weight initialization also initializes Conv3d layers
        inside the injector.

        Calling reset_zero() afterwards restores the exact
        identity initialization of every ZeroSPADE block.
        """

        for blocks in self.blocks.values():
            for block in blocks:
                block.reset_zero()