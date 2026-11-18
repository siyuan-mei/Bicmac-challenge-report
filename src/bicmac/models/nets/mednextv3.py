import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..register import MODELS
from .injector import SpatialConditionInjector


def _to_list(value, length):
    if isinstance(value, int):
        return [value] * length

    value = list(value)

    if len(value) != length:
        raise ValueError(f"Expected length {length}, got {len(value)}: {value}")

    return value


def _same_spatial(x, ref):
    if x.shape[2:] == ref.shape[2:]:
        return x

    return F.interpolate(
        x,
        size=ref.shape[2:],
        mode="trilinear",
        align_corners=False,
    )


class ConvNeXtBlock3D(nn.Module):
    """
    Clean 3D ConvNeXt / MedNeXt-style block.

    No GRN.
    InstanceNorm3d affine=True.
    """

    def __init__(
        self,
        channels,
        kernel_size=3,
        expansion=4,
        dropout=0.0,
        layer_scale_init=1.0e-6,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)

        padding = tuple((k - 1) // 2 for k in kernel_size)
        hidden_channels = int(channels * expansion)

        self.dwconv = nn.Conv3d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=True,
        )

        self.norm = nn.InstanceNorm3d(channels, affine=True)

        self.mlp = nn.Sequential(
            nn.Conv3d(channels, hidden_channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Dropout3d(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Conv3d(hidden_channels, channels, kernel_size=1, bias=True),
        )

        if layer_scale_init is None or float(layer_scale_init) <= 0:
            self.layer_scale = None
        else:
            self.layer_scale = nn.Parameter(
                float(layer_scale_init) * torch.ones(1, channels, 1, 1, 1)
            )

    def forward(self, x, modulator=None, condition=None, present=None):
        residual = x

        x = self.dwconv(x)
        x = self.norm(x)
        if modulator is not None:
            x = modulator.after_norm(x, condition, present)
        x = self.mlp(x)
        if modulator is not None:
            x = modulator.after_mlp(x, condition, present)

        if self.layer_scale is not None:
            x = x * self.layer_scale

        return residual + x


class MedNeXtStage3D(nn.Module):
    def __init__(
        self,
        channels,
        depth,
        kernel_size=5,
        expansion=4,
        dropout=0.0,
        layer_scale_init=1.0e-6,
        use_checkpoint=False,
    ):
        super().__init__()
        self.use_checkpoint = bool(use_checkpoint)

        self.blocks = nn.ModuleList(
            [
                ConvNeXtBlock3D(
                    channels=channels,
                    kernel_size=kernel_size,
                    expansion=expansion,
                    dropout=dropout,
                    layer_scale_init=layer_scale_init,
                )
                for _ in range(int(depth))
            ]
        )

    def forward(self, x, condition=None, present=None, modulators=None):
        if modulators is not None and len(modulators) != len(self.blocks):
            raise ValueError("Expected one condition modulator per MedNeXt block.")
        for index, block in enumerate(self.blocks):
            modulator = None if modulators is None else modulators[index]
            if self.use_checkpoint and self.training and x.requires_grad:
                if modulator is None:
                    x = checkpoint(block, x, use_reentrant=False)
                else:
                    def conditioned_forward(
                        value,
                        embedding,
                        mask,
                        current_block=block,
                        current_modulator=modulator,
                    ):
                        return current_block(
                            value, current_modulator, embedding, mask
                        )

                    x = checkpoint(
                        conditioned_forward,
                        x,
                        condition,
                        present,
                        use_reentrant=False,
                    )
            else:
                x = block(x, modulator, condition, present)
        return x


class ResidualDownsample3D(nn.Module):
    """
    Residual downsampling.

    Main:
        depthwise Conv3d stride=2
        IN
        1x1 expansion
        GELU
        1x1 projection

    Skip:
        AvgPool3d
        1x1 projection
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        expansion=4,
    ):
        super().__init__()

        hidden_channels = int(in_channels * expansion)

        self.main = nn.Sequential(
            nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=in_channels,
                bias=True,
            ),
            nn.InstanceNorm3d(in_channels, affine=True),
            nn.Conv3d(in_channels, hidden_channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv3d(hidden_channels, out_channels, kernel_size=1, bias=True),
        )

        self.skip = nn.Sequential(
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True),
        )

    def forward(self, x):
        main = self.main(x)
        skip = self.skip(x)
        skip = _same_spatial(skip, main)

        return main + skip


class ResidualUpsample3D(nn.Module):
    """
    Residual upsampling.

    Main:
        depthwise ConvTranspose3d stride=2
        IN
        1x1 expansion
        GELU
        1x1 projection

    Skip:
        trilinear upsample
        1x1 projection
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        expansion=4,
    ):
        super().__init__()

        hidden_channels = int(in_channels * expansion)

        self.main = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels,
                in_channels,
                kernel_size=2,
                stride=2,
                groups=in_channels,
                bias=True,
            ),
            nn.InstanceNorm3d(in_channels, affine=True),
            nn.Conv3d(in_channels, hidden_channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv3d(hidden_channels, out_channels, kernel_size=1, bias=True),
        )

        self.skip_proj = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=True,
        )

    def forward(self, x):
        main = self.main(x)

        skip = F.interpolate(
            x,
            scale_factor=2,
            mode="trilinear",
            align_corners=False,
        )
        skip = self.skip_proj(skip)
        skip = _same_spatial(skip, main)

        return main + skip


class SkipFuse3D(nn.Module):
    """
    Decoder skip fusion.

    skip_fusion:
        add:
            x + skip

        cat:
            concat(skip, x) -> 1x1 conv -> IN -> GELU

        none:
            no encoder skip
    """

    def __init__(
        self,
        channels,
        skip_fusion="add",
    ):
        super().__init__()

        self.skip_fusion = str(skip_fusion)

        if self.skip_fusion == "cat":
            self.fuse = nn.Sequential(
                nn.Conv3d(
                    channels * 2,
                    channels,
                    kernel_size=1,
                    bias=True,
                ),
            )
        elif self.skip_fusion in ("add", "none"):
            self.fuse = None
        else:
            raise ValueError(
                f"Unsupported skip_fusion: {skip_fusion}. "
                "Use 'add', 'cat', or 'none'."
            )

    def forward(self, x, skip):
        if self.skip_fusion == "none":
            return x

        skip = _same_spatial(skip, x)

        if self.skip_fusion == "add":
            return x + skip

        return self.fuse(torch.cat([skip, x], dim=1))


@MODELS.register(type="MedNeXtv3")
class MedNeXtv3(nn.Module):
    """
    Clean MedNeXt-like 3D encoder-decoder.

    Design:
        - no deep supervision
        - no GRN
        - InstanceNorm3d affine=True everywhere
        - residual downsampling / upsampling
        - optional skip fusion: add / cat / none
        - single output head

    Input:
        [B, in_channels, X, Y, Z]

    Output:
        [B, out_channels, X, Y, Z]
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        input_channels=None,
        n_classes=None,
        features=(32, 64, 128, 256, 512),
        depths=(2, 3, 4, 4, 4),
        decoder_depths=None,
        kernel_sizes=(3, 3, 3, 3, 3),
        expansion=4,
        down_expansion=None,
        up_expansion=None,
        dropout=0.0,
        skip_fusion="add",
        layer_scale_init=1.0e-6,
        use_checkpoint=False,
        input_fusion="concat",
        input_keys=None,
        primary_keys=None,
        condition_channels=(64, 128, 256, 512),
    ):
        super().__init__()

        if input_channels is not None:
            in_channels = input_channels

        if n_classes is not None:
            out_channels = n_classes

        features = list(features)
        num_stages = len(features)

        depths = _to_list(depths, num_stages)
        kernel_sizes = _to_list(kernel_sizes, num_stages)
        expansion = _to_list(expansion, num_stages)

        if decoder_depths is None:
            decoder_depths = [max(1, d // 2) for d in depths[:-1]]
        else:
            decoder_depths = _to_list(decoder_depths, num_stages - 1)

        if down_expansion is None:
            down_expansion = expansion[:-1]
        else:
            down_expansion = _to_list(down_expansion, num_stages - 1)

        if up_expansion is None:
            up_expansion = expansion[1:]
        else:
            up_expansion = _to_list(up_expansion, num_stages - 1)

        self.num_stages = num_stages

        self.input_fusion = str(input_fusion)
        if self.input_fusion not in ("concat", "injector"):
            raise ValueError(f"Unsupported input_fusion: {input_fusion}")

        stem_channels = in_channels
        self.injector = None
        if self.input_fusion == "injector":
            if input_keys is None or len(input_keys) != in_channels:
                raise ValueError("Injector fusion requires one input_keys entry per channel.")
            primary_keys = tuple(primary_keys or ("pet",))
            stem_channels = len(primary_keys)
            self.injector = SpatialConditionInjector(
                input_keys=input_keys,
                primary_keys=primary_keys,
                feature_channels_by_scale={
                    2 ** index: channels for index, channels in enumerate(features)
                },
                depth_by_scale={
                    2 ** index: depth for index, depth in enumerate(depths)
                },
                condition_channels=condition_channels,
            )

        self.stem = nn.Sequential(
            nn.Conv3d(
                stem_channels,
                features[0],
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.InstanceNorm3d(features[0], affine=True),
            nn.GELU(),
        )

        self.encoder_stages = nn.ModuleList()
        self.downs = nn.ModuleList()

        for i in range(num_stages):
            self.encoder_stages.append(
                MedNeXtStage3D(
                    channels=features[i],
                    depth=depths[i],
                    kernel_size=kernel_sizes[i],
                    expansion=expansion[i],
                    dropout=dropout,
                    layer_scale_init=layer_scale_init,
                    use_checkpoint=use_checkpoint,
                )
            )

            if i < num_stages - 1:
                self.downs.append(
                    ResidualDownsample3D(
                        in_channels=features[i],
                        out_channels=features[i + 1],
                        expansion=down_expansion[i],
                    )
                )

        self.ups = nn.ModuleList()
        self.skip_fuses = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()

        for i in reversed(range(num_stages - 1)):
            self.ups.append(
                ResidualUpsample3D(
                    in_channels=features[i + 1],
                    out_channels=features[i],
                    expansion=up_expansion[i],
                )
            )

            self.skip_fuses.append(
                SkipFuse3D(
                    channels=features[i],
                    skip_fusion=skip_fusion,
                )
            )

            self.decoder_stages.append(
                MedNeXtStage3D(
                    channels=features[i],
                    depth=decoder_depths[i],
                    kernel_size=kernel_sizes[i],
                    expansion=expansion[i],
                    dropout=dropout,
                    layer_scale_init=layer_scale_init,
                    use_checkpoint=use_checkpoint,
                )
            )

        self.out_conv = nn.Conv3d(
            features[0],
            out_channels,
            kernel_size=1,
            bias=True,
        )

        self._init_weights()
        if self.injector is not None:
            self.injector.reset_zero()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.trunc_normal_(m.weight, std=0.02)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.InstanceNorm3d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _distill_feature(x, pool_size):
        if pool_size is None:
            return x
        size = tuple(min(int(pool_size), value) for value in x.shape[2:])
        return F.adaptive_avg_pool3d(x, size)

    def _encode(self, x, feature_pool_size=None):
        state = None
        if self.injector is not None:
            x, state = self.injector.encode_inputs(x)
        x = self.stem(x)

        skips = []
        encoder_features = []

        for i in range(self.num_stages):
            if self.injector is not None:
                contexts, present = state
                modulators = self.injector.modulators(2 ** i)
                condition = contexts.get(2 ** i)
            else:
                condition = present = modulators = None
            x = self.encoder_stages[i](
                x,
                condition=condition,
                present=present,
                modulators=modulators,
            )
            encoder_features.append(self._distill_feature(x, feature_pool_size))

            if i < self.num_stages - 1:
                skips.append(x)
                x = self.downs[i](x)

        return x, skips[::-1], tuple(encoder_features)

    def encode_features(self, x, feature_pool_size=None):
        """Return encoder targets without paying for a decoder forward."""
        _, _, features = self._encode(x, feature_pool_size)
        return features

    def forward_features(self, x, feature_pool_size=None):
        x, skips, encoder_features = self._encode(x, feature_pool_size)

        for up, fuse, dec, skip in zip(
            self.ups,
            self.skip_fuses,
            self.decoder_stages,
            skips,
        ):
            x = up(x)
            x = fuse(x, skip)
            x = dec(x)

        return self.out_conv(x), encoder_features

    def forward(self, x):
        output, _ = self.forward_features(x)
        return output
