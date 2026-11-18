import copy

import torch
import torch.nn.functional as F

from .encoder_decoder import EncoderDecoder


class ACPETDistillationEncoderDecoder(EncoderDecoder):
    """Stage-3 fine-tuning with a training-only frozen EMA AC-PET teacher."""

    def __init__(self, config):
        super().__init__(config)

        settings = config.model_cfg.distillation

        if "pet" not in self.input_keys:
            raise ValueError(
                "AC-PET distillation requires pet in data_cfg.input_keys."
            )

        self.pet_index = self.input_keys.index("pet")
        self.distill_weight = float(settings.get("weight", 0.05))
        self.distill_pool_size = int(settings.get("pool_size", 16))
        self.distill_levels = tuple(
            int(level)
            for level in settings.get("levels", range(5))
        )

        self.ema_decay = float(settings.get("ema_decay", 0.999))

        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError(
                f"ema_decay must be in [0, 1), got {self.ema_decay}"
            )

        # Intentionally NOT registered as an nn.Module.
        # Created lazily after Stage-2 weights are loaded and model is on GPU.
        self.__dict__["_ema_teacher"] = None

    @staticmethod
    def _feature_loss(student, teacher):
        student = F.normalize(student.float(), dim=1)
        teacher = F.normalize(teacher.float(), dim=1)

        return (
            1.0
            - (student * teacher).sum(dim=1)
        ).mean()

    def _teacher_x(self, batch):
        if "ac_pet" not in batch:
            raise KeyError(
                "Stage 3 requires the aligned AC-PET key 'ac_pet'."
            )

        channels = [
            batch["ac_pet"] if key == "pet" else batch[key]
            for key in self.input_keys
        ]

        x = torch.cat(channels, dim=1)

        if self.channels_last:
            x = x.contiguous(
                memory_format=torch.channels_last_3d
            )

        return x

    @torch.no_grad()
    def _get_teacher(self):
        teacher = self.__dict__.get("_ema_teacher")

        if teacher is None:
            # At this point:
            # 1. Stage-2 checkpoint has already been loaded.
            # 2. self.net is already on the correct local GPU.
            teacher = copy.deepcopy(self.net)

            teacher.requires_grad_(False)
            teacher.eval()

            self.__dict__["_ema_teacher"] = teacher

        return teacher

    @torch.no_grad()
    def _update_teacher(self):
        teacher = self._get_teacher()

        decay = self.ema_decay
        one_minus_decay = 1.0 - decay

        for teacher_param, student_param in zip(
            teacher.parameters(),
            self.net.parameters(),
        ):
            teacher_param.mul_(decay).add_(
                student_param.detach(),
                alpha=one_minus_decay,
            )

        for teacher_buffer, student_buffer in zip(
            teacher.buffers(),
            self.net.buffers(),
        ):
            teacher_buffer.copy_(student_buffer)

    def training_step(self, batch):
        # ---------------------------------------------------------
        # Student
        # ---------------------------------------------------------

        x = self.get_x(batch)

        prediction, student_features = self.net.forward_features(
            x,
            feature_pool_size=self.distill_pool_size,
        )

        # ---------------------------------------------------------
        # EMA teacher
        # ---------------------------------------------------------

        teacher = self._get_teacher()

        with torch.no_grad():
            teacher_features = teacher.encode_features(
                self._teacher_x(batch),
                feature_pool_size=self.distill_pool_size,
            )

        # ---------------------------------------------------------
        # Main loss
        # ---------------------------------------------------------

        loss_items = self.compute_loss(
            pred=prediction,
            target=batch["ct"],
            prediction_mask=batch.get("prediction_mask"),
            organ_seg=batch.get("organ_seg"),
            body_seg=batch.get("body_seg"),
        )

        # ---------------------------------------------------------
        # Feature distillation
        # ---------------------------------------------------------

        level_losses = [
            self._feature_loss(
                student_features[level],
                teacher_features[level],
            )
            for level in self.distill_levels
        ]

        feature_loss = torch.stack(level_losses).mean()

        loss_items["feature_distill"] = feature_loss.detach()

        loss_items["total_loss"] = (
            loss_items["total_loss"]
            + self.distill_weight * feature_loss
        )

        batch["prediction"] = prediction

        # Update after teacher has produced the current target.
        self._update_teacher()

        return batch, loss_items