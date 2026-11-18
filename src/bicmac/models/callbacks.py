import hashlib
import json
import os
import re
import warnings
from collections import deque
from pathlib import Path
from typing import List, Optional, Union

import nibabel as nib
import numpy as np
import torch
import wandb
from matplotlib import cm
from PIL import Image
from scipy.ndimage import gaussian_filter
from lightning_fabric.wrappers import _unwrap_objects
from lightning_utilities import apply_to_collection


def foreground_crop_2d(image: np.ndarray, mask: np.ndarray, margin: int = 8):
    ys, xs = np.where(mask > 0)

    if len(ys) == 0 or len(xs) == 0:
        return image

    ymin, ymax = ys.min(), ys.max() + 1
    xmin, xmax = xs.min(), xs.max() + 1

    ymin = max(0, ymin - margin)
    ymax = min(mask.shape[0], ymax + margin)
    xmin = max(0, xmin - margin)
    xmax = min(mask.shape[1], xmax + margin)

    if image.ndim == 2:
        return image[ymin:ymax, xmin:xmax]

    return image[:, ymin:ymax, xmin:xmax]


class CheckpointCallback:
    def __init__(
        self,
        interval: int = 1,
        max_keep_ckpts: int = -1,
        save_last: bool = True,
        save_best: Optional[Union[str, List[str]]] = None,
        mode: Union[str, List[str]] = "max",
        dirpath: str = "work_dir/checkpoints",
        filename_tmpl: Optional[str] = None,
        specific_save_epoch=None,
    ):
        self.interval = interval
        self.max_keep_ckpts = max_keep_ckpts
        self.save_last = save_last
        self.save_best = (
            save_best
            if isinstance(save_best, list)
            else [save_best]
            if save_best
            else []
        )
        self.mode = mode if isinstance(mode, list) else [mode] * len(self.save_best)
        self.dirpath = dirpath
        os.makedirs(self.dirpath, exist_ok=True)
        self.filename_tmpl = filename_tmpl or "epoch_{}.ckpt"

        self.best_ckpt_path = {k: None for k in self.save_best}
        self.best_score = {
            k: -float("inf") if m == "max" else float("inf")
            for k, m in zip(self.save_best, self.mode)
        }
        self.best_state_path = os.path.join(self.dirpath, "best_metrics.json")
        self._restore_best_state()

        self.keep_ckpt_ids = deque(
            maxlen=max_keep_ckpts if max_keep_ckpts > 0 else None
        )
        self.specific_save_epoch = specific_save_epoch

    def checkpoint_on_train_epoch_end(self, *, fabric, state):
        epoch = state["epoch"]

        if epoch % self.interval == 0:
            self._save(fabric, self.filename_tmpl.format(epoch), state, step=epoch)

        if self.specific_save_epoch is not None and epoch in self.specific_save_epoch:
            self._save(
                fabric,
                f"specific_epoch{epoch}.ckpt",
                {"model": state["model"]},
                step=epoch,
            )

        if self.save_last:
            self._save(fabric, "last.ckpt", state, step=epoch)

    def checkpoint_on_validation_epoch_end(self, *, fabric, state, metrics):
        best_state = {"model": state["model"]}
        epoch = state["epoch"]

        for k in self.save_best:
            if k not in metrics:
                continue

            score = metrics[k]
            rule = self.mode[self.save_best.index(k)]

            is_better = (
                score > self.best_score[k]
                if rule == "max"
                else score < self.best_score[k]
            )

            if not is_better:
                continue

            self.best_score[k] = score
            filename = f"best_{k}.ckpt"
            ckpt_path = os.path.join(self.dirpath, filename)
            self.best_ckpt_path[k] = ckpt_path
            self._save(fabric, filename, best_state, step=epoch)
            if fabric.is_global_zero:
                self._save_best_state()

    def _restore_best_state(self):
        if not os.path.isfile(self.best_state_path):
            return
        try:
            with open(self.best_state_path, "r", encoding="utf-8") as handle:
                saved = json.load(handle)
            for key in self.save_best:
                if key in saved:
                    self.best_score[key] = float(saved[key])
                    self.best_ckpt_path[key] = os.path.join(
                        self.dirpath, f"best_{key}.ckpt"
                    )
        except (OSError, ValueError, TypeError) as error:
            warnings.warn(f"Could not restore best checkpoint scores: {error}")

    def _save_best_state(self):
        temporary = f"{self.best_state_path}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(self.best_score, handle, indent=2, sort_keys=True)
        os.replace(temporary, self.best_state_path)

    def _save(self, fabric, filename, state, step):
        ckpt_path = os.path.join(self.dirpath, filename)
        state = state or {}

        fabric._strategy.save_checkpoint(
            path=ckpt_path,
            state=_unwrap_objects(state),
        )

        if not fabric.is_global_zero:
            return

        print(f"[CheckpointCallback] Saved checkpoint: {ckpt_path}")

        if self.max_keep_ckpts > 0 and not filename.startswith(
            ("last", "best", "specific")
        ):
            try:
                if len(self.keep_ckpt_ids) == self.keep_ckpt_ids.maxlen:
                    old_step = self.keep_ckpt_ids[0]
                    old_path = os.path.join(
                        self.dirpath,
                        self.filename_tmpl.format(old_step),
                    )

                    if os.path.exists(old_path):
                        os.remove(old_path)
                        print(f"[CheckpointCallback] Removed old checkpoint: {old_path}")

            except Exception as e:
                print(f"[CheckpointCallback] Failed to remove old checkpoint: {e}")

            finally:
                self.keep_ckpt_ids.append(step)


class VisualizationCallback:
    def __init__(
        self,
        output_dir="work_dir/outputs",
        epoch_interval=10,
        save_input=False,
        save_target=False,
        save_png=True,
        log_to_wandb=True,
        logger=None,
        crop_margin=8,
        save_debug_numpy=False,
        max_png_component_length=48,
        max_png_path_length=240,
        mu_spacing_mm=(1.5234375, 1.5234375, 2.0),
        mu_smooth_fwhm_mm=4.0,
    ):
        self.save_dir = output_dir
        self.epoch_interval = epoch_interval
        self.save_input = save_input
        self.save_target = save_target
        self.save_png = save_png
        self.log_to_wandb = log_to_wandb
        self.logger = logger
        self.crop_margin = crop_margin
        self.save_debug_numpy = save_debug_numpy
        self.max_png_component_length = max(16, int(max_png_component_length))
        self.max_png_path_length = max(160, int(max_png_path_length))
        self.mu_spacing_mm = tuple(float(v) for v in mu_spacing_mm)
        if len(self.mu_spacing_mm) != 3 or any(v <= 0.0 for v in self.mu_spacing_mm):
            raise ValueError("mu_spacing_mm must contain positive X, Y, Z spacing.")
        self.mu_smooth_fwhm_mm = float(mu_smooth_fwhm_mm)
        if self.mu_smooth_fwhm_mm <= 0.0:
            raise ValueError("mu_smooth_fwhm_mm must be positive.")
        self._wandb_this_call = True
        self._media_buffer = None

    def norm_to_hu(self, x):
        return x.astype(np.float32) * 3000.0 - 1000.0

    def hu_to_mu(self, hu):
        h = hu.astype(np.float32) + 1000.0
        mu = np.where(
            h < 1047.0,
            9.6e-5 * h,
            5.10e-5 * h + 4.71e-2,
        )
        return np.clip(mu, 0.0, None).astype(np.float32)

    def smooth_mu(self, mu):
        """Official pre-STIR 4-mm FWHM Gaussian smoothing on a full mu-map."""
        sigma = tuple(
            self.mu_smooth_fwhm_mm / (2.355 * spacing)
            for spacing in self.mu_spacing_mm
        )
        return gaussian_filter(
            mu.astype(np.float32, copy=False),
            sigma=sigma,
            mode="reflect",
            truncate=4.0,
        ).astype(np.float32, copy=False)

    def norm01(self, img):
        img = img.astype(np.float32)
        return ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).clip(
            0, 255
        ).astype(np.uint8)

    def norm_hu(self, img, vmin=-1000.0, vmax=1000.0):
        img = np.clip(img.astype(np.float32), vmin, vmax)
        return ((img - vmin) / (vmax - vmin + 1e-8) * 255).astype(np.uint8)

    def norm_mu(self, img, vmin=0.0, vmax=0.18):
        img = np.clip(img.astype(np.float32), vmin, vmax)
        return ((img - vmin) / (vmax - vmin + 1e-8) * 255).astype(np.uint8)

    def best_slice_index(self, mask, view="axial"):
        mask = mask > 0

        if view == "axial":
            area = mask.sum(axis=(0, 1))
        elif view == "coronal":
            area = mask.sum(axis=(0, 2))
        elif view == "sagittal":
            area = mask.sum(axis=(1, 2))
        else:
            raise ValueError(f"Unsupported view: {view}")

        if area.max() == 0:
            return None

        return int(area.argmax())

    def _first(self, x):
        while isinstance(x, (list, tuple)):
            if len(x) == 0:
                return None
            x = x[0]
        return x

    def _caption(self, batch, index, epoch):
        def value(key, default):
            values = batch.get(key, default)
            if isinstance(values, (list, tuple)) and len(values) > index:
                values = values[index]
            return str(self._first(values))

        return " | ".join(
            (
                f"epoch={epoch}",
                f"id={value('id', 'unknown')}",
                f"dataset={value('dataset', 'bicmac')}",
                f"modality={value('modality', 'unknown')}",
            )
        )

    def _vflip(self, img):
        if img.ndim == 2:
            return np.flipud(img)
        if img.ndim == 3:
            return img[:, ::-1, :]
        return img

    @staticmethod
    def _safe_path_component(value, max_length=48):
        """
        Create a Windows-safe path component.

        Long metadata such as DICOM series descriptions must not be inserted
        verbatim into filenames because the complete Windows path can reach the
        legacy 260-character limit.
        """
        text = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(value))
        text = text.strip(" ._") or "unnamed"

        if len(text) <= max_length:
            return text

        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
        prefix = text[: max(1, max_length - len(digest) - 1)].rstrip(" ._")
        return f"{prefix}_{digest}"

    @staticmethod
    def _stable_digest(*values, length=10):
        payload = "|".join(str(value) for value in values)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]

    @staticmethod
    def _io_path(path):
        """
        Return an extended-length Windows path when it is still required.

        The generated PNG names are intentionally short, so this is normally
        only a final safeguard for a very long user-selected output directory.
        """
        path = Path(path).resolve()

        if os.name != "nt":
            return str(path)

        text = str(path)
        if text.startswith("\\\\?\\"):
            return text

        if len(text) >= 248:
            if text.startswith("\\\\"):
                return "\\\\?\\UNC\\" + text[2:]
            return "\\\\?\\" + text

        return text

    def _png_save_path(self, key, caption, epoch, index):
        root = Path(self.save_dir) / "visuals"
        components = [
            self._safe_path_component(part, self.max_png_component_length)
            for part in str(key).split("/")
            if part
        ]

        image_dir = root.joinpath(*components)
        digest = self._stable_digest(key, caption, epoch, index)
        filename = f"e{int(epoch):04d}_i{int(index):02d}_{digest}.png"
        save_path = image_dir / filename

        # Compact the directory hierarchy if the selected output root is long.
        if os.name == "nt" and len(str(save_path.resolve())) > self.max_png_path_length:
            compact_key = self._safe_path_component(
                str(key).replace("/", "_"),
                max_length=40,
            )
            image_dir = root / compact_key
            save_path = image_dir / filename

        image_dir.mkdir(parents=True, exist_ok=True)
        return save_path

    def _wandb_log_images(self, key, images, captions, epoch):
        images = list(images)
        captions = list(captions)

        if len(captions) < len(images):
            captions.extend([""] * (len(images) - len(captions)))

        if self.save_png:
            for index, img in enumerate(images):
                caption = captions[index]
                save_path = self._png_save_path(
                    key=key,
                    caption=caption,
                    epoch=epoch,
                    index=index,
                )

                try:
                    Image.fromarray(np.ascontiguousarray(img)).save(
                        self._io_path(save_path)
                    )
                except OSError as exc:
                    # Visualization is auxiliary and must not terminate training.
                    warnings.warn(
                        f"Failed to save visualization '{save_path}': {exc}",
                        RuntimeWarning,
                    )

        if (
            not self._wandb_this_call
            or not self.log_to_wandb
            or self.logger is None
        ):
            return

        if self._media_buffer is not None:
            buffered_images, buffered_captions = self._media_buffer.setdefault(
                key, ([], [])
            )
            buffered_images.extend(images)
            buffered_captions.extend(captions)
            return

        self._commit_wandb_media({key: (images, captions)}, epoch)

    def _commit_wandb_media(self, media, epoch):
        if not media or not self._wandb_this_call:
            return

        run = getattr(self.logger, "experiment", None)
        if run is not None and hasattr(run, "log"):
            payload = {"epoch/epochs": epoch}
            for key, (images, captions) in media.items():
                images = self._pad_image_batch(images)
                payload[key] = [
                    wandb.Image(img, caption=caption)
                    for img, caption in zip(images, captions)
                ]
            run.log(payload)
            return

        for key, (images, captions) in media.items():
            self.logger.log_image(
                key=key,
                images=images,
                caption=captions,
                step=epoch,
            )

    @staticmethod
    def _pad_image_batch(images):
        """Pad a W&B image list to one canvas size without rescaling anatomy."""
        if not images:
            return images

        height = max(image.shape[0] for image in images)
        width = max(image.shape[1] for image in images)
        padded = []
        for image in images:
            pad_y = height - image.shape[0]
            pad_x = width - image.shape[1]
            top, left = pad_y // 2, pad_x // 2
            pad_width = (
                ((top, pad_y - top), (left, pad_x - left))
                if image.ndim == 2
                else ((top, pad_y - top), (left, pad_x - left), (0, 0))
            )
            padded.append(np.pad(image, pad_width, mode="constant"))
        return padded

    def _begin_media_batch(self):
        self._media_buffer = {}

    def _flush_media_batch(self, epoch):
        media = self._media_buffer or {}
        self._media_buffer = None
        self._commit_wandb_media(media, epoch)

    def _get_ref_nifti(self, batch):
        if "features_dir" in batch:
            features_dir = self._first(batch["features_dir"])
            ref_path = os.path.join(features_dir, "nacpet.nii.gz")
            return nib.load(ref_path)

        if "pet" in batch and hasattr(batch["pet"], "meta"):
            meta = batch["pet"].meta
            filename = meta.get("filename_or_obj", None)

            if filename is not None:
                filename = self._first(filename)
                return nib.load(str(filename))

        raise RuntimeError(
            "Cannot find reference NIfTI. Keep `features_dir` in batch "
            "or keep MONAI metadata for `pet`."
        )

    @staticmethod
    def _current_affine(batch, shape):
        for key in ("pet", "mr_in", "ct"):
            image = batch.get(key)
            if image is None or tuple(image.shape[-3:]) != tuple(shape):
                continue

            affine = getattr(image, "affine", None)
            if affine is None and hasattr(image, "meta"):
                affine = image.meta.get("affine")
            if affine is None:
                continue

            if torch.is_tensor(affine):
                affine = affine.detach().cpu().numpy()
            affine = np.asarray(affine)
            while affine.ndim > 2:
                affine = affine[0]
            if affine.shape == (4, 4):
                return affine
        return None

    def _save_nifti_like_ref(
            self,
            array_xyz,
            batch,
            save_dir,
            id_,
            filename,
            clip=None,
            dtype=np.float32,
    ):
        ref = self._get_ref_nifti(batch)

        array_xyz = np.asarray(array_xyz)

        if clip is not None:
            array_xyz = np.clip(array_xyz, clip[0], clip[1])

        array_xyz = array_xyz.astype(dtype)

        affine = self._current_affine(batch, array_xyz.shape)
        if array_xyz.shape != ref.shape and affine is None:
            raise ValueError(
                f"{filename}: array shape {array_xyz.shape} != reference shape "
                f"{ref.shape}, and cropped affine metadata is unavailable"
            )

        header = ref.header.copy()
        out_img = nib.Nifti1Image(array_xyz, affine if affine is not None else ref.affine, header)
        out_img.set_data_dtype(dtype)

        subject_dir = os.path.join(save_dir, id_)
        os.makedirs(subject_dir, exist_ok=True)

        out_path = os.path.join(subject_dir, filename)
        nib.save(out_img, out_path)

        print(f"[VisualizationCallback] Saved NIfTI: {out_path}")

    def _slice_xyz(self, arr, view="axial", index=None):
        """
        arr:
            [X, Y, Z] or [C, X, Y, Z]

        XYZ convention:
            axial:    fixed Z -> [Y, X]
            coronal:  fixed Y -> [Z, X]
            sagittal: fixed X -> [Z, Y]
        """
        if arr.ndim == 3:
            X, Y, Z = arr.shape

            if view == "axial":
                z = Z // 2 if index is None else index
                return arr[:, :, z].T

            if view == "coronal":
                y = Y // 2 if index is None else index
                return arr[:, y, :].T

            if view == "sagittal":
                x = X // 2 if index is None else index
                return arr[x, :, :].T

        if arr.ndim == 4:
            C, X, Y, Z = arr.shape

            if view == "axial":
                z = Z // 2 if index is None else index
                return arr[:, :, :, z].transpose(0, 2, 1)

            if view == "coronal":
                y = Y // 2 if index is None else index
                return arr[:, :, y, :].transpose(0, 2, 1)

            if view == "sagittal":
                x = X // 2 if index is None else index
                return arr[:, x, :, :].transpose(0, 2, 1)

        raise ValueError(f"Unsupported array shape {arr.shape} for view={view}")

    def _log_gray(self, key, img, captions, epoch, mode="auto"):
        img = self._vflip(img)

        if mode == "hu":
            img = self.norm_hu(img)
        elif mode == "mu":
            img = self.norm_mu(img)
        else:
            img = self.norm01(img)

        self._wandb_log_images(key, [img], captions, epoch)

    def _log_diff(self, key, diff, captions, epoch, vmax):
        diff = self._vflip(diff)
        diff = np.clip(diff, -vmax, vmax) / vmax
        diff = (diff + 1.0) / 2.0
        img = (cm.get_cmap("RdBu_r")(diff)[..., :3] * 255).astype(np.uint8)

        self._wandb_log_images(key, [img], captions, epoch)

    def _log_label(self, key, label, captions, epoch):
        label = self._vflip(label)
        label = self.norm01(label.astype(np.float32))
        img = (cm.get_cmap("tab20")(label / 255.0)[..., :3] * 255).astype(np.uint8)

        self._wandb_log_images(key, [img], captions, epoch)

    def _extract_single_volume(self, arr):
        arr = self._first(arr)

        if torch.is_tensor(arr):
            arr = arr.detach().cpu().numpy()

        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Expected ndarray/tensor volume, got {type(arr)}")

        if arr.ndim == 5:
            return arr[0, 0]
        if arr.ndim == 4:
            return arr[0]
        if arr.ndim == 3:
            return arr

        raise ValueError(f"Unsupported volume shape: {arr.shape}")

    def _extract_input_sample(self, arr, index=0):
        arr = self._first(arr)

        if torch.is_tensor(arr):
            arr = arr.detach().cpu().numpy()

        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Expected ndarray/tensor input, got {type(arr)}")

        if arr.ndim == 5:
            return arr[index]      # [B, C, X, Y, Z] -> [C, X, Y, Z]
        if arr.ndim == 4:
            return arr             # [C, X, Y, Z]
        if arr.ndim == 3:
            return arr             # [X, Y, Z] or [C, X, Z]

        raise ValueError(f"Unsupported input shape: {arr.shape}")

    def _log_input_channels(
        self,
        batch_np,
        mask2d,
        captions,
        epoch,
        view,
        slice_idx,
        stage="media_val",
        sample_idx=0,
    ):
        input_keys = [
            "pet",
            "ac_pet",
            "mr_in",
            "mr_out",
            "mr_face_mask",
        ]

        for key in input_keys:
            if key not in batch_np:
                continue

            try:
                arr = self._extract_input_sample(batch_np[key], index=sample_idx)
            except (TypeError, ValueError, AttributeError, IndexError):
                continue

            try:
                if arr.ndim == 4:
                    slice2d = self._slice_xyz(arr, view=view, index=slice_idx)

                    for c in range(slice2d.shape[0]):
                        crop = foreground_crop_2d(
                            slice2d[c],
                            mask2d,
                            margin=self.crop_margin,
                        )
                        self._log_gray(
                            f"{stage}/{view}/input_{key}_c{c}",
                            crop,
                            captions,
                            epoch,
                        )
                else:
                    slice2d = self._slice_xyz(arr, view=view, index=slice_idx)
                    crop = foreground_crop_2d(
                        slice2d,
                        mask2d,
                        margin=self.crop_margin,
                    )
                    self._log_gray(
                        f"{stage}/{view}/input_{key}",
                        crop,
                        captions,
                        epoch,
                    )
            except Exception:
                continue

    def _log_topogram_xz(
        self,
        batch_np,
        mask_vol,
        captions,
        epoch,
        stage="media_val",
        sample_idx=0,
    ):
        """
        Topogram is an XZ projection.
        Log it only once in XZ plane.

        If topogram has already been pasted to [C, X, Y, Z],
        XZ plane is the coronal view: fixed Y -> [Z, X].
        """
        mask_vol = (mask_vol > 0).astype(np.uint8)
        y_idx = self.best_slice_index(mask_vol, view="coronal")

        if y_idx is None:
            y_idx = mask_vol.shape[1] // 2

        mask_xz = self._slice_xyz(mask_vol, view="coronal", index=y_idx)

        for key in ("topogram", "tomogram"):
            if key not in batch_np:
                continue

            try:
                arr = self._extract_input_sample(batch_np[key], index=sample_idx)
            except (TypeError, ValueError, AttributeError, IndexError):
                continue

            try:
                if arr.ndim == 4:
                    # [C, X, Y, Z] -> [C, Z, X]
                    slice2d = self._slice_xyz(arr, view="coronal", index=y_idx)

                    for c in range(slice2d.shape[0]):
                        crop = foreground_crop_2d(
                            slice2d[c],
                            mask_xz,
                            margin=self.crop_margin,
                        )
                        self._log_gray(
                            f"{stage}/zx/input_{key}_c{c}",
                            crop,
                            captions,
                            epoch,
                        )

                elif arr.ndim == 3 and arr.shape[0] <= 8:
                    # [C, X, Z] -> [C, Z, X]
                    slice2d = arr.transpose(0, 2, 1)

                    for c in range(slice2d.shape[0]):
                        crop = slice2d[c]
                        if crop.shape == mask_xz.shape:
                            crop = foreground_crop_2d(
                                crop, mask_xz, margin=self.crop_margin
                            )
                        self._log_gray(
                            f"{stage}/input_{key}_c{c}",
                            crop,
                            captions,
                            epoch,
                        )

                elif arr.ndim == 3:
                    # fallback [X, Y, Z] -> [Z, X]
                    slice2d = self._slice_xyz(arr, view="coronal", index=y_idx)
                    crop = foreground_crop_2d(
                        slice2d,
                        mask_xz,
                        margin=self.crop_margin,
                    )
                    self._log_gray(
                        f"{stage}/zx/input_{key}",
                        crop,
                        captions,
                        epoch,
                    )

            except Exception:
                continue

    def vis_on_train_batch_end(
        self,
        epoch,
        batch,
        batch_idx,
        is_global_zero=False,
        n=4,
    ):
        """
        Training stage:
            batch["prediction"] and batch["ct"] are normalized CT.
            Inputs are normalized.
            CT prediction is logged in axial view.
            Topogram is logged only in XZ plane.
        """
        if not is_global_zero:
            return
        if batch_idx != 0 or epoch % self.epoch_interval != 0:
            return
        if not (self.save_png or self.log_to_wandb):
            return
        if "prediction" not in batch or "ct" not in batch:
            return

        self._wandb_this_call = is_global_zero

        n = min(n, batch["prediction"].shape[0])

        pred_norm = batch["prediction"][:n].detach().cpu().numpy()
        target_norm = batch["ct"][:n].detach().cpu().numpy()

        pred_hu = np.clip(self.norm_to_hu(pred_norm), -1024.0, 3071.0)
        target_hu = np.clip(self.norm_to_hu(target_norm), -1024.0, 3071.0)

        mask = batch.get("prediction_mask", None)
        if mask is None:
            mask_np = np.ones_like(target_hu, dtype=np.uint8)
        else:
            mask_np = (mask[:n].detach().cpu().numpy() > 0).astype(np.uint8)

        batch_np = apply_to_collection(
            batch,
            dtype=torch.Tensor,
            function=lambda t: t.detach().cpu().numpy(),
        )

        captions = [self._caption(batch, i, epoch) for i in range(n)]
        self._begin_media_batch()

        for i in range(n):
            pred_vol = pred_hu[i, 0]
            target_vol = target_hu[i, 0]
            mask_vol = mask_np[i, 0]

            slice_idx = self.best_slice_index(mask_vol, view="axial")
            mask2d = self._slice_xyz(mask_vol, view="axial", index=slice_idx)

            pred2d = foreground_crop_2d(
                self._slice_xyz(pred_vol, view="axial", index=slice_idx),
                mask2d,
                margin=self.crop_margin,
            )
            target2d = foreground_crop_2d(
                self._slice_xyz(target_vol, view="axial", index=slice_idx),
                mask2d,
                margin=self.crop_margin,
            )
            mask2d_crop = foreground_crop_2d(
                mask2d,
                mask2d,
                margin=self.crop_margin,
            )

            pred_mu2d = self.hu_to_mu(pred2d)
            target_mu2d = self.hu_to_mu(target2d)

            self._log_gray(
                f"media_train/axial/pred_hu",
                pred2d,
                [captions[i]],
                epoch,
                mode="hu",
            )
            self._log_gray(
                f"media_train/axial/target_hu",
                target2d,
                [captions[i]],
                epoch,
                mode="hu",
            )
            self._log_diff(
                f"media_train/axial/diff_hu",
                (target2d - pred2d) * (mask2d_crop > 0),
                [captions[i]],
                epoch,
                vmax=300.0,
            )

            self._log_gray(
                f"media_train/axial/pred_mu",
                pred_mu2d,
                [captions[i]],
                epoch,
                mode="mu",
            )
            self._log_gray(
                f"media_train/axial/target_mu",
                target_mu2d,
                [captions[i]],
                epoch,
                mode="mu",
            )
            self._log_diff(
                f"media_train/axial/diff_mu",
                (target_mu2d - pred_mu2d) * (mask2d_crop > 0),
                [captions[i]],
                epoch,
                vmax=0.03,
            )

            self._log_input_channels(
                batch_np=batch_np,
                mask2d=mask2d,
                captions=[captions[i]],
                epoch=epoch,
                view="axial",
                slice_idx=slice_idx,
                stage="media_train",
                sample_idx=i,
            )

            self._log_topogram_xz(
                batch_np=batch_np,
                mask_vol=mask_vol,
                captions=[captions[i]],
                epoch=epoch,
                stage="media_train",
                sample_idx=i,
            )

        self._flush_media_batch(epoch)

    def vis_on_validation_batch_end(
        self,
        epoch,
        batch,
        batch_idx,
        is_global_zero=False,
        save=True,
        exist_gt=True,
    ):
        """
        Validation / inference stage:
            batch["prediction"] and batch["ct"] are already HU.
            Topogram is logged only in XZ plane.
        """
        if exist_gt and batch_idx != 0:
            return

        self._wandb_this_call = is_global_zero

        batch_np = apply_to_collection(
            batch,
            dtype=torch.Tensor,
            function=lambda t: t.detach().cpu().numpy(),
        )

        id_ = str(self._first(batch["id"]))
        captions = [self._caption(batch, 0, epoch)]

        os.makedirs(self.save_dir, exist_ok=True)

        pred_hu = self._extract_single_volume(batch_np["prediction"])
        pred_hu = np.clip(pred_hu.astype(np.float32), -1024.0, 3071.0)
        pred_mu = self.hu_to_mu(pred_hu)

        gt_hu = None
        gt_mu = None
        recon_pred_mu = None
        pred_mu_smooth = None
        gt_mu_smooth = None

        prediction_mask_vol = None
        if "prediction_mask" in batch_np:
            prediction_mask_vol = self._extract_single_volume(
                batch_np["prediction_mask"]
            )
            prediction_mask_vol = (prediction_mask_vol > 0).astype(np.uint8)

        if exist_gt and "ct" in batch_np:
            gt_hu = self._extract_single_volume(batch_np["ct"])
            gt_hu = np.clip(gt_hu.astype(np.float32), -1024.0, 3071.0)
            gt_mu = self.hu_to_mu(gt_hu)

            # Mirror the official reconstruction responsibility model before
            # visualizing the mu-map handed toward STIR: outside prediction_mask
            # the submitted sCT is replaced by GT before HU->mu and smoothing.
            if prediction_mask_vol is not None:
                recon_pred_hu = np.where(
                    prediction_mask_vol > 0,
                    pred_hu,
                    gt_hu,
                ).astype(np.float32, copy=False)
            else:
                recon_pred_hu = pred_hu
            recon_pred_mu = self.hu_to_mu(recon_pred_hu)
            pred_mu_smooth = self.smooth_mu(recon_pred_mu)
            gt_mu_smooth = self.smooth_mu(gt_mu)

        if save:
            self._save_nifti_like_ref(
                array_xyz=pred_hu,
                batch=batch,
                save_dir=self.save_dir,
                id_=id_,
                filename="sct.nii.gz",
                clip=(-1024.0, 3071.0),
            )
            self._save_nifti_like_ref(
                array_xyz=pred_mu,
                batch=batch,
                save_dir=self.save_dir,
                id_=id_,
                filename="sct_mu.nii.gz",
                clip=(0.0, None),
            )

            if recon_pred_mu is not None:
                self._save_nifti_like_ref(
                    array_xyz=recon_pred_mu,
                    batch=batch,
                    save_dir=self.save_dir,
                    id_=id_,
                    filename="sct_recon_effective_mu.nii.gz",
                    clip=(0.0, None),
                )
            if pred_mu_smooth is not None:
                self._save_nifti_like_ref(
                    array_xyz=pred_mu_smooth,
                    batch=batch,
                    save_dir=self.save_dir,
                    id_=id_,
                    filename="sct_recon_effective_mu_smoothed_4mm.nii.gz",
                    clip=(0.0, None),
                )

            if self.save_target and gt_hu is not None:
                self._save_nifti_like_ref(
                    array_xyz=gt_hu,
                    batch=batch,
                    save_dir=self.save_dir,
                    id_=id_,
                    filename="groundtruth.nii.gz",
                    clip=(-1024.0, 3071.0),
                )
                self._save_nifti_like_ref(
                    array_xyz=gt_mu,
                    batch=batch,
                    save_dir=self.save_dir,
                    id_=id_,
                    filename="groundtruth_mu.nii.gz",
                    clip=(0.0, None),
                )
                if gt_mu_smooth is not None:
                    self._save_nifti_like_ref(
                        array_xyz=gt_mu_smooth,
                        batch=batch,
                        save_dir=self.save_dir,
                        id_=id_,
                        filename="groundtruth_mu_smoothed_4mm.nii.gz",
                        clip=(0.0, None),
                    )

            if self.save_debug_numpy:
                debug_dir = os.path.join(self.save_dir, "_debug", id_)
                os.makedirs(debug_dir, exist_ok=True)
                np.save(os.path.join(debug_dir, "pred_hu.npy"), pred_hu)
                np.save(os.path.join(debug_dir, "pred_mu.npy"), pred_mu)
                if recon_pred_mu is not None:
                    np.save(
                        os.path.join(debug_dir, "recon_pred_mu.npy"),
                        recon_pred_mu,
                    )
                if pred_mu_smooth is not None:
                    np.save(
                        os.path.join(debug_dir, "recon_pred_mu_smoothed_4mm.npy"),
                        pred_mu_smooth,
                    )

                if gt_hu is not None:
                    np.save(os.path.join(debug_dir, "gt_hu.npy"), gt_hu)
                    np.save(os.path.join(debug_dir, "gt_mu.npy"), gt_mu)
                    if gt_mu_smooth is not None:
                        np.save(
                            os.path.join(debug_dir, "gt_mu_smoothed_4mm.npy"),
                            gt_mu_smooth,
                        )

        if not (self.save_png or (self.log_to_wandb and is_global_zero)):
            return

        stage = "media_val" if exist_gt else "media_test"

        if prediction_mask_vol is not None:
            mask = prediction_mask_vol
        elif "body_seg" in batch_np:
            mask = self._extract_single_volume(batch_np["body_seg"])
            mask = (mask > 0).astype(np.uint8)
        else:
            mask = np.ones_like(pred_hu, dtype=np.uint8)
        self._begin_media_batch()

        self._log_topogram_xz(
            batch_np=batch_np,
            mask_vol=mask,
            captions=captions,
            epoch=epoch,
            stage=stage,
            sample_idx=0,
        )

        for view in ["axial", "coronal", "sagittal"]:
            slice_idx = self.best_slice_index(mask, view=view)
            mask2d = self._slice_xyz(mask, view=view, index=slice_idx)

            mask2d_crop = foreground_crop_2d(
                mask2d,
                mask2d,
                margin=self.crop_margin,
            )

            pred_hu_2d = foreground_crop_2d(
                self._slice_xyz(pred_hu, view=view, index=slice_idx),
                mask2d,
                margin=self.crop_margin,
            )
            pred_mu_2d = foreground_crop_2d(
                self._slice_xyz(pred_mu, view=view, index=slice_idx),
                mask2d,
                margin=self.crop_margin,
            )

            self._log_gray(
                f"{stage}/{id_}/{view}/pred_hu",
                pred_hu_2d,
                captions,
                epoch,
                mode="hu",
            )
            self._log_gray(
                f"{stage}/{id_}/{view}/pred_mu",
                pred_mu_2d,
                captions,
                epoch,
                mode="mu",
            )

            if gt_hu is not None:
                gt_hu_2d = foreground_crop_2d(
                    self._slice_xyz(gt_hu, view=view, index=slice_idx),
                    mask2d,
                    margin=self.crop_margin,
                )
                gt_mu_2d = foreground_crop_2d(
                    self._slice_xyz(gt_mu, view=view, index=slice_idx),
                    mask2d,
                    margin=self.crop_margin,
                )

                self._log_gray(
                    f"{stage}/{id_}/{view}/target_hu",
                    gt_hu_2d,
                    captions,
                    epoch,
                    mode="hu",
                )
                self._log_gray(
                    f"{stage}/{id_}/{view}/target_mu",
                    gt_mu_2d,
                    captions,
                    epoch,
                    mode="mu",
                )

                self._log_diff(
                    f"{stage}/{id_}/{view}/diff_hu",
                    (gt_hu_2d - pred_hu_2d) * (mask2d_crop > 0),
                    captions,
                    epoch,
                    vmax=300.0,
                )
                self._log_diff(
                    f"{stage}/{id_}/{view}/diff_mu",
                    (gt_mu_2d - pred_mu_2d) * (mask2d_crop > 0),
                    captions,
                    epoch,
                    vmax=0.03,
                )

                if pred_mu_smooth is not None and gt_mu_smooth is not None:
                    pred_smooth_2d = foreground_crop_2d(
                        self._slice_xyz(
                            pred_mu_smooth,
                            view=view,
                            index=slice_idx,
                        ),
                        mask2d,
                        margin=self.crop_margin,
                    )
                    gt_smooth_2d = foreground_crop_2d(
                        self._slice_xyz(
                            gt_mu_smooth,
                            view=view,
                            index=slice_idx,
                        ),
                        mask2d,
                        margin=self.crop_margin,
                    )
                    self._log_gray(
                        f"{stage}/{id_}/{view}/pred_mu_smoothed_4mm",
                        pred_smooth_2d,
                        captions,
                        epoch,
                        mode="mu",
                    )
                    self._log_gray(
                        f"{stage}/{id_}/{view}/target_mu_smoothed_4mm",
                        gt_smooth_2d,
                        captions,
                        epoch,
                        mode="mu",
                    )
                    # Do not multiply by prediction_mask here. Smoothing spreads
                    # attenuation error across the mask boundary exactly as in
                    # the reconstruction pipeline.
                    self._log_diff(
                        f"{stage}/{id_}/{view}/diff_mu_smoothed_4mm",
                        gt_smooth_2d - pred_smooth_2d,
                        captions,
                        epoch,
                        vmax=0.015,
                    )

            self._log_gray(
                f"{stage}/{id_}/{view}/prediction_mask",
                mask2d_crop,
                captions,
                epoch,
            )

            if "body_seg" in batch_np:
                body = self._extract_single_volume(batch_np["body_seg"])
                body2d = foreground_crop_2d(
                    self._slice_xyz(body, view=view, index=slice_idx),
                    mask2d,
                    margin=self.crop_margin,
                )
                self._log_label(
                    f"{stage}/{id_}/{view}/body_seg",
                    body2d,
                    captions,
                    epoch,
                )

            if "organ_seg" in batch_np:
                organ = self._extract_single_volume(batch_np["organ_seg"])
                organ2d = foreground_crop_2d(
                    self._slice_xyz(organ, view=view, index=slice_idx),
                    mask2d,
                    margin=self.crop_margin,
                )
                self._log_label(
                    f"{stage}/{id_}/{view}/organ_seg",
                    organ2d,
                    captions,
                    epoch,
                )

            self._log_input_channels(
                batch_np=batch_np,
                mask2d=mask2d,
                captions=captions,
                epoch=epoch,
                view=view,
                slice_idx=slice_idx,
                stage=stage,
                sample_idx=0,
            )

        self._flush_media_batch(epoch)
