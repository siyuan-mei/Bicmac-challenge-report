import os
import json
import time
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Iterable, List, Literal, Optional, Tuple, Union, cast

import numpy as np
import torch
import lightning as L
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.loggers import Logger
from lightning.fabric.strategies import Strategy
from lightning_utilities import apply_to_collection
from tqdm import tqdm

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


def cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class Trainer:
    def __init__(
        self,
        config,
        workdir,
        model,
        accelerator: Union[str, Accelerator] = "auto",
        strategy: Union[str, Strategy] = "auto",
        devices: Union[List[int], str, int] = "auto",
        precision: Union[str, int] = "32-true",
        loggers: Optional[Union[Logger, List[Logger]]] = None,
        callbacks: Optional[Union[List[Any], Any]] = None,
        grad_accum_steps: int = 1,
        use_distributed_sampler: bool = True,
        checkpoint_dir: str = "./checkpoints",
    ) -> None:
        self.config = config

        self.fabric = L.Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            precision=precision,
            callbacks=callbacks,
            loggers=loggers,
        )

        self.workdir = workdir
        self.model = model
        self.wandb_logger = loggers

        self.grad_accum_steps = max(int(grad_accum_steps), 1)
        self.max_epochs = config.train_cfg.num_epochs
        self.max_steps = cfg_get(config.train_cfg, "max_steps", None)

        self.global_step = 0
        self.current_epoch = 0
        self.should_stop = False
        runtime_minutes = os.environ.get("BICMAC_MAX_RUNTIME_MINUTES")
        self.max_runtime_seconds = (
            60.0 * float(runtime_minutes) if runtime_minutes else None
        )
        self.started_at = time.monotonic()

        self.limit_train_batches = config.train_cfg.limit_train_batches
        self.limit_val_batches = config.val_cfg.limit_val_batches
        self.limit_test_batches = cfg_get(config.val_cfg, "limit_test_batches", float("inf"))

        if not isinstance(self.limit_train_batches, int):
            assert self.limit_train_batches == float("inf")

        if not isinstance(self.limit_val_batches, int):
            assert self.limit_val_batches == float("inf")

        if not isinstance(self.limit_test_batches, int):
            assert self.limit_test_batches == float("inf")

        self.full_val_epoch = cfg_get(config.val_cfg, "full_val_epoch", [self.max_epochs])
        if self.full_val_epoch is None:
            self.full_val_epoch = []

        self.start_val = config.val_cfg.start_val
        self.validation_frequency = config.val_cfg.val_freq
        self.use_distributed_sampler = use_distributed_sampler

        self.checkpoint_dir = checkpoint_dir
        self.grad_clip = cfg_get(config.optim_cfg, "grad_clip", None)
        self._current_train_return: Union[torch.Tensor, Mapping[str, Any]] = {}
        self._current_val_return: Optional[Mapping[str, Any]] = {}
        self._train_epoch_completed = True

        if self.max_runtime_seconds is not None and self.fabric.is_global_zero:
            print(
                "[Trainer] Runtime guard: "
                f"{self.max_runtime_seconds / 60.0:.0f} minutes"
            )

    def _init_wandb_metrics(self):
        if not self.fabric.is_global_zero:
            return

        if self.wandb_logger is None:
            return

        try:
            wandb_run = self.fabric.logger.experiment

            wandb_run.define_metric("epoch/epochs")
            wandb_run.define_metric("step/steps")

            for prefix in [
                "train/*",
                "epoch/*",
                "media_train/*",
                "media_val/*",
                "media_test/*",
            ]:
                wandb_run.define_metric(prefix, step_metric="epoch/epochs")

            wandb_run.define_metric(
                "validation/*",
                step_metric="epoch/epochs",
                step_sync=True,
            )

            for prefix in ["step/*"]:
                wandb_run.define_metric(prefix, step_metric="step/steps")

            wandb_run.log(
                {
                    "epoch/epochs": self.current_epoch,
                    "step/steps": self.global_step,
                }
            )

            print(
                "[W&B] Validation uses direct W&B logging with "
                "`epoch/epochs` as x-axis."
            )

        except Exception as e:
            print(f"[W&B] Metric initialization skipped: {e}")

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader],
        seed=42,
        resume=False,
        finetune_from=None,
    ):
        if resume and finetune_from:
            raise ValueError("resume and finetune_from are mutually exclusive.")

        train_dir = os.path.join(self.workdir, "train")
        if not self.config.get("smoke_no_write", False):
            os.makedirs(train_dir, exist_ok=True)

        self.fabric.seed_everything(seed + self.fabric.global_rank, workers=True)

        if self.wandb_logger is not None:
            self._init_wandb_metrics()

        mixed_sampler = getattr(train_loader, "batch_sampler", None)
        if getattr(mixed_sampler, "is_distributed", False):
            mixed_sampler.set_distributed(self.fabric.world_size, self.fabric.global_rank)

        train_loader = self.fabric.setup_dataloaders(
            train_loader,
            use_distributed_sampler=(
                self.use_distributed_sampler
                and not getattr(mixed_sampler, "is_distributed", False)
            ),
        )

        if val_loader is not None:
            val_loader = self.fabric.setup_dataloaders(
                val_loader,
                use_distributed_sampler=self.use_distributed_sampler,
            )

        num_batches = self._num_limited_batches(train_loader, self.limit_train_batches)
        optimizer_steps_per_epoch = (
            num_batches + self.grad_accum_steps - 1
        ) // self.grad_accum_steps
        optimizer, scheduler_cfg = self._parse_optimizers_schedulers(
            self.model.configure_optimizers_schedulers(optimizer_steps_per_epoch)
        )

        if optimizer is None:
            raise RuntimeError("configure_optimizers_schedulers must return an optimizer.")

        if finetune_from:
            if not os.path.isfile(finetune_from):
                raise FileNotFoundError(f"Finetune checkpoint not found: {finetune_from}")
            self._load_finetune_weights(self.model, finetune_from)

        model, optimizer = self.fabric.setup(self.model, optimizer)

        if self.wandb_logger is not None and hasattr(self.wandb_logger, "watch"):
            self.wandb_logger.watch(model)

        state = {
            "model": model,
            "optim": optimizer,
            "epoch": self.current_epoch,
            "step": self.global_step,
        }
        if scheduler_cfg is not None:
            state["scheduler"] = scheduler_cfg["scheduler"]
        if resume:
            latest_checkpoint_path = self.get_latest_checkpoint(self.checkpoint_dir)
            if latest_checkpoint_path is None:
                raise FileNotFoundError(
                    f"--resume requested but no checkpoint found in {self.checkpoint_dir}"
                )
            self.load(state, latest_checkpoint_path)

            if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                self.should_stop = True
            if self.max_steps is not None and self.global_step >= self.max_steps:
                self.should_stop = True

        while (
            not self.should_stop
            and (self.max_epochs is None or self.current_epoch < self.max_epochs)
        ):

            if getattr(mixed_sampler, "is_distributed", False):
                mixed_sampler.set_epoch(self.current_epoch)

            train_epoch_loss = self.train_loop(
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                limit_batches=self.limit_train_batches,
                scheduler_cfg=scheduler_cfg,
            )

            saved_epoch = (
                self.current_epoch
                if self._train_epoch_completed
                else self.current_epoch - 1
            )
            state.update(step=self.global_step, epoch=saved_epoch)

            if val_loader is not None and not self.should_stop and self.should_validate:
                limit_batches = (
                    float("inf")
                    if self.current_epoch in self.full_val_epoch
                    else self.limit_val_batches
                )

                self.val_loop(
                    model=model,
                    val_loader=val_loader,
                    limit_batches=limit_batches,
                )

                self.fabric.call(
                    "checkpoint_on_validation_epoch_end",
                    fabric=self.fabric,
                    state=state,
                    metrics=self._current_val_return,
                )

            self.fabric.call(
                "checkpoint_on_train_epoch_end",
                fabric=self.fabric,
                state=state,
            )

            if self._train_epoch_completed:
                self.step_scheduler(
                    scheduler_cfg=scheduler_cfg,
                    level="epoch",
                    current_value=self.current_epoch,
                )

            self.fabric.log_dict(
                {
                    "epoch/epochs": self.current_epoch,
                    **self._lr_metrics(optimizer, "epoch"),
                    **train_epoch_loss,
                },
                step=self.current_epoch,
            )

            if self._train_epoch_completed:
                self.current_epoch += 1

        completed = (
            (self.max_epochs is not None and self.current_epoch >= self.max_epochs)
            or (self.max_steps is not None and self.global_step >= self.max_steps)
        )
        self.should_stop = False
        return completed

    def evaluate(
        self,
        test_loader,
        short_ckpt_path=None,
        full_ckpt_path=None,
        exist_gt=False,
    ):
        eval_dir = os.path.join(self.workdir, "eval")
        os.makedirs(eval_dir, exist_ok=True)

        test_loader = self.fabric.setup_dataloaders(
            test_loader,
            use_distributed_sampler=self.use_distributed_sampler,
        )

        model = self.fabric.setup(self.model)

        if self.wandb_logger is not None and hasattr(self.wandb_logger, "watch"):
            self.wandb_logger.watch(model)

        state = {"model": model}

        if short_ckpt_path is None and full_ckpt_path is None:
            raise ValueError("Either short_ckpt_path or full_ckpt_path must be provided.")

        ckpt_path = full_ckpt_path or os.path.join(self.checkpoint_dir, short_ckpt_path)
        print("ckpt_path:", ckpt_path)

        remainder = self.fabric.load(ckpt_path, state, strict=False, weights_only=False)
        if remainder:
            remainder.pop("ema", None)
        if remainder:
            print(f"[Info] Evaluation ignored checkpoint state: {sorted(remainder)}")

        if not exist_gt:
            self.test_loop(model, test_loader, self.limit_test_batches)
        else:
            self.val_loop(model, test_loader)

            if self.fabric.is_global_zero:
                with open(os.path.join(eval_dir, "results.txt"), "w") as f:
                    json.dump(self._current_val_return, f, indent=4)

    def train_loop(
        self,
        model: L.LightningModule,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        limit_batches: Union[int, float] = float("inf"),
        scheduler_cfg: Optional[Mapping[str, Any]] = None,
    ):
        num_batches = self._num_limited_batches(train_loader, limit_batches)

        iterable = self.progbar_wrapper(
            train_loader,
            total=num_batches,
            desc=f"Epoch {self.current_epoch}",
        )

        optimizer.zero_grad(set_to_none=True)

        epoch_loss_accumulator = defaultdict(float)
        processed_batches = 0

        for batch_idx, batch in enumerate(iterable):
            if self.should_stop or batch_idx >= num_batches:
                break

            processed_batches += 1

            is_last_batch = processed_batches == num_batches
            should_optim_step = (
                processed_batches % self.grad_accum_steps == 0
                or is_last_batch
            )

            window_start = (
                (processed_batches - 1) // self.grad_accum_steps
            ) * self.grad_accum_steps
            window_size = min(self.grad_accum_steps, num_batches - window_start)
            loss_scale = 1.0 / float(window_size)

            with self.fabric.no_backward_sync(model, enabled=not should_optim_step):
                self.training_step(
                    model=model,
                    batch=batch,
                    batch_idx=batch_idx,
                    loss_scale=loss_scale,
                )

            if should_optim_step:
                if self.grad_clip is not None and self.grad_clip > 0:
                    self.fabric.clip_gradients(
                        model,
                        optimizer,
                        max_norm=self.grad_clip,
                        error_if_nonfinite=False
                    )

                optimizer_stepped = self._optimizer_step(optimizer)
                optimizer.zero_grad(set_to_none=True)

                if optimizer_stepped:
                    self.global_step += 1
                    self.step_scheduler(
                        scheduler_cfg=scheduler_cfg,
                        level="step",
                        current_value=self.global_step,
                    )
                    self.fabric.log_dict(
                        {
                            "step/steps": self.global_step,
                            **self._lr_metrics(optimizer, "step"),
                        },
                        step=self.global_step,
                    )
                elif self.fabric.is_global_zero:
                    print("[AMP] Non-finite gradients: optimizer and scheduler step skipped.")

            for k, v in self._current_train_return.items():
                epoch_loss_accumulator[k] += self._to_float(v)

            self._format_iterable(iterable, self._current_train_return, "train")

            self.fabric.log_dict(
                {
                    "step/steps": self.global_step,
                    **{
                        f"step/{k}": self._to_float(v)
                        for k, v in self._current_train_return.items()
                    },
                },
                step=self.global_step,
            )

            if self.max_steps is not None and self.global_step >= self.max_steps:
                self.should_stop = True
                break

            if (
                should_optim_step
                and self.max_runtime_seconds is not None
                and time.monotonic() - self.started_at >= self.max_runtime_seconds
            ):
                if self.fabric.is_global_zero:
                    print("[Trainer] Runtime guard reached; saving last.ckpt and stopping.")
                self.should_stop = True
                break

        denominator = max(processed_batches, 1)
        self._train_epoch_completed = processed_batches == num_batches

        return {
            f"epoch/{k}": v / denominator
            for k, v in epoch_loss_accumulator.items()
        }

    def training_step(
        self,
        model: L.LightningModule,
        batch: Any,
        batch_idx: int,
        loss_scale: float = 1.0,
    ):
        with self.fabric.autocast():
            outputs = model.training_step(batch)

        if isinstance(outputs, torch.Tensor):
            out_batch = batch
            loss = outputs
            loss_items = {"total_loss": loss}
        else:
            out_batch, loss_items = outputs

            if isinstance(loss_items, torch.Tensor):
                loss = loss_items
                loss_items = {"total_loss": loss}
            elif isinstance(loss_items, Mapping):
                if "total_loss" in loss_items:
                    loss = loss_items["total_loss"]
                elif "total" in loss_items:
                    loss = loss_items["total"]
                else:
                    raise KeyError("loss_items must contain `total_loss` or `total`.")
            else:
                raise TypeError(f"Unsupported training_step output type: {type(loss_items)}")

        self.fabric.backward(loss * loss_scale)

        self._current_train_return = apply_to_collection(
            loss_items,
            dtype=torch.Tensor,
            function=lambda x: x.detach().float().cpu(),
        )

        self.fabric.call(
            "vis_on_train_batch_end",
            self.current_epoch,
            out_batch,
            batch_idx,
            self.fabric.is_global_zero,
        )

        return loss

    def val_loop(
        self,
        model: L.LightningModule,
        val_loader: Optional[torch.utils.data.DataLoader],
        limit_batches: Union[int, float] = float("inf"),
    ):
        if val_loader is None:
            return

        was_training = model.training
        model.eval()

        num_batches = self._num_limited_batches(val_loader, limit_batches)

        iterable = self.progbar_wrapper(
            val_loader,
            total=num_batches,
            desc="Validation",
        )

        overall_metrics = defaultdict(list)

        with torch.no_grad():
            for batch_idx, batch in enumerate(iterable):
                if batch_idx >= num_batches:
                    break

                with self.fabric.autocast():
                    out_batch, acc_dict = model.validation_step(batch)

                acc_dict = apply_to_collection(
                    acc_dict,
                    dtype=torch.Tensor,
                    function=lambda x: x.detach(),
                )

                self._format_iterable(iterable, acc_dict, "val")

                self.fabric.call(
                    "vis_on_validation_batch_end",
                    self.current_epoch,
                    out_batch,
                    batch_idx,
                    self.fabric.is_global_zero,
                )

                for key, value in acc_dict.items():
                    for synced_value in self._sync_scalar(value):
                        overall_metrics[key].append(synced_value)

        self._current_val_return = {}

        for key, values in overall_metrics.items():
            values_np = np.asarray(values, dtype=np.float32)

            mean = float(np.nanmean(values_np))
            std = float(np.nanstd(values_np))

            self._current_val_return[f"{key}_mean"] = mean
            self._current_val_return[f"{key}_std"] = std

        val_payload = {
            "epoch/epochs": self.current_epoch,
            **{
                f"validation/{k}": v
                for k, v in self._current_val_return.items()
            },
        }
        if self.wandb_logger is not None:
            if self.fabric.is_global_zero:
                self.fabric.logger.experiment.log(val_payload)
        else:
            self.fabric.log_dict(val_payload, step=self.current_epoch)

        if was_training:
            model.train()

    def test_loop(
        self,
        model: L.LightningModule,
        test_loader: Optional[torch.utils.data.DataLoader],
        limit_batches: Union[int, float] = float("inf"),
    ):
        if test_loader is None:
            return

        was_training = model.training
        model.eval()

        num_batches = self._num_limited_batches(test_loader, limit_batches)
        iterable = self.progbar_wrapper(
            test_loader,
            total=num_batches,
            desc="Test",
        )

        test_started_at = time.perf_counter()
        with torch.no_grad():
            for batch_idx, batch in enumerate(iterable):
                if batch_idx >= num_batches:
                    break
                case_started_at = time.perf_counter()
                with self.fabric.autocast():
                    out_batch = model.test_step(batch)

                self.fabric.call(
                    "vis_on_validation_batch_end",
                    self.current_epoch,
                    out_batch,
                    batch_idx,
                    self.fabric.is_global_zero,
                    True,
                    False,
                )
                if self.fabric.is_global_zero:
                    case_id = batch.get("id", f"index-{batch_idx}")
                    if isinstance(case_id, (list, tuple)):
                        case_id = case_id[0]
                    elapsed = time.perf_counter() - case_started_at
                    print(f"[Timing] {case_id}: {elapsed:.2f} s (inference + save)")

        if self.fabric.is_global_zero:
            elapsed = time.perf_counter() - test_started_at
            print(f"[Timing] challenge inference total: {elapsed:.2f} s")

        if was_training:
            model.train()

    def step_scheduler(
        self,
        scheduler_cfg: Optional[Mapping[str, Any]],
        level: Literal["step", "epoch"],
        current_value: int,
    ) -> None:
        if scheduler_cfg is None:
            return

        if scheduler_cfg.get("interval") != level:
            return

        frequency = cast(int, scheduler_cfg.get("frequency", 1))
        if frequency > 1 and current_value % frequency != 0:
            return

        scheduler = scheduler_cfg["scheduler"]
        scheduler_type = scheduler_cfg.get("type", None)

        if scheduler_type == "cosine":
            scheduler.step()
            return

        if scheduler_type in {"poly", "warmup"}:
            scheduler.step(current_value)
            return

        if scheduler_type == "multistep":
            scheduler.step(self.current_epoch)
            return

        if scheduler_type == "plateau":
            monitor = scheduler_cfg.get("monitor", None)
            metric = self._get_monitor_value(monitor)

            if metric is None:
                if self.fabric.is_global_zero:
                    print(f"[Trainer] Plateau scheduler skipped: monitor `{monitor}` not found.")
                return

            scheduler.step(self.current_epoch, metric=metric)
            return

        raise ValueError(f"Scheduler {scheduler_type} not defined.")

    @property
    def should_validate(self) -> bool:
        return (
            (self.current_epoch % self.validation_frequency == 0
             or self.current_epoch == self.max_epochs - 1)
            and self.current_epoch >= self.start_val
        )

    def progbar_wrapper(self, iterable: Iterable, total: int, **kwargs: Any):
        if self.fabric.is_global_zero:
            return tqdm(iterable, total=total, **kwargs)
        return iterable

    def load(self, state: Optional[Mapping], path: str) -> None:
        if state is None:
            state = {}

        remainder = self.fabric.load(path, state, weights_only=False)
        if remainder:
            remainder.pop("ema", None)

        self.global_step = int(state.get("step", 0))
        self.current_epoch = int(state.get("epoch", -1)) + 1

        if remainder:
            raise RuntimeError(f"Unused checkpoint values: {remainder}")

    def _load_finetune_weights(self, model, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        weights = checkpoint.get("model", checkpoint)
        if not isinstance(weights, Mapping):
            raise TypeError(f"Checkpoint does not contain a model state dict: {path}")

        incompatible = model.load_state_dict(weights, strict=False)
        invalid_missing = [
            key for key in incompatible.missing_keys if "injector." not in key
        ]
        if invalid_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Finetune checkpoint is not backbone-compatible. "
                f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
            )

        if self.fabric.is_global_zero:
            loaded = len(model.state_dict()) - len(incompatible.missing_keys)
            print(f"finetuning weights from: {path}")
            print(
                "[Finetune] Loaded "
                f"{loaded}/{len(model.state_dict())} tensors; "
                f"initialized {len(incompatible.missing_keys)} injector tensors."
            )

    @staticmethod
    def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
        if not os.path.isdir(checkpoint_dir):
            return None

        files = os.listdir(checkpoint_dir)
        items = [
            os.path.join(checkpoint_dir, f)
            for f in files
            if f.startswith("epoch") or f.startswith("last")
        ]

        if not items:
            return None

        latest = max(items, key=os.path.getmtime)
        print("resuming checkpoint from:", latest)

        return latest

    def _parse_optimizers_schedulers(
        self,
        configure_optim_output,
    ) -> Tuple[Optional[torch.optim.Optimizer], Optional[Mapping[str, Any]]]:
        if isinstance(configure_optim_output, torch.optim.Optimizer):
            return configure_optim_output, None

        if isinstance(configure_optim_output, LRScheduler):
            return None, {
                "type": "torch",
                "scheduler": configure_optim_output,
                "interval": "epoch",
                "frequency": 1,
                "monitor": None,
            }

        if isinstance(configure_optim_output, Mapping):
            return None, configure_optim_output

        if isinstance(configure_optim_output, (list, tuple)):
            if len(configure_optim_output) == 1:
                return self._parse_optimizers_schedulers(configure_optim_output[0])

            if len(configure_optim_output) == 2:
                opt, _ = self._parse_optimizers_schedulers(configure_optim_output[0])
                _, sched = self._parse_optimizers_schedulers(configure_optim_output[1])
                return opt, sched

            if all(isinstance(x, torch.optim.Optimizer) for x in configure_optim_output):
                raise NotImplementedError("This trainer supports only one optimizer.")

        return None, None

    def _get_monitor_value(self, monitor):
        if monitor is None or self._current_val_return is None:
            return None

        keys = [
            monitor,
            monitor.replace("validation/", ""),
            monitor.replace("val/", ""),
            f"{monitor}_mean",
            f"{monitor.replace('validation/', '')}_mean",
            f"{monitor.replace('val/', '')}_mean",
        ]

        for key in keys:
            if key in self._current_val_return:
                return float(self._current_val_return[key])

        return None

    def _sync_scalar(self, value):
        value = self._to_float(value)
        tensor = torch.tensor(value, device=self.fabric.device, dtype=torch.float32)

        try:
            gathered = self.fabric.all_gather(tensor)
            return gathered.reshape(-1).detach().cpu().tolist()
        except Exception:
            return [value]

    @staticmethod
    def _to_float(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().float()
            if value.numel() > 1:
                value = value.mean()
            return float(value.cpu().item())

        return float(value)

    @staticmethod
    def _lr_metrics(optimizer, prefix):
        metrics = {f"{prefix}/lr": optimizer.param_groups[0]["lr"]}
        for group in optimizer.param_groups:
            name = group.get("name")
            if name in {"backbone", "injector", "refiner"}:
                metrics[f"{prefix}/lr_{name}"] = group["lr"]
        return metrics

    def _optimizer_step(self, optimizer):
        scaler = getattr(self.fabric.strategy.precision, "scaler", None)
        scale_before = scaler.get_scale() if scaler is not None else None
        optimizer.step()
        return scaler is None or scaler.get_scale() >= scale_before

    @staticmethod
    def _num_limited_batches(loader, limit_batches):
        if limit_batches == float("inf"):
            return len(loader)

        return min(len(loader), int(limit_batches))

    @staticmethod
    def _format_iterable(
        prog_bar,
        candidates: Optional[
            Union[torch.Tensor, Mapping[str, Union[torch.Tensor, float, int]]]
        ],
        prefix: str,
    ):
        if not isinstance(prog_bar, tqdm) or candidates is None:
            return

        postfix = ""

        if isinstance(candidates, torch.Tensor):
            value = candidates.detach().float()
            if value.numel() > 1:
                value = value.mean()
            postfix += f" {prefix}_loss: {value.item():.4f}"

        elif isinstance(candidates, Mapping):
            for k, v in candidates.items():
                if isinstance(v, torch.Tensor):
                    v = v.detach().float()
                    if v.numel() > 1:
                        v = v.mean()
                    v = v.item()

                if isinstance(v, (float, int)):
                    postfix += f" {prefix}_{k}: {float(v):.4f}"

        if postfix:
            prog_bar.set_postfix_str(postfix)
