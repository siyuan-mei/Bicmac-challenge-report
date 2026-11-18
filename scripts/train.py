import argparse
import hashlib
import json
import math
import os
import time

import torch
import torch.backends.cudnn as cudnn
from monai.data import list_data_collate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import _paths  # noqa: F401  (adds ./src to sys.path)
from bicmac import config as configs
from bicmac.data.dataset.bicmac import BICMACDataset
from bicmac.data.dataset.bicmac_extra import BICMACExtraDataset, MixedBatchSampler
from bicmac.data.transforms import build_train_pipeline, build_val_pipeline
from bicmac.data.transforms_extra import (
    build_extra_train_pipeline,
    canonicalize_input_keys,
)
from bicmac.models.encoder_decoder import EncoderDecoder
from bicmac.models.distillation import ACPETDistillationEncoderDecoder
from bicmac.models.callbacks import CheckpointCallback, VisualizationCallback
from bicmac.trainer_fabric import Trainer


torch.set_float32_matmul_precision("high")
cudnn.benchmark = True

os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")


def run(
    config,
    workdir,
    run_name="bicmac",
    if_train=True,
    if_eval=False,
    resume=False,
    finetune_from=None,
    short_ckpt_path=None,
    full_ckpt_path=None,
    eval_split=None,
    exist_gt=False,
    require_complete=False,
):
    if resume and finetune_from:
        raise ValueError("--resume and --finetune-from are mutually exclusive.")
    if (
        if_train
        and config.train_cfg.get("require_finetune_checkpoint", False)
        and not (resume or finetune_from)
    ):
        raise ValueError(
            "This finetune config requires --finetune-from for its first run "
            "or --resume for a continued run."
        )

    root = str(_paths.REPO_ROOT)
    launch_rank = int(os.environ.get("RANK", "0"))
    launch_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    launch_rank_zero = launch_rank == 0

    def abspath(path):
        path = os.path.expanduser(str(path))
        return os.path.abspath(path if os.path.isabs(path) else os.path.join(root, path))

    workdir = os.path.realpath(abspath(workdir))
    finetune_from = abspath(finetune_from) if finetune_from else None
    smoke_no_write = bool(config.get("smoke_no_write", False))
    data_root = abspath(os.environ.get("DATA_ROOT") or config.data_cfg.data_root)
    base_dataset = str(config.data_cfg.get("base_dataset", "bicmac"))
    is_distillation = config.model_cfg.get("distillation") is not None
    extra_dataset = list(config.data_cfg.get("extra_dataset", []))
    use_extra_dataset = len(extra_dataset) > 0
    mixed_batch = bool(config.data_cfg.get("mixed_batch", False))
    cv_fold = config.data_cfg.get("cv_fold")
    cv_fold = None if cv_fold is None else int(cv_fold)
    val_cv_fold = config.data_cfg.get("val_cv_fold")
    val_cv_fold = cv_fold if val_cv_fold is None else int(val_cv_fold)
    if mixed_batch and not use_extra_dataset:
        raise ValueError("mixed_batch=True requires a non-empty extra_dataset.")
    if cv_fold is not None and use_extra_dataset:
        raise ValueError("Cross-validation is supported for BIC-MAC-only training.")
    config.data_cfg.use_extra_dataset = use_extra_dataset

    base_data_root = os.path.join(data_root, "bicmac_data")
    if not os.path.isdir(base_data_root):
        base_data_root = data_root
    # Checkpoints remain durable; disposable media and W&B staging can live in
    # node-local TMPDIR on Helma and are uploaded before the process exits.
    output_dir = os.environ.get(
        "BICMAC_MEDIA_DIR", os.path.join(workdir, "outputs")
    )
    log_root = os.environ.get(
        "BICMAC_LOG_DIR", os.path.join(workdir, "logs")
    )
    wandb_root = os.environ.get("WANDB_DIR", log_root)
    checkpoint_dir = os.path.join(workdir, "checkpoints")

    # A fresh training/finetuning run must never reuse a non-empty local run
    # directory. This keeps checkpoints, visualizations, configs, and W&B files
    # from different runs isolated even when their display names are identical.
    if (
        launch_rank_zero
        and if_train
        and not resume
        and not smoke_no_write
        and os.path.isdir(workdir)
    ):
        existing_entries = [
            name for name in os.listdir(workdir) if name not in {".DS_Store"}
        ]
        if existing_entries:
            raise FileExistsError(
                "Fresh training cannot reuse a non-empty work directory:\n"
                f"{workdir}\n"
                "Use --resume for the same run, or choose a new --run-instance/"
                "--workdir for a new run."
            )

    if if_train and resume and not smoke_no_write:
        last_checkpoint = os.path.join(checkpoint_dir, "last.ckpt")
        if not os.path.isfile(last_checkpoint):
            raise FileNotFoundError(
                "Cannot resume because last.ckpt does not exist:\n"
                f"{last_checkpoint}"
            )

    devices = config.train_cfg.get("devices", "auto")
    devices = torch.cuda.device_count() if devices == "auto" else int(devices)
    devices = max(devices, 1)
    config.train_cfg.devices = devices
    world_size = devices
    config.train_cfg.world_size = world_size

    scale_lr = bool(config.optim_cfg.get("scale_lr_by_world_size", True))
    for lr_key in ("base_lr", "injector_lr", "backbone_lr"):
        if lr_key not in config.optim_cfg:
            continue
        old_lr = float(config.optim_cfg[lr_key])
        if scale_lr:
            config.optim_cfg[lr_key] = old_lr * math.sqrt(world_size)
            print(
                f"[Info] {lr_key} scaled by sqrt({world_size}): "
                f"{old_lr:.2e} -> {config.optim_cfg[lr_key]:.2e}"
            )

    input_keys = tuple(config.data_cfg.get("input_keys", ("pet",)))
    input_keys = canonicalize_input_keys(input_keys)
    config.data_cfg.input_keys = list(input_keys)
    if use_extra_dataset:
        if "pet" not in input_keys:
            raise ValueError("Extra-data training requires pet in data_cfg.input_keys.")
        if "synthrad2025" in extra_dataset and not {"mr_in", "mr_out"}.intersection(input_keys):
            raise ValueError("synthrad2025 requires mr_in and/or mr_out in data_cfg.input_keys.")
    mask_keys = tuple(config.data_cfg.get("mask_keys", ("prediction_mask", "body_seg", "organ_seg")))
    auxiliary_input_keys = ("ac_pet",) if is_distillation else ()
    crop_size = tuple(config.train_cfg.crop_size)

    num_workers = int(config.train_cfg.get("num_workers", 0))
    if os.name == "nt" and num_workers > 2:
        print(f"[Info] Windows DataLoader workers limited: {num_workers} -> 2")
        num_workers = 2

    config.data_cfg.data_root = data_root
    config.train_cfg.num_workers = num_workers

    if not smoke_no_write:
        initialized_marker = os.path.join(workdir, ".run_initialized")
        if launch_rank_zero:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(log_root, exist_ok=True)
            os.makedirs(checkpoint_dir, exist_ok=True)
            OmegaConf.save(config, os.path.join(workdir, "config.yaml"))
            OmegaConf.save(config, os.path.join(log_root, "config.yaml"))
            with open(initialized_marker, "w", encoding="ascii") as handle:
                handle.write("initialized\n")
        elif launch_world_size > 1:
            deadline = time.monotonic() + 120.0
            while not os.path.isfile(initialized_marker):
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Rank 0 did not initialize the run directory: {workdir}"
                    )
                time.sleep(0.1)

    print("data_root:", data_root)
    print("base_data_root:", base_data_root)
    print("base_dataset:", base_dataset)
    print("extra_dataset:", extra_dataset)
    print("use_extra_dataset:", use_extra_dataset)
    print("mixed_batch:", mixed_batch)
    print("cv_fold:", cv_fold)
    print("val_cv_fold:", val_cv_fold)
    print("psma_fraction:", config.data_cfg.get("psma_fraction", 0.0))
    print("workdir:", workdir)
    print("output_dir:", output_dir)
    print("log_root:", log_root)
    print("wandb_root:", wandb_root)
    print("checkpoint_dir:", checkpoint_dir)
    print("devices:", devices)
    print("world_size:", world_size)
    print("base_lr:", config.optim_cfg.base_lr)
    if "injector_lr" in config.optim_cfg:
        print("injector_lr:", config.optim_cfg.injector_lr)
    print("optimizer:", config.optim_cfg.get("optimizer", "adamw"))
    print("weight_decay:", config.optim_cfg.get("weight_decay", 0.02))
    print("input_keys:", input_keys)
    print("mask_keys:", mask_keys)
    print("acpet_distillation:", is_distillation)

    logger = None

    if config.train_cfg.get("use_wandb", True):
        from wandb.integration.lightning.fabric import WandbLogger

        # The display name may be reused. The stable W&B ID is derived from the
        # unique local work directory, so Slurm resume jobs reconnect to exactly
        # the same run while a new run-instance receives a different ID.
        wandb_run_id = hashlib.sha1(
            os.path.realpath(workdir).encode("utf-8")
        ).hexdigest()[:16]

        if resume:
            wandb_resume_mode = "must"
        elif if_train:
            wandb_resume_mode = "never"
        else:
            # Evaluation-only runs may append to an existing run or create it if
            # no remote run exists yet.
            wandb_resume_mode = "allow"

        if not smoke_no_write:
            with open(
                os.path.join(workdir, "wandb_run_id.txt"),
                "w",
                encoding="ascii",
            ) as handle:
                handle.write(f"{wandb_run_id}\n")

        print("wandb_name:", run_name)
        print("wandb_id:", wandb_run_id)
        print("wandb_resume:", wandb_resume_mode)

        logger = WandbLogger(
            name=run_name,
            id=wandb_run_id,
            resume=wandb_resume_mode,
            project=config.train_cfg.get("wandb_project", "BICMAC"),
            save_dir=wandb_root,
            log_model=False,
            config=OmegaConf.to_container(config, resolve=True),
        )

    loader_kwargs = {
        "num_workers": num_workers,
        "collate_fn": list_data_collate,
        "persistent_workers": (
            num_workers > 0 and bool(config.train_cfg.get("persistent_workers", True))
        ),
        "pin_memory": (
            torch.cuda.is_available() and bool(config.train_cfg.get("pin_memory", True))
        ),
    }

    if num_workers > 0:
        prefetch_factor = int(config.train_cfg.get("prefetch_factor", 2))
        loader_kwargs["prefetch_factor"] = 1 if os.name == "nt" else prefetch_factor

    train_loader = None
    val_loader = None
    eval_loader = None

    train_size = 0
    val_size = 0
    eval_size = 0

    if if_train:
        if use_extra_dataset:
            train_transform = build_extra_train_pipeline(
                input_keys=input_keys,
                crop_size=crop_size,
                num_samples=config.data_cfg.get("num_samples", 1),
                pet_to_nac_prob=config.data_cfg.get("pet_to_nac_prob", 0.0),
            )
        else:
            train_transform = build_train_pipeline(
                label_keys=("ct",),
                input_keys=input_keys,
                auxiliary_input_keys=auxiliary_input_keys,
                auxiliary_inputs_aligned=bool(
                    is_distillation and os.environ.get("BICMAC_ACPET_DIR")
                ),
                mask_keys=mask_keys,
                crop_size=crop_size,
                num_samples=config.data_cfg.get("num_samples", 1),
            )

        val_transform = build_val_pipeline(
            label_keys=("ct",),
            input_keys=input_keys,
            mask_keys=mask_keys,
        )

        train_split = config.data_cfg.get("train_split", "train")
        val_split = config.data_cfg.get("val_split", "validation")

        if use_extra_dataset:
            train_dataset = BICMACExtraDataset(
                data_root=data_root,
                split_path=train_split,
                base_dataset=base_dataset,
                extra_dataset=extra_dataset,
                mixed_batch=mixed_batch,
                psma_fraction=config.data_cfg.get("psma_fraction", 0.0),
                seed=config.get("seed", 42),
                transform=train_transform,
            )
        else:
            cv_kwargs = {
                "cv_fold": cv_fold,
                "cv_seed": config.data_cfg.get("cv_seed", config.get("seed", 42)),
                "cv_num_folds": config.data_cfg.get("cv_num_folds", 5),
                "cv_val_size": config.data_cfg.get("cv_val_size", 8),
            }
            train_dataset = BICMACDataset(
                data_root=base_data_root,
                split_path=train_split,
                use_ct_labels=True,
                use_pet_labels=is_distillation,
                transform=train_transform,
                cv_role="train" if cv_fold is not None else None,
                **cv_kwargs,
            )

        val_dataset = BICMACDataset(
            data_root=base_data_root,
            split_path=val_split,
            use_ct_labels=True,
            transform=val_transform,
            cv_fold=val_cv_fold,
            cv_role="validation" if val_cv_fold is not None else None,
            cv_seed=config.data_cfg.get("cv_seed", config.get("seed", 42)),
            cv_num_folds=config.data_cfg.get("cv_num_folds", 5),
            cv_val_size=config.data_cfg.get("cv_val_size", 8),
        )

        if len(train_dataset) == 0:
            raise RuntimeError(f"Empty train dataset: {os.path.join(data_root, train_split)}")

        if len(val_dataset) == 0:
            raise RuntimeError(f"Empty val dataset: {os.path.join(data_root, val_split)}")

        def split_entries(dataset):
            return [
                {
                    "id": sample["id"],
                    "source_split": sample.get("source_split", train_split),
                }
                for sample in dataset.samples
                if sample.get("dataset", "bicmac") == "bicmac"
            ]

        train_entries = split_entries(train_dataset)
        val_entries = split_entries(val_dataset)
        train_ids = {entry["id"] for entry in train_entries}
        val_ids = {entry["id"] for entry in val_entries}
        overlap = sorted(train_ids & val_ids)
        if overlap:
            # Stage 3 intentionally trains and validates on the AC-PET-labelled
            # subjects for fold 1; the overlap is recorded, not rejected.
            print("[Warning] train/validation subject overlap:", overlap)

        split_manifest = {
            "cv_fold": cv_fold,
            "val_cv_fold": val_cv_fold,
            "train": train_entries,
            "validation": val_entries,
            "overlap": overlap,
        }
        print("train_subjects:", [entry["id"] for entry in train_entries])
        print("val_subjects:", [entry["id"] for entry in val_entries])
        if launch_rank_zero and not smoke_no_write:
            with open(
                os.path.join(workdir, "split_manifest.json"),
                "w",
                encoding="ascii",
            ) as handle:
                json.dump(split_manifest, handle, indent=2, sort_keys=True)
                handle.write("\n")

        materialize_count = int(config.train_cfg.get("materialize_train_cases", 0))
        if materialize_count > 0 and mixed_batch:
            print("[Info] materialize_train_cases ignored for mixed batches to preserve random crops.")
        elif materialize_count > 0:
            indices = (
                train_dataset.representative_indices(materialize_count)
                if hasattr(train_dataset, "representative_indices")
                else range(min(materialize_count, len(train_dataset)))
            )
            train_dataset = [train_dataset[index] for index in indices]
            print(f"materialized_train_cases: {len(train_dataset)}")

        if mixed_batch:
            batch_sampler = MixedBatchSampler(
                train_dataset,
                batch_size=config.train_cfg.get("batch_size", 1),
                base_probability=config.data_cfg.get("bicmac_mix_prob", 0.25),
                seed=config.get("seed", 42),
            )
            batch_sampler.set_distributed(world_size, 0)
            print(
                "mixed_batch_layout:",
                f"up to {batch_sampler.batch_size} samples/GPU;",
                f"bicmac_mix_prob={batch_sampler.base_probability};",
                f"{len(batch_sampler)} steps/epoch;",
                f"all {len(train_dataset.extra_samples)} extra cases exactly once",
            )
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=batch_sampler,
                **loader_kwargs,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.train_cfg.get("batch_size", 1),
                shuffle=True,
                drop_last=config.train_cfg.get("drop_last", False),
                **loader_kwargs,
            )

        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )

        train_size = len(train_dataset)
        val_size = len(val_dataset)

        print("train_split:", train_split, "num:", train_size)
        print("val_split:", val_split, "num:", val_size)

    if if_eval:
        eval_split = eval_split or config.data_cfg.get(
            "eval_split",
            "validation" if exist_gt else "challenge_val",
        )

        eval_transform = build_val_pipeline(
            label_keys=("ct",) if exist_gt else (),
            input_keys=input_keys,
            mask_keys=mask_keys if exist_gt else (),
        )
        eval_dataset = BICMACDataset(
            data_root=base_data_root,
            split_path=eval_split,
            use_ct_labels=exist_gt,
            transform=eval_transform,
            cv_fold=cv_fold if exist_gt and eval_split == "validation" else None,
            cv_role="validation" if cv_fold is not None else None,
            cv_seed=config.data_cfg.get("cv_seed", config.get("seed", 42)),
            cv_num_folds=config.data_cfg.get("cv_num_folds", 5),
            cv_val_size=config.data_cfg.get("cv_val_size", 8),
        )

        if len(eval_dataset) == 0:
            raise RuntimeError(f"Empty eval dataset: {os.path.join(data_root, eval_split)}")

        eval_loader = DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )

        eval_size = len(eval_dataset)

        print("eval_split:", eval_split, "num:", eval_size, "exist_gt:", exist_gt)

    model = (
        ACPETDistillationEncoderDecoder(config)
        if is_distillation
        else EncoderDecoder(config)
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    efficiency = {
        "efficiency/params_M": total_params / 1e6,
        "efficiency/trainable_params_M": trainable_params / 1e6,
    }

    print("params(M):", f"{efficiency['efficiency/params_M']:.3f}")
    print("trainable params(M):", f"{efficiency['efficiency/trainable_params_M']:.3f}")

    if logger is not None:
        logger.log_metrics({
            "params_total_M": total_params / 1e6,
            "params_trainable_M": trainable_params / 1e6,
        })

    callbacks = []

    if not smoke_no_write:
        save_best = config.train_cfg.get("save_best", ["1-mae_mu_mean", "4-mae_hu_mean"])
        save_best = [save_best] if isinstance(save_best, str) else list(save_best)

        save_best_mode = config.train_cfg.get("save_best_mode", ["min"] * len(save_best))
        save_best_mode = [save_best_mode] if isinstance(save_best_mode, str) else list(save_best_mode)

        if len(save_best_mode) == 1 and len(save_best) > 1:
            save_best_mode = save_best_mode * len(save_best)

        callbacks = [
            CheckpointCallback(
                interval=config.train_cfg.get("save_freq", 1),
                save_last=True,
                max_keep_ckpts=config.train_cfg.get("max_keep_ckpts", 1),
                save_best=save_best,
                mode=save_best_mode,
                dirpath=checkpoint_dir,
                specific_save_epoch=config.val_cfg.get("full_val_epoch", None),
            ),
            VisualizationCallback(
                output_dir=output_dir,
                epoch_interval=config.train_cfg.get("vis_epoch_interval", 50),
                save_input=config.train_cfg.get("save_input", False),
                save_target=config.train_cfg.get("save_target", False),
                save_png=config.train_cfg.get("save_visuals", True),
                log_to_wandb=config.train_cfg.get("use_wandb", True),
                logger=logger,
                crop_margin=config.train_cfg.get("vis_crop_margin", 8),
                save_debug_numpy=config.train_cfg.get("save_debug_numpy", False),
            ),
        ]

    trainer = Trainer(
        config=config,
        workdir=workdir,
        model=model,
        accelerator=config.train_cfg.get("accelerator", "gpu" if torch.cuda.is_available() else "cpu"),
        strategy=config.train_cfg.get("strategy", "auto"),
        devices=devices,
        precision=config.train_cfg.get("precision", "32-true"),
        loggers=logger,
        callbacks=callbacks,
        grad_accum_steps=config.train_cfg.get("grad_accum_steps", 1),
        use_distributed_sampler=config.train_cfg.get("use_distributed_sampler", True),
        checkpoint_dir=checkpoint_dir,
    )

    if if_train:
        completed = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            seed=config.get("seed", 42),
            resume=resume,
            finetune_from=finetune_from,
        )
        if completed and trainer.fabric.is_global_zero and not smoke_no_write:
            with open(os.path.join(workdir, "TRAINING_COMPLETE"), "w", encoding="ascii") as handle:
                handle.write("complete\n")
        if require_complete and not completed:
            raise RuntimeError(
                "The allocation ended before the configured training schedule "
                "completed. Submit another --resume job."
            )

    if if_eval:
        trainer.evaluate(
            test_loader=eval_loader,
            short_ckpt_path=short_ckpt_path,
            full_ckpt_path=full_ckpt_path,
            exist_gt=exist_gt,
        )

    if logger is not None:
        logger.experiment.finish()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        "--config_path",
        dest="config_path",
        nargs="+",
        default=None,
        help="One or more config fragments, merged from left to right.",
    )
    parser.add_argument("--model", default=None, help="Model fragment, e.g. mednext_l.")
    parser.add_argument("--data", default=None, help="Data fragment, e.g. pet_all_target.")
    parser.add_argument("--train", default=None, help="Training fragment, e.g. finetune.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="OmegaConf dot-list override; repeat as needed.",
    )
    parser.add_argument(
        "--run-name",
        "--run_name",
        dest="run_name",
        default="bicmac",
        help="W&B display name. Duplicate display names are allowed.",
    )
    parser.add_argument(
        "--run-instance",
        "--run_instance",
        dest="run_instance",
        default=None,
        help=(
            "Unique local run directory name. Reuse it only with --resume. "
            "Defaults to --run-name for backward compatibility."
        ),
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default=None,
        help="Explicit work directory; overrides --run-instance.",
    )
    parser.add_argument("--data-root", "--data_root", dest="data_root", default=None)
    parser.add_argument(
        "--psma-fraction",
        "--psma_fraction",
        dest="psma_fraction",
        type=float,
        default=None,
        help="Fraction of available AutoPET PSMA cases to include (0 to 1).",
    )
    parser.add_argument(
        "--bicmac-mix-prob",
        dest="bicmac_mix_prob",
        type=float,
        default=None,
        help="Probability of adding one BIC-MAC sample to an extra-data batch.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        choices=range(1, 6),
        default=None,
        help="Optional deterministic BIC-MAC cross-validation fold (1-5).",
    )
    parser.add_argument(
        "--val-fold",
        type=int,
        choices=range(1, 6),
        default=None,
        help="Validation fold override; leaves the training split unchanged.",
    )
    parser.add_argument("--no-train", "--no_train", dest="no_train", action="store_true")
    parser.add_argument("--eval", "--if_eval", dest="if_eval", action="store_true")
    parser.add_argument("--eval-split", "--eval_split", dest="eval_split", default=None)
    parser.add_argument("--ground-truth", "--exist_gt", dest="exist_gt", action="store_true")

    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail the job if its training schedule is still incomplete.",
    )
    parser.add_argument(
        "--finetune-from",
        "--finetune_from",
        dest="finetune_from",
        default=None,
        help="Load model weights only and start a new optimizer/schedule.",
    )
    parser.add_argument("--short-checkpoint", "--short_ckpt_path", dest="short_ckpt_path", default=None)
    parser.add_argument("--checkpoint", "--full_ckpt_path", dest="full_ckpt_path", default=None)

    args = parser.parse_args()

    config_names = list(args.config_path or [])
    if not config_names and not any((args.model, args.data, args.train)):
        config_names.append("stages/pretrain/mednext_l-extra")
    for prefix, value in (
        ("model", args.model),
        ("data", args.data),
        ("train", args.train),
    ):
        if value:
            config_names.append(f"{prefix}/{value}")
    config = configs.load_and_merge_config(config_names)
    if args.overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))

    if args.data_root is not None:
        config.data_cfg.data_root = args.data_root
    if args.psma_fraction is not None:
        config.data_cfg.psma_fraction = args.psma_fraction
    if args.bicmac_mix_prob is not None:
        config.data_cfg.bicmac_mix_prob = args.bicmac_mix_prob
    if args.fold is not None:
        config.data_cfg.cv_fold = args.fold
    if args.val_fold is not None:
        config.data_cfg.val_cv_fold = args.val_fold
    root = str(_paths.REPO_ROOT)
    run_instance = args.run_instance or args.run_name
    workdir = args.workdir or os.path.join(
        root,
        "work_dir",
        "bicmac",
        run_instance,
    )

    run(
        config=config,
        workdir=workdir,
        run_name=f"bicmac/{args.run_name}",
        if_train=not args.no_train,
        if_eval=args.if_eval,
        resume=args.resume,
        finetune_from=args.finetune_from,
        short_ckpt_path=args.short_ckpt_path,
        full_ckpt_path=args.full_ckpt_path,
        eval_split=args.eval_split,
        exist_gt=args.exist_gt,
        require_complete=args.require_complete,
    )


if __name__ == "__main__":
    main()
