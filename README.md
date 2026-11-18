# BIC-MAC Challenge: Multimodal-to-CT Synthesis with Conditional Feature Injection

Training and inference code of our BIC-MAC challenge entry
*"Generative Modelling with Conditional Feature Injection for Multimodal-to-CT
Synthesis and PET Attenuation Correction"*. The two-page challenge report is
included as [`BICMAC_Challenge_Report_2page_Final.pdf`](BICMAC_Challenge_Report_2page_Final.pdf).

The code is released for the collaborative BIC-MAC paper (IEEE TMI, in
preparation). It is a cleaned copy of the research code used for the
challenge; model architecture, preprocessing, losses, checkpoint format and
inference numerics are unchanged (see [Reproducibility](#reproducibility-and-verification)).

## Task and method

**Supported task (only):** whole-body synthetic CT (sCT) generation for PET
attenuation correction from the deployment-visible BIC-MAC inputs

| Input | File (per subject, `features/`) | Use |
|---|---|---|
| non-attenuation-corrected PET | `nacpet.nii.gz` | primary input (all stages) |
| Dixon MR, in-phase | `mri_combined_in_phase.nii.gz` | condition (Stage 2/3) |
| Dixon MR, out-of-phase | `mri_combined_out_phase.nii.gz` | condition (Stage 2/3) |
| topogram (X, 1, Z) | `topogram.nii.gz` | condition (Stage 2/3) |

The output is one CT volume in Hounsfield units on the NAC-PET grid. PET
reconstruction with the synthesized CT is performed by the official challenge
tooling and is **not** part of this repository.

The generator is a five-level MedNeXt-style 3D encoder-decoder
(features 32-64-128-256-512, depths 2-3-4-4-4, 26.71 M parameters with the
condition branch) trained in three stages:

1. **Stage 1 - PET prior.** PET-only pretraining on BIC-MAC plus public
   PET/CT (AutoPET FDG cases and QUADRA healthy controls). Public
   attenuation-corrected PET is stochastically degraded towards NAC-PET
   appearance. Mixed batches; 600 epochs; constant LR.
2. **Stage 2 - conditional feature injection.** The two Dixon MR channels and
   the Y-broadcast topogram form a four-scale condition pyramid
   (1/2 ... 1/16). Zero-initialised SPADE-like scale/shift/gate projections
   modulate encoder and bottleneck blocks, so the Stage-1 network is exactly
   preserved at initialisation. Five deterministic folds, 300 epochs.
3. **Stage 3 - AC-PET feature distillation.** A frozen EMA teacher receives
   aligned AC-PET (training-only privileged input) while the student receives
   NAC-PET with identical conditions and crops; normalised encoder features
   are matched with a cosine loss at five levels (weight 0.05, 50 epochs).
   The teacher is discarded at inference.

**Submitted model.** According to the official validation results in the
report (Table 3), the final system is the **Stage-2 fold-1 + fold-4
prediction ensemble** (two forward passes averaged in FP32). Stage 3 kept the
CT error similar but degraded reconstructed-PET metrics and was therefore not
used. Stage-3 training code is still provided for completeness.

## Repository structure

```
.
|-- BICMAC_Challenge_Report_2page_Final.pdf   challenge report
|-- README.md
|-- requirements.txt
|-- configs/
|   |-- default_config.yaml                   global defaults
|   |-- model/ data/ train/                   composable fragments
|   |-- stages/pretrain/mednext_l-extra.yaml   Stage-1 recipe
|   |-- stages/finetune/spade.yaml             Stage-2 recipe (submitted model)
|   |-- stages/distillation/acpet.yaml         Stage-3 recipe
|   `-- final/                                 resolved configs recorded by the actual runs
|       |-- stage1_pet_prior.yaml
|       |-- stage2_conditioning.yaml           <- inference config of the submission
|       `-- stage3_distillation.yaml
|-- scripts/
|   |-- predict.py                            inference entry point
|   |-- train.py                              training entry point (all stages)
|   |-- prepare_acpet.py                      AC-PET -> NAC-PET grid cache (Stage 3)
|   |-- fuse_checkpoints.py                   weight averaging ("model soup", explored only)
|   `-- extra_data/                           AutoPET / QUADRA-HC preparation (Stage 1)
|-- src/bicmac/
|   |-- inference.py                          preprocessing, ensemble, sliding window, I/O
|   |-- config.py                             YAML includes + merging
|   |-- trainer_fabric.py                     Lightning-Fabric training loop
|   |-- data/                                 datasets, CV folds, MONAI transforms, NAC degradation
|   |-- models/                               EncoderDecoder, losses, distillation,
|   |   |                                     PET-segmented sliding-window inferer, callbacks
|   |   `-- nets/                             MedNeXtv3 backbone + spatial condition injector
|   `-- eval_metrics/                         HU / mu metrics used for validation
`-- tests/                                    synthetic-data unit and smoke tests
```

## Installation

Tested with Python 3.10 and the package versions pinned in `requirements.txt`
(PyTorch 2.6.0 / CUDA 12.6, MONAI 1.5.2, Lightning 2.6.5).

```bash
git clone https://github.com/siyuan-mei/Bicmac-challenge-report.git
cd Bicmac-challenge-report
python -m venv .venv && source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126   # pick the build for your CUDA
pip install -r requirements.txt
```

No installation of the package itself is needed; the scripts add `src/` to
`sys.path`.

## Data layout

BIC-MAC data must be obtained from the challenge organisers; no images are
distributed here. Training expects the challenge directory layout:

```
<DATA_ROOT>/                        (or <DATA_ROOT>/bicmac_data/)
|-- train/sub-XXX/
|   |-- features/  nacpet.nii.gz  mri_combined_in_phase.nii.gz
|   |              mri_combined_out_phase.nii.gz  topogram.nii.gz  metadata.json
|   |-- ct-label/  ct.nii.gz  body_seg.nii.gz  organ_seg.nii.gz  prediction_mask.nii.gz
|   `-- pet-label/ pet.nii.gz                       (AC-PET, Stage 3 only)
`-- validation/sub-XXX/ ...                          (same layout; 8 subjects)
```

All inputs and labels of one subject share the NAC-PET voxel grid
(512 x 512 x Z, 1.5234375 x 1.5234375 x 2.0 mm in the challenge data); the
topogram is stored as (512, 1, Z). For inference only the four `features/`
images are required.

Stage-1 additionally reads `<DATA_ROOT>/extra_data/{autopet,quadra_hc}/<case>/{ct,pet,body_mask}.nii.gz`,
produced by `scripts/extra_data` (resampled to 1.5234375 x 1.5234375 x 2.0 mm,
RAS, CT-defined grid; AutoPET centre-cropped to 400 x 400; body mask = largest
CT > -500 HU component). Example:

```bash
python scripts/extra_data/preprocess_autopet.py   --input-root /path/to/autoPET   --output-root $DATA_ROOT/extra_data/autopet   --bicmac-root $DATA_ROOT/bicmac_data
python scripts/extra_data/preprocess_quadra_hc.py --input-root /path/to/QUADRA_HC --output-root $DATA_ROOT/extra_data/quadra_hc --bicmac-root $DATA_ROOT/bicmac_data
python scripts/extra_data/postprocess_resampled_extra.py --extra-root $DATA_ROOT/extra_data
```

## Inference

### Checkpoints

Model weights are **not** included in this repository and no download link is
provided at this time. Inference needs Stage-2 checkpoints saved by
`scripts/train.py` (`{"model": state_dict}`; full training checkpoints with
optimizer state are also accepted). The submission used two checkpoints, one
per fold (fold 1 and fold 4) of the Stage-2 recipe; in our pipeline the
checkpoint selected per fold is the one with the best validation mu-MAE
(`checkpoints/best_1-mae_mu_mean.ckpt`). Loading is strict for all network
tensors: a mismatching checkpoint raises an error.

### Commands

Whole directory in the challenge layout (`<input_dir>/<subject>/features/...`):

```bash
python scripts/predict.py \
    --input_dir  /path/to/challenge_val \
    --output_dir outputs/s2_fold1_fold4 \
    --config     configs/final/stage2_conditioning.yaml \
    --checkpoints /path/to/s2_fold1/best_1-mae_mu_mean.ckpt \
                  /path/to/s2_fold4/best_1-mae_mu_mean.ckpt \
    --device cuda
```

Single subject:

```bash
python scripts/predict.py \
    --features_dir /path/to/sub-XXX/features \
    --output_ct    outputs/sub-XXX/ct.nii.gz \
    --config       configs/final/stage2_conditioning.yaml \
    --checkpoints  /path/to/s2_fold1/best_1-mae_mu_mean.ckpt
```

Outputs: `<output_dir>/<subject>/ct.nii.gz`, float32 HU, same shape, affine
and header (except data type) as `nacpet.nii.gz`. Options: `--overlap`,
`--sw_batch_size`, `--subjects`, `--overwrite`, `--no_segmented` (plain
whole-volume sliding window; not the challenge setting).

### What inference does

1. **Loading.** NIfTI files are read with nibabel on their native grid; no
   resampling or reorientation (all inputs share the NAC-PET grid). NaN/Inf are set to 0.
2. **Intensity preprocessing** (per volume, identical to training):
   NAC-PET values <= 10 -> 0; MR values <= 0 -> 0; topogram values <= -50 -> -1024;
   then each channel is scaled from its 1st-98th percentile to [0, 1] and clipped.
   The topogram is broadcast along Y to the PET shape.
3. **PET-segmented sliding window.** Voxels with raw NAC-PET > 10 define the
   axial body extent (+16 slices); it is split into three overlapping Z
   segments (24-slice overlap), each with its own transaxial box from the
   0.05 %/99.95 % coordinate quantiles plus margins of 48/64/48 voxels, a
   16-voxel model context, and alignment to multiples of 16. Each segment is
   processed with MONAI Gaussian sliding-window inference
   (128^3 windows, overlap 0.5, sigma scale 0.125); neighbouring segments are
   merged with cosine weights along Z. Voxels outside all segments get the
   normalised value 0 (= -1000 HU).
4. **Ensembling.** Checkpoints are evaluated sequentially for every window and
   averaged online in FP32 (bf16 autocast on CUDA for the network itself).
5. **Inverse scaling.** HU = 3000 * y - 1000, clipped to [-1024, 3071].

## Training

Training uses Lightning Fabric. The resolved per-GPU batch size, crops and
epochs are in the stage recipes; the base learning rate is multiplied by
sqrt(number of GPUs) (`optim_cfg.scale_lr_by_world_size`). All challenge
models were trained on one node with 4 x NVIDIA H100 and `bf16-mixed`
precision; the corresponding resolved configs (after LR scaling) are stored
in `configs/final/`. Global seed: 42 (`seed`), which also fixes the
cross-validation folds.

Multi-GPU launch (as used for the challenge):

```bash
fabric run --accelerator=gpu --strategy=ddp --devices=4 --num-nodes=1 \
    scripts/train.py <arguments>
```

Single GPU: `python scripts/train.py <arguments> --set train_cfg.devices=1`
(the learning rate is then not scaled up).

Common arguments: `--data-root` (or env `DATA_ROOT`), `--run-name`,
`--workdir` (default `work_dir/bicmac/<run-name>`), `--resume`
(continue from `<workdir>/checkpoints/last.ckpt`), `--finetune-from`
(load weights only, new optimizer/schedule), `--fold k` / `--val-fold k`,
and repeated `--set key=value` OmegaConf overrides. W&B logging is on by
default (`--set train_cfg.use_wandb=false` to disable). The optional env
variable `BICMAC_MAX_RUNTIME_MINUTES` stops training cleanly and saves
`last.ckpt` before a scheduler time limit.

### Stage 1 - PET prior

```bash
fabric run --accelerator=gpu --strategy=ddp --devices=4 --num-nodes=1 scripts/train.py \
    --config stages/pretrain/mednext_l-extra --data-root $DATA_ROOT --run-name stage1
```

PET-only input; 128^3 crops, batch 4/GPU; HU-MAE + prediction-mask mu-MAE (1:1);
AdamW (betas 0.9/0.95, no weight decay), constant LR 2e-4 on 4 GPUs, 600
epochs; BIC-MAC samples are mixed into extra-data batches with probability
0.25; public PET is degraded to NAC-like appearance with probability 0.5.
The challenge models use `checkpoints/last.ckpt` of this stage.

### Stage 2 - conditioning (submitted model)

For each fold `k` in 1..5 (the submission uses folds 1 and 4):

```bash
fabric run --accelerator=gpu --strategy=ddp --devices=4 --num-nodes=1 scripts/train.py \
    --config stages/finetune/spade --data-root $DATA_ROOT --run-name s2_fold$k --fold $k \
    --finetune-from work_dir/bicmac/stage1/checkpoints/last.ckpt
```

Inputs PET + MR in/out + topogram; 128^3 crops, 2 crops x batch 2 per GPU,
sampled 50/50 from body-label 1 (trunk) and 2 (head/extremities) inside the
prediction mask; loss 0.2 HU-MAE + 1.0 masked mu-MAE + 1.0 PET-projection
loss; cosine LR 1e-4 -> 1e-5 (4 GPUs) with 10 warm-up epochs; 300 epochs;
validation every 10 epochs on the full held-out volumes; best checkpoint
by validation mu-MAE.

**Folds.** Fold 1 validates on the 8 official validation subjects; folds 2-5
validate on disjoint 8-subject subsets of the training subjects (shuffled
with seed 42); each fold trains on the remaining 67 subjects. Every run writes
its subject lists to `split_manifest.json`.

### Stage 3 - AC-PET distillation (not used in the submission)

```bash
python scripts/prepare_acpet.py --data-root $DATA_ROOT --output-dir /tmp/acpet_aligned   # optional cache
export BICMAC_ACPET_DIR=/tmp/acpet_aligned
fabric run --accelerator=gpu --strategy=ddp --devices=4 --num-nodes=1 scripts/train.py \
    --config stages/distillation/acpet --data-root $DATA_ROOT --run-name s3_fold$k --val-fold $k \
    --finetune-from work_dir/bicmac/s2_fold$k/checkpoints/best_1-mae_mu_mean.ckpt
```

Trains on the 8 AC-PET-labelled subjects of the official validation split
(for fold 1, training and validation subjects therefore coincide; the
overlap is reported in the log). Cosine LR 2e-5 -> 0 (4 GPUs), 50 epochs,
4 crops x batch 1, feature-distillation weight 0.05, EMA decay 0.999.

### Exploratory utilities

`scripts/fuse_checkpoints.py` averages checkpoint weights ("model soup").
This was evaluated during development (see the report's runtime table) but is
not the submitted configuration.

## Reproducibility and verification

* `tests/` contains synthetic-data tests (no private data needed):

  ```bash
  cd tests && python -m unittest -v
  ```

  They check the config recipes against the recorded final runs, equality of
  inference preprocessing with the MONAI validation pipeline, deterministic
  disjoint folds, zero-initialised injector identity with the PET prior,
  PET-segmented stitching, strict checkpoint loading, single/ensemble
  inference (shape, affine, spacing, dtype, HU range, FP32 averaging), the
  CLIs, and a CPU smoke run of all three training stages.
* With the original fold-1 and fold-4 checkpoints we verified locally that
  this code loads them strictly and reproduces the network output of the
  original research code bit-for-bit on a real 128^3 input patch, and that
  input preprocessing on a full validation subject matches the original
  training/validation pipeline (max. abs. difference 1.3e-7).

## Limitations

* **Final post-processing not included.** The report (Sec. 4) describes an
  additional conservative 2.5D head/body-envelope clean-up with a scanner-FOV
  constraint and a tighter, head-specific PET box, applied after
  HU conversion in the submitted container. That step is not part of the code
  snapshot released here, so outputs of `scripts/predict.py` can differ from
  the submitted volumes outside the body. It will be added once the exact
  submitted implementation has been released.
* Checkpoints and BIC-MAC data are not redistributed.
* The Stage-1 checkpoint was trained on the original 67-subject training
  split; Stage-2 folds 2-5 therefore had upstream exposure to their validation
  subjects (fold 1 does not). Fold results other than fold 1 are not strict
  nested cross-validation.
* The code was cleaned for release (package layout, configurable paths,
  removal of unused backbones/losses, one fix to optional-config handling);
  training was not re-run with this exact copy beyond the smoke tests.

## Citation

The collaborative BIC-MAC paper (IEEE Transactions on Medical Imaging) is in
preparation. Citation details will be added here once available.

## License

No license has been specified yet. Until a license is added, please contact
the authors before reusing the code. Third-party dependencies (PyTorch, MONAI,
Lightning, etc.) are used under their own licenses and are not redistributed
here.
