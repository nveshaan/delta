# DELTA

## Setup

```bash
git clone https://github.com/nveshaan/delta.git
cd delta
uv sync
uv run python utils/patch_mlflow_ui.py
```

## Data

The datasets have to be downloaded externally and are to be placed in the `data/` folder in the following way:

```bash
./data
├── chest
│   ├── cardiomegaly_set
│   ├── covid_affected_and_normal
│   ├── COVID-19_Radiography_Dataset
│   ├── emphysema
│   ├── RSNA
│   └── VinCXR
├── fundus
│   ├── AMDNet23 Dataset
│   ├── Diabetic Retinopathy and Retinal Diseases Fundus I
│   ├── Eye Disease Image Dataset
│   ├── G1020
│   ├── LAG
│   ├── ORIGA
│   └── Retinal Fundus Images
├── mri
│   ├── Alzheimer_s Dataset
│   ├── Br35H
│   ├── Brain Tumor MRI Dataset
│   ├── brain_tumor_dataset
│   ├── BrainTumor
│   ├── BraTS2021
│   ├── brisc2025
│   ├── Epic and CSCR hospital Dataset
│   └── OASIS Alzheimer's Detection
└── oct
    ├── ARMD OCT
    ├── OCT2017
    ├── Retinal OCT images
    ├── RetinalOCT_Dataset
    └── ZhangLabData OCT
```

## Generate embeddings

Generate embeddings with all configured encoders and modalities with:

```bash
uv run python scripts/generate_embeddings.py
```

Useful command-line options include:

```bash
# Run one encoder and one modality.
uv run python scripts/generate_embeddings.py \
    encoder=CLIP \
    modality=chest

# Process one configured parent dataset.
uv run python scripts/generate_embeddings.py \
    encoder=BiomedCLIP \
    modality=fundus \
    dataset="AMDNet23 Dataset"

# Override the batch size, device, or regenerate existing files.
uv run python scripts/generate_embeddings.py \
    runtime.batch_size=8 \
    runtime.device=cpu \
    overwrite=true
```

Use `uv run python scripts/generate_embeddings.py --help` for all command-line options.

The generated files are written to
`data/<modality>/<dataset_name>/<encoder>_embeds.npy` and
`data/<modality>/<dataset_name>/labels.npy`.

> **MedSigLIP** is a gated model. Before running this, request access on its model page on the HF Hub, then authenticate locally with:
> `hf auth login`
> (or set the `HF_TOKEN` environment variable).

## Pipeline

All three scripts are Hydra apps. Every key below can be set with `key=value`,
and `-m` sweeps comma-separated values (`modality=chest,oct`). Use `--cfg job`
to print the resolved config, or `--help` to list the config groups.

| Argument | Default | Values |
| --- | --- | --- |
| `modality` | `chest` | `chest`, `fundus`, `mri`, `oct` (`configs/modality/`) |
| `method` | `fuse` | `fuse`, `knnvote_msde`, `laplacianshot`, `laplacianshot_msde` (`configs/method/`) |
| `encoder` | `CLIP` | `MedImageInsight`, `MedSigLIP`, `BiomedCLIP`, `UniMedCLIP`, `CLIP` |
| `support_size` | `5` | labeled support samples per class |
| `seed` | `42` | |
| `device` | `auto` | `auto`, `cpu`, `cuda`, `mps` |
| `campaign` | `adhoc` | free-form label for grouping runs (`tags.campaign`); not hashed |
| `method.<key>` | | any key of the method's YAML, e.g. `method.confidence=0.9`, `method.mode=no_msde` |
| `modality.<key>` | | any key of the modality's YAML, e.g. `modality.max_dataset_size=5000` |
| `mlflow.experiment_name` | `delta` | also `mlflow.tracking_uri`, `mlflow.artifact_location`, `mlflow.run_name` |

### Few-Shot Pseudo Labels

```bash
uv run python scripts/few_shot.py modality=chest method=laplacianshot_msde support_size=5
```

Arguments (also accepted by `distill_mlp.py`):

| Argument | Default | Values |
| --- | --- | --- |
| `split` | `train` | dataset split to pseudolabel |
| `support_sizes` | `null` | list, e.g. `support_sizes=[5,10]`; runs each and returns the mean objective |
| `modalities` | `null` | list, e.g. `modalities=[chest,oct]`; same as above |
| `redo` / `--redo` | `false` | delete an identical finished run and run again |
| `+experiment=optuna_method` | | stage-1 Optuna sweep over the method's section of `configs/search_space.yaml` (use with `-m`; sets `campaign=optuna`) |

### Distillation

```bash
# Missing pseudolabels are generated (as a few_shot run) automatically.
uv run python scripts/distill_mlp.py modality=chest method=laplacianshot_msde support_size=5 mlp_targets=labels
uv run python scripts/distill_mlp.py modality=chest method=laplacianshot_msde support_size=5 mlp_targets=scores
uv run python scripts/distill_mlp.py modality=chest method=laplacianshot_msde support_size=5 mlp_targets=distances
```

Arguments (in addition to the few-shot ones):

| Argument | Default | Values |
| --- | --- | --- |
| `mlp_targets` | `scores` | `labels` (pseudolabels), `scores` (same_label MSDE + GDE), `distances` (zero_label MSDE) |
| `distill_mlp` | `true` | `false` skips MLP training and evaluates the targets directly |
| `msde.<key>` | | `k`, `nbd_sample_count_threshold`, `learning_rate`, `max_iters_shift`, `shift_threshold`, ... (ignored for `labels`) |
| `training.<key>` | | `epochs`, `learning_rate`, `weight_decay`, `validation_fraction`, `patience` |
| `mlp._target_` | `models.VanillaNetwork` | |
| `+experiment=optuna_msde` | | stage-2 Optuna sweep over `msde` in `configs/search_space.yaml` (use with `-m`; sets `campaign=optuna`) |

### Zero-Shot Evaluation

```bash
uv run python scripts/zero_shot.py \
    method=laplacianshot_msde \
    modality=chest \
    encoder=CLIP \
    support_size=5 \
    mlp_targets=scores

# Evaluate a specific distill_mlp run.
uv run python scripts/zero_shot.py method=laplacianshot_msde modality=chest distill_run_id=<run id>
```

Arguments:

| Argument | Default | Values |
| --- | --- | --- |
| `split` | `test` | dataset split to evaluate |
| `mlp_targets` | `scores` | `labels`, `scores`, `distances`; selects which distill_mlp run to load |
| `distill_run_id` | `null` | evaluate this distill_mlp run; otherwise the latest finished run matching the settings |
| `trial_hash` | `null` | restrict the search to one Optuna trial; otherwise restrict to `campaign` |
| `mlp._target_` | `models.VanillaNetwork` | must match the distilled model |

Use the MLflow UI to inspect runs:

```bash
uv run mlflow ui --backend-store-uri sqlite:///experiments/mlruns.db \
    --default-artifact-root experiments/mlartifacts
```

## Experiments

### Set 1: encoder x method x modality x support_size x mlp_targets

```bash
uv run python scripts/distill_mlp.py -m \
    encoder=MedImageInsight,MedSigLIP,BiomedCLIP,UniMedCLIP,CLIP \
    method=fuse,knnvote_msde,laplacianshot,laplacianshot_msde \
    modality=chest,fundus,mri,oct \
    support_size=5,10,20,30,50 \
    mlp_targets=labels,scores,distances

uv run python scripts/zero_shot.py -m \
    encoder=MedImageInsight,MedSigLIP,BiomedCLIP,UniMedCLIP,CLIP \
    method=fuse,knnvote_msde,laplacianshot,laplacianshot_msde \
    modality=chest,fundus,mri,oct \
    support_size=5,10,20,30,50 \
    mlp_targets=labels,scores,distances
```

### Set 2: hyperparameter optimization of the top methods and encoders

Two stages. Stage 1 tunes each method's pseudolabeler on the few-shot
pseudolabel objective; stage 2 fixes the stage-1 best parameters (so every
trial reuses the same cached pseudolabels) and tunes MSDE for the `scores` and
`distances` targets. `labels` uses no MSDE, so its stage 2 is a single
distillation with the stage-1 parameters. Every trial covers all support sizes
and modalities. Search spaces: `configs/search_space.yaml`.

```bash
# Stage 1: one study per method x encoder (study <method>_<encoder>_method),
# exported to experiments/optuna_best/<method>/<encoder>/method/best_params.yaml.
for method in laplacianshot laplacianshot_msde; do
  for encoder in MedImageInsight MedSigLIP; do
    uv run python scripts/run_optuna.py --stage method --method "$method" --encoder "$encoder"
  done
done

# Stage 2: one study per method x encoder x mlp_targets (study
# <method>_<encoder>_<targets>_msde), exported to
# experiments/optuna_best/<method>/<encoder>/<targets>/best_params.yaml.
for method in laplacianshot laplacianshot_msde; do
  for encoder in MedImageInsight MedSigLIP; do
    for targets in labels scores distances; do
      uv run python scripts/run_optuna.py --stage msde \
          --method "$method" --encoder "$encoder" --mlp-targets "$targets"
    done
  done
done

# Extra arguments are passed to Hydra, e.g. hydra.sweeper.n_trials=20.
# Rerunning a stage adds n_trials more trials to its existing study (the
# wrapper offsets the sampler seed so resumed trials are not repeats).

# Zero-shot evaluation of one trial: take its tags.trial_hash from the MLflow
# run with the highest metrics.trial_objective.
uv run python scripts/zero_shot.py -m \
    method=<method> encoder=<encoder> mlp_targets=<targets> trial_hash=<trial hash> \
    modality=chest,fundus,mri,oct \
    support_size=5,10,20,30,50
```

## Acknowledgements

- Kar, P., Bordoloi, R., Wolkenhauer, O., & Bej, S. (2026). Anomaly Detection via Mean Shift Density Enhancement. arXiv:2602.03293. https://doi.org/10.48550/arXiv.2602.03293

- Kar, P., Lakshmi, G., & Bej, S. (2026). Improved Anomaly Detection in Medical Images via Mean Shift Density Enhancement. arXiv:2604.19191. https://doi.org/10.48550/arXiv.2604.19191

## License

This project is distributed under the MIT License. See the `LICENSE` file for details.
