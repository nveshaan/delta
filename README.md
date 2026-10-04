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

### Few-Shot Pseudo Labels

```bash
uv run python scripts/few_shot.py modality=chest method=laplacianshot_msde support_size=5
```

### Distillation

```bash
# Missing pseudolabels are generated (as a few_shot run) automatically.
uv run python scripts/distill_mlp.py modality=chest method=laplacianshot_msde support_size=5 mlp_targets=labels
uv run python scripts/distill_mlp.py modality=chest method=laplacianshot_msde support_size=5 mlp_targets=scores
uv run python scripts/distill_mlp.py modality=chest method=laplacianshot_msde support_size=5 mlp_targets=distances
```

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

```bash
# One Optuna study per method x encoder x mlp_targets; every trial covers all
# support sizes and modalities. Search spaces: configs/search_space.yaml.
for method in <method 1> <method 2>; do
  for encoder in <encoder 1> <encoder 2>; do
    for targets in labels scores distances; do
      uv run python scripts/distill_mlp.py -m +experiment=optuna \
          method=$method encoder=$encoder mlp_targets=$targets
    done
  done
done

# Zero-shot evaluation of one trial: take its tags.trial_hash from the MLflow
# run with the highest metrics.trial_objective.
uv run python scripts/zero_shot.py -m \
    method=<method> encoder=<encoder> mlp_targets=<targets> trial_hash=<trial hash> \
    modality=chest,fundus,mri,oct \
    support_size=5,10,20,30,50
```

## Plots

```bash
# Experiment set 1 (default --campaign adhoc).
uv run python plots/zero_shot_encoder_method_consistency.py
uv run python plots/zero_shot_mlp_targets_delta.py

# Experiment set 2.
uv run python plots/zero_shot_encoder_method_consistency.py --campaign optuna
uv run python plots/zero_shot_mlp_targets_delta.py --campaign optuna --encoders <encoder 1> <encoder 2> --methods <method 1> <method 2>
```

## Acknowledgements

- Kar, P., Bordoloi, R., Wolkenhauer, O., & Bej, S. (2026). Anomaly Detection via Mean Shift Density Enhancement. arXiv:2602.03293. https://doi.org/10.48550/arXiv.2602.03293

- Kar, P., Lakshmi, G., & Bej, S. (2026). Improved Anomaly Detection in Medical Images via Mean Shift Density Enhancement. arXiv:2604.19191. https://doi.org/10.48550/arXiv.2604.19191

## License

This project is distributed under the MIT License. See the `LICENSE` file for details.
