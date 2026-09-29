# DELTA

## Setup

```bash
git clone https://github.com/nveshaan/delta.git
cd delta
uv sync
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

## Few-shot training

Run a default experiment with:

```bash
uv run python scripts/few_shot.py
```

Other examples:

```bash
# LaplacianShot pseudolabeling.
uv run python scripts/few_shot.py \
    modality=mri \
    encoder=CLIP \
    method=laplacianshot \
    support_size=10

# KNN-vote + MSDE pseudolabeling.
uv run python scripts/few_shot.py \
    modality=fundus \
    method=knnvote_msde

# Distill directly from pseudolabels without MSDE/GDE.
uv run python scripts/few_shot.py \
    modality=chest \
    method=fuse \
    apply_msde_gde=false
```

The base experiment configuration is [configs/few_shot.yaml](configs/few_shot.yaml).
It contains the MSDE/GDE settings, MLP training settings, MLflow settings, and
the MLP Hydra target. Dataset configurations are selected from
`configs/modality/<modality>.yaml`; pseudolabeler targets are selected
from `configs/method/`.

Method-specific ablations stay in the method configs. For example, FUSE modes
are selected by editing `configs/method/fuse.yaml`; they are not exposed as
few-shot command-line flags.

MLflow uses SQLite by default under `experiments/` and
creates nested runs for:

```text
type → hyperparams → modality → dataset
```

The run artifacts include resolved configs, label mappings, predictions, metrics,
MLP checkpoints, and loss curves. Use the MLflow UI to inspect runs:

```bash
uv run mlflow ui --backend-store-uri sqlite:///experiments/mlruns.db \
    --default-artifact-root experiments/mlartifacts
```

Use `uv run python scripts/few_shot.py --help` for all command-line options.

## Zero-shot evaluation

Evaluate the latest matching distilled few-shot MLP on the test split:

```bash
uv run python scripts/zero_shot.py \
    method=fuse \
    modality=chest \
    encoder=CLIP \
    support_size=5
```

The script downloads `model.pt` from the matching `few_shot` MLflow run,
evaluates it on `split=test`, and logs per-dataset metrics in a new
`zero_shot` MLflow run.

## Acknowledgements

- Kar, P., Bordoloi, R., Wolkenhauer, O., & Bej, S. (2026). Anomaly Detection via Mean Shift Density Enhancement. arXiv:2602.03293. https://doi.org/10.48550/arXiv.2602.03293

- Kar, P., Lakshmi, G., & Bej, S. (2026). Improved Anomaly Detection in Medical Images via Mean Shift Density Enhancement. arXiv:2604.19191. https://doi.org/10.48550/arXiv.2604.19191

## License

This project is distributed under the MIT License. See the `LICENSE` file for details.
