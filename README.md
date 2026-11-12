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

Use `uv run python scripts/generate_embeddings.py --help` for all command-line options.

The generated files are written to
`data/<modality>/<dataset_name>/<encoder>_embeds.npy` and
`data/<modality>/<dataset_name>/labels.npy`.

> **MedSigLIP** is a gated model. Before running this, request access on its model page on the HF Hub, then authenticate locally with:
> `hf auth login`
> (or set the `HF_TOKEN` environment variable).

## Run MSDE

`scripts/run_msde.py` runs Mean Shift Density Enhancement on the embeddings of an encoder, using the settings in `configs/msde.yaml`. The movement of each point is its anomaly score (label 0 is normal, any other label is anomalous).

```bash
# One dataset.
uv run python scripts/run_msde.py modality=chest encoder=CLIP dataset=VinCXR

# Every dataset listed in configs/modality/<modality>.yaml (one MLflow run each).
uv run python scripts/run_msde.py modality=chest encoder=CLIP dataset=null

# Three runs; runs after the first use pseudo labels from the previous one with label_mode=zero_label.
uv run python scripts/run_msde.py pseudo_labels.rounds=3 pseudo_labels.anomaly_fraction=0.2

# Override MSDE hyperparameters.
uv run python scripts/run_msde.py model.k=20 model.learning_rate=0.1
```

The density, score, stability and weight used are selected by the config groups in `configs/{density,score,stability,weight}/`.

Everything is logged with MLflow to `experiments/mlflow.db` (artifacts in `experiments/artifacts`), view it with `uv run mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db`:

- **Params:** dataset, modality, encoder, density/score/stability/weight names, seed, distance, sample counts and the model hyperparameters.
- **Metrics:** AUROC, AUPRC and P@n (n = number of anomalies), each also computed on the negated score (`*_flipped`), since the direction of the movement score depends on the encoder. For datasets with several anomaly labels, the same metrics are logged per subtype as `subtype_<label>/<metric>` (each subtype against the normal samples). With several rounds, metrics are logged per round (`step`).
- **Artifacts:** the resolved config and the density, score, stability and weight configs under `configs/`.

## Acknowledgements

- Kar, P., Bordoloi, R., Wolkenhauer, O., & Bej, S. (2026). Anomaly Detection via Mean Shift Density Enhancement. arXiv:2602.03293. https://doi.org/10.48550/arXiv.2602.03293

- Kar, P., Lakshmi, G., & Bej, S. (2026). Improved Anomaly Detection in Medical Images via Mean Shift Density Enhancement. arXiv:2604.19191. https://doi.org/10.48550/arXiv.2604.19191

## License

This project is distributed under the MIT License. See the `LICENSE` file for details.
