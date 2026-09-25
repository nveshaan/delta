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
│   └── OASIS Alzheimer,s Detection
└── oct
    ├── ARMD OCT
    ├── OCT2017
    ├── Retinal OCT images
    ├── RetinalOCT_Dataset
    └── ZhangLabData OCT
```

Then, to generate embeddings of **MedImageInsight**, **MedSigLIP**, **BiomedCLIP**, **UniMedCLIP** and **CLIP**, run the following command:

```bash
uv run python scripts/generate_embeddings.py
```

> **MedSigLIP** is a gated model. Before running this, request access on its model page on the HF Hub, then authenticate locally with:
> `hf auth login`
> (or set the `HF_TOKEN` environment variable).

## Acknowledgements
- Kar, P., Bordoloi, R., Wolkenhauer, O., & Bej, S. (2026). Anomaly Detection via Mean Shift Density Enhancement. arXiv:2602.03293. https://doi.org/10.48550/arXiv.2602.03293

- Kar, P., Lakshmi, G., & Bej, S. (2026). Improved Anomaly Detection in Medical Images via Mean Shift Density Enhancement. arXiv:2604.19191. https://doi.org/10.48550/arXiv.2604.19191

## License
This project is distributed under the MIT License. See the `LICENSE` file for details.
