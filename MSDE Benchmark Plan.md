# MSDE Zero-Shot and Few-Shot Anomaly Detection Benchmark Plan

# 1. Benchmark Goal

Evaluate **MSDE** as a training-free, unsupervised anomaly detection method in three complementary experiments:

1. **Zero-shot anomaly detection (ZSAD):** no target support samples or target labels are provided.
2. **Few-shot anomaly detection (FSAD):** a small normal-only target support set is provided, with no anomaly labels.
3. **Encoder comparison:** evaluate several anomaly detectors across the same set of pretrained encoders to separate the contribution of the representation from the detector.

MSDE is evaluated in two feature settings:

- **MSDE (CLIP):** OpenAI CLIP.
- **MSDE (Best4):** the best-performing encoder among MedImageInsight, UniMed-CLIP, BiomedCLIP, and MedSigLIP.

---

# 2. Terminology

| Term | Meaning in this benchmark |
|---|---|
| **Zero-shot (Z)** | No target support examples or target labels are supplied. |
| **Few-shot (F)** | A small number of target **normal** support examples are supplied. |
| **Unsupervised (U)** | No target anomaly labels are used. |
| **Training-free** | No target-task model parameters are optimized. |
| **Transductive (T)** | The method can use the target/query set collectively. |
| **Inductive (I)** | The method can score a query using a fixed reference/support representation without jointly processing the query set. |

## 2.1 Property overview
| Method | Training | Supervision | Regime | Inference |
|---|---|---|---|---|
| **AnomalyCLIP** | Auxiliary training | Auxiliary anomaly information | Zero-shot | Inductive |
| **AdaCLIP** | Auxiliary training | Auxiliary anomaly information | Zero-shot | Inductive |
| **FAPrompt** | Auxiliary training | Auxiliary training data | Zero-shot | Inductive |
| **AA-CLIP** | Auxiliary training | Auxiliary training data | Zero-shot | Inductive |
| **Bayes-PFL** | Auxiliary training | Auxiliary training data | Zero-shot | Inductive |
| **MSDE** | **None** | **None** | **Zero-shot** | **Transductive** |
| **ANoCo** | None | None | Few-shot — normal support | Inductive |
| **DPMM** | Support-set fitting | None | Few-shot — normal support | Inductive |
| **MediCLIP** | Few-shot adaptation | Synthetic anomalies used for adaptation | Few-shot — normal support | Inductive |
| **UniVAD** | None | None | Few-shot — normal support | Inductive |
| **AnomaMed** | Few-shot adaptation | Synthetic anomalies used for adaptation | Few-shot — normal support | Inductive |
| **MSDE** | **None** | **Normal support only** | **Few-shot — normal support** | **Transductive/Inductive** |

---

# 3. Zero-Shot Benchmark

### Protocol

- No target support images are supplied.
- No target anomaly labels are supplied.
- MSDE (CLIP) uses OpenAI CLIP ViT-L/14@336 image embeddings, the same CLIP weights as the published baselines (which use its patch tokens).
- MSDE (Best4) uses the best of the four medical encoders.
- Classical training-free operations may exploit the unlabeled target set collectively.
- The Best4 encoder must be selected without using final test-set anomaly labels.

## 3.1 Methods

| Method | Feature space | Target support | Target labels | Target training | Role |
|---|---|---|---|---|---|
| AnomalyCLIP | OpenAI CLIP ViT-L/14@336 patch tokens | None | None | None | Published ZSAD |
| AdaCLIP | OpenAI CLIP ViT-L/14@336 patch tokens | None | None | None | Published ZSAD |
| FAPrompt | OpenAI CLIP ViT-L/14@336 patch tokens | None | None | None | Published ZSAD |
| AA-CLIP | OpenAI CLIP ViT-L/14@336 patch tokens (to verify) | None | None | None | Published ZSAD |
| Bayes-PFL | OpenAI CLIP ViT-L/14@336 patch tokens (to verify) | None | None | None | Published ZSAD |
| **MSDE (CLIP)** | **OpenAI CLIP ViT-L/14@336 image embedding** | **None** | **None** | **None** | **Proposed** |
| **MSDE (Best4)** | **Best of MedImageInsight / UniMed-CLIP / BiomedCLIP / MedSigLIP** | **None** | **None** | **None** | **Proposed** |

### Unsupervised vs zero-shot

Zero-shot does not imply unsupervised.

A method can be zero-shot but use supervised or auxiliary training performed before deployment, as long as it requires no target-domain support examples at inference.

- **AnomalyCLIP:** zero-shot, but trained using auxiliary data.
- **AdaCLIP:** zero-shot, but trained using auxiliary data.
- **FAPrompt:** zero-shot, but trained using auxiliary data.
- **AA-CLIP:** zero-shot, but trained using auxiliary data.
- **Bayes-PFL:** zero-shot, but trained using auxiliary data.
- **ANoCo:** few-shot and unsupervised.
- **UniVAD:** few-shot and unsupervised.
- **DPMM:** few-shot and unsupervised.
- **MediCLIP:** few-shot and unsupervised with adaptation/training.
- **AnomaMed:** few-shot and unsupervised with synthetic-anomaly adaptation.
- **MSDE:** can be **zero-shot and unsupervised** or **few-shot and unsupervised**.

---

# 4. Few-Shot Benchmark

### Protocol

The main few-shot benchmark is **normal-only**.

For a given shot count `K`:

> Provide exactly **K normal support images** to each method and evaluate on the held-out test set.

No abnormal support images or anomaly labels are provided.

Recommended shot sweep:

**K ∈ {1, 2, 4, 8, 16}**

The zero-shot setting can be viewed conceptually as **K = 0**, although its protocol differs because the target set may be used transductively.

### Support-set rules

- The support set comes from the designated training/support split.
- The test set remains completely separate.
- All normal-only methods receive the same number `K` of normal support images.
- Methods do not bring their own private support set.
- Support selection should follow a fixed/randomized protocol shared across methods.
- No anomaly labels are used to construct or select the support set.

## 4.2 Methods

| Method | Feature space | Support | Target adaptation/training | Real abnormal support? | Role |
|---|---|---|---|---:|---|
| ANoCo | DINOv3-L/16 patch features (layer 18) | K normal images | None | No | Published training-free FSAD |
| DPMM | DINOv2-S/14 patch features | K normal images | Fit mixture model (published on the full normal training set) | No | Published unsupervised FSAD |
| MediCLIP | CLIP patch features (variant to verify) | K normal images | Published adaptation | No | Published medical FSAD |
| UniVAD | CLIP and DINOv2 patch features with segmentation masks (to verify) | K normal images | None | No | Published training-free FSAD |
| AnomaMed | Multi-layer CLIP patch features (variant to verify) | K normal images | Synthetic anomaly generation + adaptation | No | Published medical FSAD |
| **MSDE (CLIP)** | **OpenAI CLIP ViT-L/14@336 image embedding** | **K normal images** | **None** | **No** | **Proposed training-free FSAD** |
| **MSDE (Best4)** | **Best of MedImageInsight / UniMed-CLIP / BiomedCLIP / MedSigLIP** | **K normal images** | **None** | **No** | **Proposed training-free FSAD** |

### Training-free vs few-shot

Few-shot does not imply supervised training.

A method can be few-shot, unsupervised, and training-free if it directly uses the supplied normal support images without optimizing target-task parameters.

- **AnomalyCLIP:** zero-shot; not training-free due to auxiliary training.
- **AdaCLIP:** zero-shot; not training-free due to auxiliary training.
- **FAPrompt:** zero-shot; not training-free due to auxiliary training.
- **AA-CLIP:** zero-shot; not training-free due to auxiliary training.
- **Bayes-PFL:** zero-shot; not training-free due to auxiliary training.
- **ANoCo:** few-shot; training-free.
- **DPMM:** few-shot; unsupervised but requires fitting the mixture model.
- **MediCLIP:** few-shot; uses adaptation/training.
- **UniVAD:** few-shot; training-free.
- **AnomaMed:** few-shot; generates synthetic anomalies and performs adaptation/training.
- **MSDE:** **zero-shot or few-shot; training-free**.

### Data Efficiency Curve

Evaluate the effect of target-domain support size using \(K \in \{0,1,2,4,8,16\}\) normal samples. \(K=0\) corresponds to MSDE's zero-shot setting, while \(K>0\) corresponds to its few-shot setting. Plot anomaly-detection performance (AUROC/AUPRO) against \(K\), keeping the held-out test set fixed. For each \(K>0\), repeat support sampling across multiple random seeds and report mean ± standard deviation. This directly measures how performance changes with increasing target-domain data availability and highlights MSDE's data efficiency.

---

# 5. Encoder Comparison

## 5.1 Purpose

The encoder experiment asks:

> How much of the performance comes from the pretrained representation, and how much comes from the anomaly detector?

Therefore, the same anomaly detectors are evaluated across the same encoder set.

## 5.2 Encoder × Detector Matrix

**Encoders**

- OpenAI CLIP
- MedImageInsight
- UniMed-CLIP
- BiomedCLIP
- MedSigLIP

**Detectors**

- Cosine softmax with text embedding
- kNN
- LOF
- PCA reconstruction
- MSDE

| Encoder | Cosine softmax + text embedding | kNN | LOF | PCA | MSDE |
|---|---:|---:|---:|---:|---:|
| **OpenAI CLIP** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **MedImageInsight** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **UniMed-CLIP** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **BiomedCLIP** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **MedSigLIP** | ✓ | ✓ | ✓ | ✓ | ✓ |

### Encoder-comparison protocol

- Use the same dataset split for every encoder.
- Use the same anomaly-score protocol for every detector.
- For few-shot evaluation, use the same K-shot support protocol.
- Do not use test labels to select the encoder.
- Report MSDE separately with each encoder rather than only reporting Best4.

## 5.3 Encoder × K-Shot Matrix (MSDE only)

The encoder comparison is repeated for few-shot MSDE, to ask:

> Does the gain from normal support depend on the representation, and does the encoder ranking change with K?

MSDE is run with every encoder at **K = 0, 1, 2, 4, 8, 16** normal support samples (K = 0 is the zero-shot condition, the MSDE column of Section 5.2).

| Encoder | K = 0 | K = 1 | K = 2 | K = 4 | K = 8 | K = 16 |
|---|---:|---:|---:|---:|---:|---:|
| **OpenAI CLIP** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **MedImageInsight** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **UniMed-CLIP** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **BiomedCLIP** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **MedSigLIP** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### Encoder × K-shot protocol

- The support samples are chosen from the labels alone, so every encoder gets the **same support images** for a given K and seed.
- The support samples are removed from the evaluated set, for every encoder.
- For K > 0, repeat over several support-sampling seeds (e.g. 5) and report mean ± standard deviation; the evaluated set and every other setting stay fixed across seeds.
- Use the same MSDE configuration for every encoder and K; the bias function is the default (a `learning_rate` step towards the support mean).
- Best4 is chosen at K = 0 without test labels, and stays fixed for every K.

---

# 6. Recommended MSDE Ablations

## 6.1 Encoder Sensitivity

Report MSDE using:

- OpenAI CLIP
- MedImageInsight
- UniMed-CLIP
- BiomedCLIP
- MedSigLIP

## 6.2 Shot Sensitivity

For the normal-only few-shot setting:

**K = 0, 1, 2, 4, 8, 16**

where K = 0 corresponds to the zero-shot condition.

Plot performance against the number of normal support images.

## 6.3 Transductive vs Inductive MSDE

Where feasible, compare:

- **Transductive MSDE:** uses the target/query set collectively.
- **Inductive MSDE:** uses only the fixed normal support/reference set when constructing the anomaly score.

This isolates the contribution of exploiting the unlabeled target distribution.

## 6.4 Modifications over the Original MSDE

Start from the original MSDE of Kar et al. (arXiv:2602.03293) and add this repo's modifications one at a time, in a fixed order, until the default MSDE of `configs/msde.yaml` is reached. Each row keeps every modification above it, so its gain is measured on top of all earlier ones.

Every component config group (`density`, `score`, `weight`, `metric`) has two options: **`original`**, the paper's function, and **`default`**, the proposed one. `stability` stays at the paper's built-in constant 1 in every row.

| Step | Modification | Overrides relative to the row above |
|---|---|---|
| 0 | **Original MSDE**: raw features, plain kNN reselected every iteration, paper density, weights and score, paper hyperparameters | `density=original score=original weight=original metric=original model.sphere=0 model.recompute_neighbors=1` |
| 1 | + Fixed neighbours (selected once) | `model.recompute_neighbors=0` |
| 2 | + Unit-sphere normalization | `model.sphere=1` |
| 3 | + Fréchet-mean centring | `model.sphere=2` |
| 4 | + Proposed density | `density=default` |
| 5 | + Proposed neighbour score | `score=default` |
| 6 | + Proposed neighbour weights | `weight=default` |
| 7 | + Proposed anomaly metric (= default MSDE) | `metric=default` |

Protocol:

- Zero-shot (transductive) setting, OpenAI CLIP embeddings. Repeat with the Best4 encoder if space allows.
- Same datasets, cap and seed for every row. Report AUROC averaged over the datasets of each modality.
- In a cumulative table, a modification's gain depends on what comes before it. Complement it with a leave-one-out check: remove each modification from the default MSDE separately.

Optional modifications that are not part of the default can be added as extra rows after step 7, e.g. a stability function or pseudo-label rounds (`pseudo_labels.rounds > 1`).

---

# 7. Main Paper Tables

## Table 1 — Zero-Shot Benchmark

| Method | Feature space | Target support | Training-free? | Unsupervised? | Zero-shot? |
|---|---|---|---:|---:|---:|
| AnomalyCLIP | OpenAI CLIP ViT-L/14@336 patch tokens | None | No | No | Yes |
| AdaCLIP | OpenAI CLIP ViT-L/14@336 patch tokens | None | No | No | Yes |
| FAPrompt | OpenAI CLIP ViT-L/14@336 patch tokens | None | No | No | Yes |
| AA-CLIP | OpenAI CLIP ViT-L/14@336 patch tokens (to verify) | None | No | No | Yes |
| Bayes-PFL | OpenAI CLIP ViT-L/14@336 patch tokens (to verify) | None | No | No | Yes |
| **MSDE (CLIP)** | **OpenAI CLIP ViT-L/14@336 image embedding** | **None** | **Yes** | **Yes** | **Yes** |
| **MSDE (Best4)** | **Best4 medical encoder** | **None** | **Yes** | **Yes** | **Yes** |

## Table 2 — Few-Shot Benchmark

| Method | Feature space | Normal support | Abnormal support | Training-free? | Unsupervised? |
|---|---|---:|---:|---:|---:|
| ANoCo | DINOv3-L/16 | K | 0 | Yes | Yes |
| DPMM | DINOv2-S/14 | K | 0 | No | Yes |
| MediCLIP | CLIP (variant to verify) | K | 0 | No | No* |
| UniVAD | CLIP + DINOv2 (to verify) | K | 0 | Yes | Yes |
| AnomaMed | CLIP, multi-layer (variant to verify) | K | 0 | No | No* |
| **MSDE (CLIP)** | **OpenAI CLIP ViT-L/14@336** | **K** | **0** | **Yes** | **Yes** |
| **MSDE (Best4)** | **Best4 medical encoder** | **K** | **0** | **Yes** | **Yes** |

## Table 3 — Encoder Comparison

| Encoder | Cosine softmax + text embedding | kNN | LOF | PCA | MSDE |
|---|---:|---:|---:|---:|---:|
| OpenAI CLIP | ✓ | ✓ | ✓ | ✓ | ✓ |
| MedImageInsight | ✓ | ✓ | ✓ | ✓ | ✓ |
| UniMed-CLIP | ✓ | ✓ | ✓ | ✓ | ✓ |
| BiomedCLIP | ✓ | ✓ | ✓ | ✓ | ✓ |
| MedSigLIP | ✓ | ✓ | ✓ | ✓ | ✓ |

## Table 4 — Modifications over the Original MSDE

AUROC (%), zero-shot, OpenAI CLIP. Δ prev = gain over the row above; Δ orig = gain over the original MSDE (step 0). Steps as in Section 6.4.

| Step | Configuration | Chest | Fundus | MRI | OCT | Mean | Δ prev | Δ orig |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | Original MSDE | | | | | | — | — |
| 1 | + Fixed neighbours | | | | | | | |
| 2 | + Unit sphere | | | | | | | |
| 3 | + Fréchet-mean centring | | | | | | | |
| 4 | + Proposed density | | | | | | | |
| 5 | + Proposed neighbour score | | | | | | | |
| 6 | + Proposed neighbour weights | | | | | | | |
| 7 | + Proposed anomaly metric (**default MSDE**) | | | | | | | |

## Table 5 — Encoder × K-Shot (MSDE)

AUROC (%), mean over modalities; for K > 0, mean ± standard deviation over support-sampling seeds. Δ = gain of K = 16 over K = 0. Settings as in Section 5.3.

| Encoder | K = 0 | K = 1 | K = 2 | K = 4 | K = 8 | K = 16 | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| OpenAI CLIP | | | | | | | |
| MedImageInsight | | | | | | | |
| UniMed-CLIP | | | | | | | |
| BiomedCLIP | | | | | | | |
| MedSigLIP | | | | | | | |

Per-modality versions of this table go in the appendix.

## Figure 1 — Data Efficiency Curve

Performance against the number of normal support images, as described in Section 4 (Data Efficiency Curve).

- **x-axis:** K ∈ {0, 1, 2, 4, 8, 16} normal support images (log-spaced ticks). K = 0 is MSDE's zero-shot setting.
- **y-axis:** AUROC (and AUPRO where pixel labels exist), on the same fixed held-out test set for every K.
- **Panels:** one per modality (Chest, Fundus, MRI, OCT), plus an optional mean panel.
- **Lines:** MSDE (CLIP), MSDE (Best4), and the few-shot baselines (ANoCo, DPMM, MediCLIP, UniVAD, AnomaMed) at K ≥ 1.
- **Reference lines:** the best zero-shot baseline as a horizontal dashed line, since zero-shot methods do not use K.
- **Uncertainty:** for K > 0, mean ± standard deviation over several support-sampling seeds (e.g. 5), shown as shaded bands. All methods use the same support images for a given K and seed.

The numbers behind the figure are also reported as a table:

| Method | K = 0 | K = 1 | K = 2 | K = 4 | K = 8 | K = 16 |
|---|---:|---:|---:|---:|---:|---:|
| MSDE (CLIP) | | | | | | |
| MSDE (Best4) | | | | | | |
| ANoCo | — | | | | | |
| DPMM | — | | | | | |
| MediCLIP | — | | | | | |
| UniVAD | — | | | | | |
| AnomaMed | — | | | | | |

---

# 8. Fairness and Interpretation

## 8.1 Method-Level Comparison

**MSDE (CLIP)** vs the other detectors using OpenAI CLIP.

This asks:

> Does MSDE provide an advantage over standard text-guided, geometric, density, and subspace anomaly detectors when the representation is held fixed?

## 8.2 Representation-Level Comparison

**MSDE** across the five candidate encoders.

This asks:

> How much does the choice of pretrained representation affect MSDE?

The encoder × K-shot matrix (Table 5) asks the same question in the few-shot setting, and whether normal support narrows or widens the gaps between encoders.

## 8.3 Best4 Comparison

**MSDE (Best4)** demonstrates the performance achievable when MSDE is paired with the strongest candidate medical representation.

The encoder-selection procedure must not use final test labels.

---

# 9. Final Experimental Story

The benchmark should tell one coherent story:

**Zero-shot**  
→ MSDE operates without target support or labels.

**Few-shot**  
→ MSDE incorporates a small normal-only support set without training.

**Encoder comparison**  
→ MSDE is evaluated across multiple pretrained representations against common detectors.

**Data efficiency curve**  
→ shows how MSDE improves as normal support images are added, from K = 0 to K = 16.

**Modifications over the original MSDE**  
→ attributes the gain over the published MSDE to each of this repo's modifications.

**Transductive vs inductive ablation**  
→ isolates the value of exploiting the unlabeled target distribution.

The central methodological positioning is:

> **MSDE is a training-free, unsupervised anomaly detector that can operate in a zero-shot transductive regime and can naturally extend to normal-only few-shot detection.**
