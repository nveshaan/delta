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
- MSDE (CLIP) uses OpenAI CLIP embeddings.
- MSDE (Best4) uses the best of the four medical encoders.
- Classical training-free operations may exploit the unlabeled target set collectively.
- The Best4 encoder must be selected without using final test-set anomaly labels.

## 3.1 Methods

| Method | Feature space | Target support | Target labels | Target training | Role |
|---|---|---|---|---|---|
| AnomalyCLIP | CLIP | None | None | None | Published ZSAD |
| AdaCLIP | CLIP | None | None | None | Published ZSAD |
| FAPrompt | CLIP | None | None | None | Published ZSAD |
| AA-CLIP | CLIP | None | None | None | Published ZSAD |
| Bayes-PFL | CLIP | None | None | None | Published ZSAD |
| **MSDE (CLIP)** | **OpenAI CLIP** | **None** | **None** | **None** | **Proposed** |
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
| ANoCo | OpenAI CLIP / published CLIP setup | K normal images | None | No | Published training-free FSAD |
| DPMM | OpenAI CLIP | K normal images | Fit mixture model | No | Published unsupervised FSAD |
| MediCLIP | OpenAI CLIP / published CLIP setup | K normal images | Published adaptation | No | Published medical FSAD |
| UniVAD | Published visual foundation features | K normal images | None | No | Published training-free FSAD |
| AnomaMed | Published medical/CLIP setup | K normal images | Synthetic anomaly generation + adaptation | No | Published medical FSAD |
| **MSDE (CLIP)** | **OpenAI CLIP** | **K normal images** | **None** | **No** | **Proposed training-free FSAD** |
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

| Step | Modification | Overrides relative to the row above |
|---|---|---|
| 0 | **Original MSDE**: raw features, plain kNN reselected every iteration, paper hyperparameters | `score=original dist=euclidean model.sphere=0 model.recompute_neighbors=1 model.k=100 model.nbd_sample_count_threshold=70 model.learning_rate=0.1 model.max_iters_shift=6 model.shift_threshold=0.0003` |
| 1 | + Unit-sphere normalization | `model.sphere=1` |
| 2 | + Fréchet-mean centring | `model.sphere=2` |
| 3 | + Score-based neighbour selection (density × stability × distance kernel) | `score=default` |
| 4 | + Fixed neighbours (selected once) | `model.recompute_neighbors=0` |
| 5 | + Run to convergence | `model.max_iters_shift=1000 model.shift_threshold=1e-4` |
| 6 | + Repo hyperparameters (= default MSDE) | `model.k=50 model.nbd_sample_count_threshold=30 model.learning_rate=0.3` |

Protocol:

- Zero-shot (transductive) setting, OpenAI CLIP embeddings. Repeat with the Best4 encoder if space allows.
- Same datasets, cap and seed for every row. Report AUROC averaged over the datasets of each modality.
- In a cumulative table, a modification's gain depends on what comes before it. Complement it with a leave-one-out check: remove each modification from the default MSDE separately.
- The repo hyperparameters (step 6) must not be chosen using test labels.

Optional modifications that are not part of the default can be added as extra rows after step 6, e.g. a stability function or pseudo-label rounds (`pseudo_labels.rounds > 1`).

---

# 7. Main Paper Tables

## Table 1 — Zero-Shot Benchmark

| Method | Feature space | Target support | Training-free? | Unsupervised? | Zero-shot? |
|---|---|---|---:|---:|---:|
| AnomalyCLIP | CLIP | None | No | No | Yes |
| AdaCLIP | CLIP | None | No | No | Yes |
| FAPrompt | CLIP | None | No | No | Yes |
| AA-CLIP | CLIP | None | No | No | Yes |
| Bayes-PFL | CLIP | None | No | No | Yes |
| **MSDE (CLIP)** | **OpenAI CLIP** | **None** | **Yes** | **Yes** | **Yes** |
| **MSDE (Best4)** | **Best4 medical encoder** | **None** | **Yes** | **Yes** | **Yes** |

## Table 2 — Few-Shot Benchmark

| Method | Feature space | Normal support | Abnormal support | Training-free? | Unsupervised? |
|---|---|---:|---:|---:|---:|
| ANoCo | CLIP | K | 0 | Yes | Yes |
| DPMM | CLIP | K | 0 | No | Yes |
| MediCLIP | CLIP | K | 0 | No | No* |
| UniVAD | Published feature space | K | 0 | Yes | Yes |
| AnomaMed | Published feature space | K | 0 | No | No* |
| **MSDE (CLIP)** | **OpenAI CLIP** | **K** | **0** | **Yes** | **Yes** |
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
| 1 | + Unit sphere | | | | | | | |
| 2 | + Fréchet-mean centring | | | | | | | |
| 3 | + Score-based neighbour selection | | | | | | | |
| 4 | + Fixed neighbours | | | | | | | |
| 5 | + Run to convergence | | | | | | | |
| 6 | + Repo hyperparameters (**default MSDE**) | | | | | | | |

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
