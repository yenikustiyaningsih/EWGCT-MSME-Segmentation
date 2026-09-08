# EWGCT-MSME-Segmentation

[![Python](https://img.shields.io/badge/Python-3.10.7-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![MethodsX](https://img.shields.io/badge/Journal-MethodsX-005A9C.svg)](https://www.sciencedirect.com/journal/methodsx)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ewgct-msme-segmentation.streamlit.app)

A reproducible implementation of the **Entropy-Weighted Gower–CLARA–TOPSIS (EWGCT)** procedure for mixed-type Micro, Small, and Medium Enterprise (MSME) segmentation and within-cluster prioritization.

This repository accompanies the MethodsX manuscript:

> **A Reproducible Entropy-Weighted Gower-CLARA-TOPSIS Procedure for Mixed-Type MSME Segmentation and Prioritization**

---

## Live Application

The interactive Streamlit application is publicly available at:

[Launch the EWGCT Streamlit Application](https://ewgct-msme-segmentation.streamlit.app)

---

## Overview

The EWGCT procedure integrates:

* **Entropy Weight Method (EWM)** for objective feature weighting;
* **Gower dissimilarity** for mixed numerical, count, binary, and categorical data;
* **CLARA with PAM/K-Medoids** for scalable medoid-based clustering;
* **Top-N feature selection** for evaluating reduced feature configurations;
* **Internal cluster validation** using the Silhouette Coefficient, medoid-based Davies–Bouldin Index, and Calinski–Harabasz Index; and
* **Within-cluster TOPSIS** for ranking MSMEs among comparable peer groups.

The workflow is demonstrated using anonymized or de-identified MSME records from Sampang, Indonesia.

---

## Method Workflow

```text
1. Input Mixed-Type MSME Data
   Numerical, count, binary, and nominal variables
    ↓
2. Data Cleaning and Documentation
   Remove duplicates, inspect missing values, handle outliers,
   and document preprocessing decisions
    ↓
3. Dual Variable Representation
   Preserve raw categorical values for Gower dissimilarity
   and prepare binary, count, or numerical representations
   for entropy weighting
    ↓
4. Entropy Weight Calculation
   Normalize entropy-eligible variables and calculate entropy,
   divergence, and objective feature weights
    ↓
5. Weighted Gower Dissimilarity Construction
   Combine numerical, binary, and categorical feature-level
   dissimilarities using normalized entropy weights
    ↓
6. CLARA Clustering Scenarios
   Run baseline CLARA, entropy-weighted all-feature CLARA,
   and Top-N feature-subset CLARA
    ↓
7. Candidate k Evaluation
   Evaluate candidate cluster numbers using repeated CLARA
   sampling and PAM medoids
    ↓
8. Cluster Validation and Model Selection
   Evaluate each configuration using the Silhouette Coefficient,
   medoid-based Davies–Bouldin Index, Calinski–Harabasz Index,
   stability, and interpretability
    ↓
9. Cluster Profiling
   Describe the selected clusters using cluster sizes, medoids,
   and average economic and operational characteristics
    ↓
10. Within-Cluster TOPSIS Ranking
    Apply TOPSIS separately within each selected cluster using
    benefit- and cost-consistent criteria
    ↓
11. Final Output
    Report entropy weights, selected features, medoids,
    cluster labels, validation metrics, cluster profiles,
    and within-cluster ranking tables
```

---

## Repository Structure

```text
EWGCT-MSME-Segmentation/
│
├── Data/
│   └── dataset_umkm_sampang_25 new.xlsx
│
├── Docs/
│   ├── README_TOPSIS.md
│   ├── SAMPLE_OUTPUT_TOPSIS.md
│   └── TOPSIS_IMPLEMENTATION_GUIDE.md
│
├── Notebook/
│   └── entropy_clara_topsis.ipynb
│
├── Outputs/
│   ├── Clustering and TOPSIS results
│   └── Evaluation and visualization outputs
│
├── Streamlit_app/
│   └── new_streamlit.py
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## Dataset

The demonstration dataset contains mixed-type MSME attributes, including:

* formal business registration;
* workforce;
* annual production capacity;
* annual revenue;
* assets;
* social-media use;
* marketplace use; and
* land ownership.

Direct identifiers have been removed or replaced with anonymous codes. The repository does not contain the original non-anonymized administrative records.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yenikustiyahningsih/EWGCT-MSME-Segmentation.git
cd EWGCT-MSME-Segmentation
```

Alternatively, select **Code → Download ZIP** on the repository page and extract the downloaded file.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Activate the environment on Linux or macOS:

```bash
source .venv/bin/activate
```

### 3. Install the Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The implementation was tested using **Python 3.10.7**. Exact package versions are provided in `requirements.txt` to support reproducible execution.

---

## Running the Analysis Notebook

Open the following notebook using Jupyter Notebook, JupyterLab, or Visual Studio Code:

```text
Notebook/entropy_clara_topsis.ipynb
```

Run all cells sequentially from the beginning to reproduce:

* preprocessing results;
* entropy feature weights;
* baseline and weighted Gower matrices;
* CLARA clustering results;
* cluster validation metrics;
* Top-N feature evaluation;
* cluster profiles; and
* within-cluster TOPSIS rankings.

---

## Running the Streamlit Application

The deployed application can be accessed directly through Streamlit Community Cloud:

[Open the EWGCT Streamlit Application](https://ewgct-msme-segmentation.streamlit.app)

Alternatively, run the application locally from the repository root:

```bash
streamlit run Streamlit_app/new_streamlit.py
```

The application provides interactive access to:

* dataset upload and inspection;
* preprocessing results;
* CLARA baseline evaluation;
* entropy-weight calculations;
* entropy-weighted CLARA results;
* Top-N feature-selection results;
* Silhouette, Davies–Bouldin, and Calinski–Harabasz metrics;
* cluster profiles and visualizations;
* prediction of the cluster for a new MSME;
* within-cluster TOPSIS results; and
* downloadable clustering and ranking outputs.

For deployment on Streamlit Community Cloud, select **Python 3.10** in the advanced deployment settings to ensure compatibility with the dependency versions listed in `requirements.txt`.

---

## Main Demonstration Results

The selected configuration retained four operational and financial features:

1. annual production capacity;
2. assets;
3. annual revenue; and
4. workforce.

The selected Top-4 configuration at `k = 4` produced:

| Validation metric | Result |
| --- | ---: |
| Silhouette Coefficient | 0.5588 |
| Medoid-based Davies–Bouldin Index | 0.6181 |
| Calinski–Harabasz Index | 992.3958 |

These values should be reproduced using the supplied dataset, implementation, parameters, and dependency versions.

---

## TOPSIS Interpretation

TOPSIS is applied **separately within each selected cluster**.

A higher TOPSIS score indicates greater relative economic-operational readiness among MSMEs belonging to the same cluster. Because each cluster has its own normalization and ideal solutions, TOPSIS scores must not be compared directly across different clusters.

The ranking represents relative readiness under the declared benefit criteria. It should not be interpreted as a causal measure of MSME growth or as a direct measure of support need.

---

## Reproducibility Notes

For reproducible execution:

* use Python 3.10.7;
* install the exact dependency versions listed in `requirements.txt`;
* retain the feature order defined in the notebook;
* use the declared random seed and CLARA sampling parameters;
* calculate the Silhouette Coefficient and medoid-based DBI from the appropriate Gower dissimilarity matrix;
* calculate the Calinski–Harabasz Index from the corresponding feature matrix, not from the distance matrix; and
* perform TOPSIS independently within each cluster.

Entropy weights represent the observed information dispersion of the dataset. They do not represent causal influence, universal feature importance, or policy priorities.

---

## Documentation

Additional TOPSIS documentation is available in:

* [`Docs/README_TOPSIS.md`](Docs/README_TOPSIS.md)
* [`Docs/SAMPLE_OUTPUT_TOPSIS.md`](Docs/SAMPLE_OUTPUT_TOPSIS.md)
* [`Docs/TOPSIS_IMPLEMENTATION_GUIDE.md`](Docs/TOPSIS_IMPLEMENTATION_GUIDE.md)

---

## Authors and Contributors

* Yeni Kustiyahningsih
* Eza Rahmanita
* Imamah
* Aeri Rachmad
* Herdiyanti Fifin Purwaningrum
* Firdausi Putri Cahyani
* Miswanto
* Yuli Panca Asmoro
* Darnah
* Andrea Tri Rian Dani
* Bain Khusnul Khotimah
* Jaka Purnama

---

## Citation

If you use this procedure, code, or demonstration materials, please cite the associated MethodsX article. Complete publication details and DOI information will be added after publication.
