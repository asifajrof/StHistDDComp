# StHistDDComp

### _Spatial Transcriptomics Histology-based Domain Detection Comparison_

---

## Overview

Currently, the repository supports and integrates the following methods:

1.  [stLearn](https://github.com/BiomedicalMachineLearning/stLearn)
2.  [SpaGCN](https://github.com/jianhuupenn/SpaGCN)
3.  [conST](https://github.com/ys-zong/conST)
4.  [DeepST](https://github.com/JiangBioLab/DeepST)
5.  [ScribbleDom](https://github.com/1alnoman/ScribbleDom)

Each method is wrapped with a uniform interface for running experiments, saving results, and ensuring reproducibility across random seeds.

---

## Repository Structure

```txt
    StHistDDComp/
    │
    ├── README.md
    ├── conST_run/                 # Runner and setup scripts for conST
    │   ├── README.md
    │   ├── run_20.py
    │   └── setup_conST_run.sh
    ├── datasets/                  # Dataset folder (DLPFC, HBC, etc.)
    │   └── ...
    ├── DeepST_run/                # Runner and setup scripts for DeepST
    │   ├── README.md
    │   ├── config.json
    │   ├── run_20.py
    │   └── setup_DeepST_run.sh
    ├── results/                   # Output folder for domain labels and metrics
    ├── run_analysis/              # Analysis scripts for metrics and visualization
    │   ├── config.json
    │   └── wilcox.py
    ├── ScribbleDom_run/           # Runner and setup scripts for ScribbleDom
    │   ├── README.md
    │   ├── config.json
    │   ├── run_20.py
    │   ├── ScribbleDom_run.py
    │   └── setup_ScribbleDom_run.sh
    ├── SpaGCN_run/                # Runner and setup scripts for SpaGCN
    │   ├── README.md
    │   ├── config.json
    │   ├── run_20.py
    │   └── setup_SpaGCN_run.sh
    └── stLearn_run/               # Runner and setup scripts for stLearn
        ├── README.md
        ├── config.json
        ├── run_20.py
        └── setup_stLearn_run.sh
```

---

## Setup Instructions

### 1\. Clone the repository

```bash
git clone https://github.com/asifajrof/StHistDDComp.git
cd StHistDDComp
```

### 2\. Prepare datasets

Download and place your spatial transcriptomics datasets in the `datasets/` directory.  
By default, the pipeline expects:

```txt
    datasets/
    └── DLPFC/
        ├── 151507/
        │   └── filtered_feature_bc_matrix.h5
        ├── 151508/
        └── ...
```

---

## Running the Methods

Each method has its own environment and setup instructions.  
You can run them independently as described below.

---

### **conST**

#### Setup

```bash
cd conST_run
conda create -n conST_run python=3.10
conda activate conST_run
bash setup_conST_run.sh
```

#### MAE model weights download

download the model weights from [here](https://drive.google.com/file/d/1I7uLoL8ay8Scu_sYdjPg1jRaOtcpTNnP/view?usp=drive_link)

#### Configuration

Edit `/conST_run/config.json` to customize.

#### Run

```bash
python conST_run.py <path/to/config.json> <seed>
```

Or run multiple seeds automatically:

```bash
python run_20.py DLPFC
```

**Outputs:**

- `results/<histology_use>/results_seed_<seed>/<sample_name>/labels.csv`  
  Contains columns:

  - `label` – domain assignments
  - `refined_label` – post-refinement domains

---

### **DeepST**

#### Setup

```bash
cd DeepST_run
conda create -n DeepST_run python=3.10
conda activate DeepST_run
bash setup_DeepST_run.sh
```

#### Configuration

Edit `/DeepST_run/config.json` to customize.

#### Run

```bash
python DeepST_run.py <path/to/config.json> <seed>
```

Or run multiple seeds automatically:

```bash
python run_20.py DLPFC
```

**Outputs:**

- `results/w_hist/results_seed_<seed>/<sample_name>/labels.csv`  
  Contains columns:

  - `label` – domain assignments
  - `refined_label` – post-refinement domains

---

### **SpaGCN**

#### Setup

```bash
cd SpaGCN_run
conda create -n SpaGCN_run python=3.10
conda activate SpaGCN_run
bash setup_SpaGCN_run.sh
```

#### Configuration

Edit `/SpaGCN_run/config.json` to customize.

#### Run

```bash
python SpaGCN_run.py <path/to/config.json> <seed>
```

Or run multiple seeds automatically:

```bash
python run_20.py DLPFC
```

**Outputs:**

- `results/<histology_use>/results_seed_<seed>/<sample_name>/labels.csv`  
  Contains columns:

  - `label` – domain assignments
  - `refined_label` – post-refinement domains

---

### **stLearn**

#### Setup

```bash
cd stLearn_run
conda create -n stLearn_run python=3.10
conda activate stLearn_run
bash setup_stLearn_run.sh
```

#### Configuration

Edit `/stLearn_run/config.json` to customize.

#### Run

```bash
python stLearn_run.py <path/to/config.json> <seed>
```

Or run multiple seeds automatically:

```bash
python run_20.py DLPFC
```

**Outputs:**

- `results/w_hist/results_seed_<seed>/<sample_name>/labels.csv`  
  Contains columns:

  - `label` – domain assignments
  - `refined_label` – post-refinement domains

---

### **ScribbleDom**

#### Setup

```bash
cd ScribbleDom_run
git clone https://github.com/1alnoman/ScribbleDom.git
cd ScribbleDom
conda env create -f environment.yml
cd ..
```

#### Run

```bash
conda activate scribble_dom
python run_20.py
```

This executes **ScribbleDom_run.py**, which:

- Updates seed values dynamically
- Runs the model on DLPFC and HBC datasets
- Saves results in `ScribbleDom_run/results/<seed>/`

---

## Reproducibility

Each runner script supports a **`seed`** parameter to ensure reproducibility.

Example:

```bash
python DeepST_run.py config.json 123
```

The framework also provides `run_20.py` scripts to batch-run experiments across 20 predefined seeds for robust comparison.

---
