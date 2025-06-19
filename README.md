# MDS Capstone Project: Afforestation Monitoring

![Tests](https://github.com/ptompalski/MDSAfforestationMonitoring/actions/workflows/run_test.yaml/badge.svg)
[![License](https://img.shields.io/badge/License-GPL--3-blue)](./LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.12.11-blue)](https://www.python.org/downloads/release/python-31211/)
[![PyTorch Version](https://img.shields.io/badge/PyTorch-2.7.0-red)](https://pytorch.org/blog/pytorch-2-7/)
[![Report](https://img.shields.io/badge/Report-Technical-orange)](./reports/technical/report.pdf)

## Summary

### Project Overview

Monitoring afforestation across hundreds of remote and ecologically diverse sites in Canada presents a significant challenge, particularly due to the weak spectral signals produced by sparse canopies during the early stages of tree growth. This project investigates the feasibility of leveraging remote sensing and machine learning to support large-scale forest restoration efforts.

We focus on two central research questions:

1. Can satellite-derived vegetation indices and site-level data accurately predict tree survival over time in large-scale afforestation programs?
2. What modeling approaches are most effective for this predictive task?

Using data from Canada’s **2 Billion Trees** initiative, we trained a suite of classical machine learning models, including:

- [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)  
  via the [Scikit-learn](https://scikit-learn.org/stable/index.html) library
- [Gradient Boosting Machine (GBM)](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn)  
  using the [XGBoost](https://xgboost.readthedocs.io/en/latest/index.html) library

To better model the sequential and seasonal dynamics of vegetation indices, we further developed deep learning models using [PyTorch](https://pytorch.org/), specifically:

- [Long Short-Term Memory (LSTM)](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) networks
- [Gated Recurrent Unit (GRU)](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html) networks

### Deliverables

This project provides the following key deliverables:

1. **Reproducible model pipeline**:  
   A complete, tested, and modular pipeline for data preprocessing, model training, and performance evaluation, implemented in Python.  
   Users can run the entire workflow via the provided [Makefile](./Makefile).  
   See the [`Quick Start Guide`](./notebooks/data_product_quickstart.ipynb) for setup and usage instructions.

2. **Technical report**:  
   A comprehensive [technical report](./reports/technical/report.pdf) detailing the analysis, modeling approaches, results, and recommendations for future development.

### Repository Structure

The deliverables are organized into the following directories:

- [`data/`](./data):  
  Placeholder directories for the raw, interim (partially processed), and fully processed datasets:  
  - [`raw/`](./data/raw): Unmodified source data  
  - [`interim/`](./data/interim): Data after partial preprocessing  
  - [`processed/`](./data/processed): Final cleaned dataset used for modeling  
  
  **Note:** Due to privacy restrictions, data is not included in the repository. To request access, please contact [Piotr Tompalski](https://github.com/ptompalski).

- [`src/`](./src):  
  Contains all core Python scripts used to run the full modeling pipeline. Subdirectories are organized by pipeline stage:
  - [`data/`](./src/data):  
    Scripts for data preprocessing, cleaning, and preparation prior to modeling.
  - [`models/`](./src/models):  
    Scripts for constructing untrained model instances (classical and deep learning).
  - [`training/`](./src/training):  
    Handles model training and hyperparameter tuning routines.
  - [`slurm_jobs/`](./src/slurm_jobs):  
    Job submission scripts for training deep learning models on the [UBC Sockeye Computing Platform](https://arc.ubc.ca/compute-storage/ubc-arc-sockeye), using the [Slurm](https://slurm.schedmd.com/overview.html) cluster manager.  
    These are not required for local execution, but are included to support reproducibility and extension on high-performance computing systems.
  - [`evaluation/`](./src/evaluation):  
    Scripts for evaluating trained models using various classification metrics.

- [`models/`](./models):  
  Contains serialized model objects (`.joblib` for classical ML, `.pth` for deep learning).  
  Models are grouped by the binary classification threshold used during training:  
  - [`50/`](./models/50), [`60/`](./models/60), [`70/`](./models/70), [`80/`](./models/80)  
    For example, a tuned Gradient Boosting model trained with a 70% survival threshold is saved at:  
    `models/70/tuned_gradient_boosting.joblib`  

  Each threshold folder also includes a `logs/` subdirectory with CSV files summarizing hyperparameter search results.
  
- [`results/`](./results):  
  Stores model evaluation outputs, including `.csv` and `.joblib` files with key error metrics—such as confusion matrices, precision-recall (PR) and ROC curves, and $F_1$ scores.  
  Results are organized by classification threshold, consistent with the structure of the [`models/`](./models) directory.

- [`reports/`](./reports):  
  Contains all documents and source files related to the three reports generated during the project:
  - [`proposal/`](./reports/proposal):  
    Includes the [proposal report](./reports/proposal/report.pdf), outlining the project's initial objectives, proposed methodology, and expected deliverables.
  - [`technical/`](./reports/technical):  
    Contains the [technical report](./reports/technical/report.pdf), which provides a detailed account of the full analysis—including methodology, modeling results, evaluation, and future directions.
  - [`final/`](./reports/final):  
    Includes the [MDS final report](./reports/final/report.pdf), a concise summary of the technical report, submitted for internal program assessment.

- [`img/`](./img/):  
  Contains plots, diagrams, and other visual assets used in the reports and documentation.

- [`tests/`](./tests/):  
  A comprehensive test suite for validating all core pipeline scripts, implemented using the [pytest](https://docs.pytest.org/en/stable/) framework.

## Prerequisites

### Quarto

[Quarto](https://quarto.org/) is an open-source publishing system that supports reproducible documents using Markdown and executable code. While rendered reports are already included in the repository, **Quarto is required to re-render them locally** to validate reproducibility. Installation instructions are available in the [Get Started Guide](https://quarto.org/docs/get-started/).

### Conda

[Conda](https://docs.conda.io/projects/conda/en/latest/index.html) is an environment and package manager used to handle software dependencies in isolated environments. This project uses Conda to ensure reproducibility and prevent library conflicts. Three environments are provided at the root of the repository:

- [`environment.yml`](./environment.yml): Core environment for executing the pipeline.
- [`environment-dev.yml`](./environment-dev.yml): Development environment for testing and report generation; includes additional packages beyond the core environment.
- [`environment-dev-gpu.yml`](./environment-dev-gpu.yml): GPU-compatible environment for use on the UBC Sockeye platform. Included for transparency; not intended for general use.

To install Conda, refer to the [User Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html). If you prefer a minimal installation, you can install [Miniconda3](https://docs.conda.io/en/latest/miniconda.html), which provides the same functionality with a smaller footprint.

For instructions on creating and activating environments, see the [`Quick Start Guide`](./notebooks/data_product_quickstart.ipynb).

## Rendering the Reports

To generate any of the project reports as PDFs, run the following command in your terminal:

```bash
quarto render reports/<report_name>/report.qmd --to pdf
```

Replace `<report_name>` with one of the following:

- `proposal`: The proposal report
- `technical`: The technical report
- `final`: The final MDS report

For example, to render the technical report:

```bash
quarto render reports/technical/report.qmd --to pdf
```

**Note:** Both Quarto and the development environment provided by `environment-dev.yml` **must** be installed prior to rendering the report.

## Running the Pipeline - Quick Start

To quickly get started with the project, you can refer to the [`Data Product Quick Start`](./notebooks/data_product_quickstart.ipynb). This provides a step-by-step guide on how to set up the environment, run the scripts, and visualize the results.