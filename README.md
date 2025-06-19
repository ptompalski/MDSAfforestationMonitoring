# MDS Capstone Project: Afforestation Monitoring

![tests](https://img.shields.io/github/actions/workflow/status/ptompalski/MDSAfforestationMonitoring/run_test.yaml?branch=main&label=tests)


## Summary

Monitoring afforestation progress across hundreds of remote and ecologically diverse sites in Canada poses significant challenge, particularly due to the weak spectral signals from newly planted trees with sparse canopies in early growth stages. This project seeks to address two key research questions:

1. Can satellite-derived vegetation indices and site-level data be used to accurately predict tree survival over time in large-scale afforestation programs?

2. What modeling approaches are most effective for this task?

Using data from Canada’s 2 Billion Trees initiative, we began by training a suite of classical machine learning models, including the [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) and[Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) models from the[Scikit-Learn library](https://scikit-learn.org/stable/index.html) in Python, and the [Gradient Boosting Machine](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn) model from the [XGBoost library](https://xgboost.readthedocs.io/en/latest/index.html). Following this, we attempted a different modelling approach using more advanced sequential deep learning methods, namely [Long-term Short Term Memory (LSTM)](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html) and [Gated Recurrent Unit (GRU)](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html) models implemented in the [PyTorch library](https://pytorch.org/).

## Report

This project's report is developed using Quarto, a reproducible publishing system.

### Prerequisites

Ensure Quarto CLI is installed. For installation instructions, visit [Quarto Get Started](https://quarto.org/docs/get-started/).

### Rendering the Reports

To generate the proposal report, execute the following command in your terminal:

``` bash
quarto render reports/proposal/report.qmd --to pdf
```

To generate the technical report, execute the following command in your terminal:

``` bash
quarto render reports/technical/report.qmd --to pdf
```

To generate the final report, execute the following command in your terminal:

``` bash
quarto render reports/final/report.qmd --to pdf
```

## Quick Start

To quickly get started with the project, you can refer to the notebook [`Data Product Quick Start`](./notebooks/data_product_quickstart.ipynb). This notebook provides a step-by-step guide on how to set up the environment, run the scripts, and visualize the results.