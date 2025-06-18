# MDS Afforestation Monitoring

## Summary

Monitoring afforestation progress across hundreds of remote and ecologically diverse sites in Canada poses significant challenge, particularly due to the weak spectral signals from newly planted trees with sparse canopies in early growth stages. This project seeks to address two key research questions: (1) Can satellite-derived vegetation indices and site-level data be used to accurately predict tree survival over time in large-scale afforestation programs? and (2) What modeling approaches are most effective for this task? Using data from Canada’s 2 Billion Trees initiative, we train and evaluate a suite of machine learning models—including logistic regression, random forests, gradient boosting, and, if time permits, deep learning architectures such as RNNs and LSTMs. Addressing these questions is critical for monitoring afforestation efforts, which are esssential for climate mitigation, biodiversity conservation, and ecosystem restoration.

## Report

This project's report is developed using Quarto, a reproducible publishing system.

### Prerequisites

Ensure Quarto CLI is installed. For installation instructions, visit [Quarto Get Started](https://quarto.org/docs/get-started/).

### Rendering the Report

To generate the proposal report, execute the following command in your terminal:

```bash
quarto render reports/proposal/report.qmd
```

To generate the technical report, execute the following command in your terminal:

```bash
quarto render reports/technical/report.qmd
```

## Quick Start

To quickly get started with the project, you can refer to the notebook [`Data Product Quick Start`](./notebooks/data_product_quickstart.ipynb). This notebook provides a step-by-step guide on how to set up the environment, run the scripts, and visualize the results.
