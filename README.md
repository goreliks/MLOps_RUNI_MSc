# Anomaly Detection in Water Treatment Systems

This repository contains code for a machine learning project focused on anomaly detection in water treatment systems, developed for AquaFlow Technologies. It includes algorithms and data processing and improvement scripts for predicting anomalies in water treatment operations.

## Repository Overview

- `reports`: Reports and Presentation
  - `MLOps_Midterm_Project.pdf`: The fixed mid-term exercise
  - `baseline.pdf`: Baseline report for one of the baseline models.
  - `final_model_report.pdf`: Model report including the target metric comparison to the relevant baseline model
  - `exit_report.pdf`: Final Report
  - `MLOps_AquaFlow_project.pdf`: Presentation


- `algorithms`: Contains algorithm implementations.
  - `MSET.py`: Python implementation of the Multivariate State Estimation Technique (MSET).

- `transformers`: Includes implementation of the preprocessing transformers using `sklearn pipeline`.
  - `kaiser_window_smoother.py`: Applies Kaiser window smoothing.
  - `temporal_feature_creator.py`: Generates temporal features.
  - `shap_feature_selector.py`: Feature selection based on SHAP values.

- `data`: Directory containing data files used in the project (https://github.com/waico/SKAB/tree/master/data)

- `notebooks`: Jupyter notebooks for baseline pipelines and experimentation.
  - `MSET_pipeline_experiments.ipynb`: Baseline Pipeline and Experiments with MSET pipeline.
  - `LightGBM_pipeline_experiments.ipynb`: Baseline Pipeline and Experiments with LightGBM pipeline.

- `mset_pipeline.py`: Script to execute the MSET pipeline.

- `lightgbm_pipeline.py`: Script to execute the LightGBM pipeline.

- `config.py`: Configuration settings for the pipelines.

- `environment.yml`: Specifies the Conda environment required for the project.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Conda should be installed on your machine.

## Cloning the Repository and Setting Up the Environment

To clone the repository and set up the required environment, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/goreliks/MLOps_RUNI_MSc.git
   cd MLOps_RUNI_MSc

2. Create and activate the Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate mlops_fix

## Running the Project

This project includes two distinct pipelines: `lightgbm_pipeline.py` and `mset_pipeline.py`. Each pipeline can be executed with or without an improvement step. To configure this option, edit the `config.py` file and set `IMPROVEMENT_PIPELINE` to `True` for the improved pipeline or `False` for the baseline pipeline.

### LightGBM Pipeline
Run the LightGBM pipeline using:
   ```bash
   python3 lightgbm_pipeline.py
```

### MSET Pipeline
Run the MSET pipeline using:
   ```bash
   python3 mset_pipeline.py
```


**Note**:The MSET pipeline utilizes SHAP's KernelExplainer, which is computationally intensive. Depending on the sampling configuration, the execution may take several hours (up to 70 hours). This is due to the exhaustive computation required for generating SHAP values in complex models.
