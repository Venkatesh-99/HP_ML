# Project Overview

*Helicobacter pylori* (*H. pylori*) is a globally prevalent gastric pathogen implicated in a spectrum of clinical outcomes, from chronic gastritis to gastric cancer. While most infected individuals develop gastritis, only a minority progress to severe diseases like gastric cancer, influenced by bacterial, host, and environmental factors. This study presents a supervised machine learning framework to predict whether an *H. pylori* infection will result in gastric cancer or a non-malignant outcome, using a combination of clinical and genome-derived features.

A curated dataset of 1,363 *H. pylori* genomes with host metadata and annotated genomic features was used. Feature extraction included gene presence/absence profiles, sequence descriptors from [iFeatureOmega](https://github.com/Superzchen/iFeatureOmega-CLI) and [MathFeature](https://github.com/Bonidia/MathFeature), and aggregate variant annotation features. The workflow trains a white-box logistic regression model and black-box eXtreme Gradient Boosting (XGBoost) and Random Forest models, utilizing SMOTE-NC to address class imbalance. SHAP values are used for model interpretability.

## Main Workflow

The main script, `main.py`, orchestrates the following steps:

1. **Load Dataset**: Reads the dataset from an Excel file.
2. **Data Cleaning**: (Optional, placeholder for future cleaning steps).
3. **Train/Test Split**: Stratified splitting to maintain class balance.
4. **Preprocessing**: Encodes categorical features and labels.
5. **Baseline Model**: Trains and evaluates a logistic regression model.
6. **Model Explanation**: Uses SHAP to interpret the baseline model.
7. **Feature Selection**: Selects important features using SHAP and Bayesian optimization.
8. **XGBoost Model**: Trains, calibrates, evaluates, and explains an XGBoost model.
9. **Random Forest Model**: Trains, calibrates, evaluates, and explains a Random Forest model.

## Scripts Directory

Each script in the `scripts/` folder is responsible for a specific part of the workflow:

- `load_dataset.py`: Loads the dataset from an Excel file.
- `split_and_preprocess.py`: Splits the data and preprocesses features/labels.
- `train_baseline_lr_model.py`: Trains a baseline logistic regression model.
- `evaluate_baseline_lr_model.py`: Evaluates the logistic regression model.
- `explain_baseline_lr_model.py`: Generates SHAP summary plots for the baseline model.
- `feature_selection.py`: Performs feature selection using SHAP and Bayesian optimization.
- `train_xgb_with_bayesopt.py`: Trains an XGBoost model with Bayesian optimization.
- `calibrate_model.py`: Calibrates classifiers for improved probability estimates.
- `evaluate_model.py`: Evaluates and plots results for trained models.
- `explain_black_box_models.py`: Generates SHAP plots for black-box models.
- `train_rf_with_bayesopt.py`: Trains a Random Forest model with Bayesian optimization.

## Installation

To set up the required dependencies, create a new conda environment using the provided `HP_ML.yml` file:

```bash
conda env create -f HP_ML.yml
conda activate HP_ML
```

### Installing iFeatureOmega and MathFeature

These tools are used for feature extraction and must be installed separately:

- **iFeatureOmega**
  - Visit the [iFeatureOmega GitHub page](https://github.com/Superzchen/iFeatureOmega-CLI) for full instructions.
  - Basic installation:
    ```bash
    pip install iFeatureOmegaCLI
    ```

- **MathFeature**
  - Visit the [MathFeature GitHub page](https://github.com/Bonidia/MathFeature) for full instructions.
  - Basic installation:
    ```bash
    git clone https://github.com/Bonidia/MathFeature.git MathFeature
    cd MathFeature 
    conda env create -f mathfeature-terminal.yml -n mathfeature-terminal
    ```

This will install all necessary packages and dependencies for running the workflow and notebooks.

## Usage

To run the workflow, execute:

```bash
python main.py
```

You will be prompted to enter the path to your dataset (e.g., `data/dataset.xlsx`).

## Results

Model outputs, evaluation reports, and SHAP plots are saved in the `results/` directory.

## Output Directory Structure

The main outputs are saved in the `results/` directory, organized as follows:

```
results/
├── LR_classification_report.csv
├── RF_classification_report.csv
├── XGB_classification_report.csv
└── figures/
    ├── LR_shap_summary_plot.png
    ├── XGB_calibrated_shap_summary.png
    ├── RF_calibrated_shap_summary.png
    ├── LR_ROC_PR.png
    ├── LR_ROC_PR.pdf
    ├── XGB_ROC_PR.png
    ├── XGB_ROC_PR.pdf
    ├── RF_ROC_PR.png
    └── RF_ROC_PR.pdf
```

- `results/` contains classification reports for each model.
- `results/figures/` contains SHAP summary plots, AUROC and AUPRC curves generated during model evaluation and interpretation.

## Jupyter Notebook

A notebook version of the workflow is available in `notebooks/HP_ML.ipynb`. This notebook provides step-by-step code cells and explanations for each stage of the analysis, making it easy to interactively explore the data, train models, and visualize results.

## References

- Chen, Z., et al. (2022). **iFeatureOmega: an integrative platform for engineering, visualization and analysis of features from molecular sequences, structural and ligand data sets**. [GitHub Repository](https://github.com/Superzchen/iFeatureOmega-CLI)
- Bonidia, R. P., et al. (2021). **MathFeature: feature extraction package for DNA, RNA and protein sequences based on mathematical descriptors**. [GitHub Repository](https://github.com/Bonidia/MathFeature)

If you use this repository for your research, please also cite the original tools:

- iFeatureOmega: Chen Z., et al., *Nucleic Acids Research*, 2022, [DOI](https://doi.org/10.1093/nar/gkac351).
- MathFeature: Bonidia R. P., et al., *Briefings in Bioinformatics*, 2022, [DOI](https://doi.org/10.1093/bib/bbab434).

## Contact

For questions or support, contact:  
- [Venkatesh N](mailto:venkateshn51099@gmail.com)
- [Sreya P Warrier](mailto:sreyapw@gmail.com)

---
