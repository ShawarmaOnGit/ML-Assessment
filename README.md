# Credit Card Fraud Detection

An end-to-end ML pipeline for customer segmentation and fraud detection using the Kaggle Credit Card Fraud dataset.

## Project Structure
```
├── credit_card_fraud_detection.ipynb  # Analysis with commentary
├── pipeline.py                        # Clean production pipeline
├── requirements.txt                   # Dependencies
└── plots/                             # Visualizations
```

## Files

**`credit_card_fraud_detection.ipynb`**
This is the main analysis notebook. Contains data analysis, visualizations, commentary explaining every decision, and the full modeling pipeline with outputs. Start here to understand the approach and findings.

**`pipeline.py`**
Clean, PEP-8 compliant production script that runs the full pipeline end to end. Structured with numbered sections, reusable functions, and docstrings.

## Pipeline Overview

1. Data loading via Kaggle API
2. Data analysis and preprocessing (RobustScaler, stratified split)
3. Customer segmentation (K-Means, K=2)
4. Class imbalance handling (SMOTE, sampling_strategy=0.1)
5. Stage 1 models (Logistic Regression, Random Forest, XGBoost)
6. Cost-sensitive threshold analysis
7. Stage 2 model (MLP, PyTorch)
8. Model comparison and recommendation

## Result

**Recommended model: Random Forest**
Highest F1 (0.85) and ROC-AUC (0.9926) with only 9 false alarms.
Deploy at threshold 0.57 based on cost-sensitive analysis.

## Setup

```bash
pip install -r requirements.txt
python pipeline.py
```