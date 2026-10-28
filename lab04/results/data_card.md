# UCI Adult Data Card

## Source and license

UCI Machine Learning Repository, Adult Census Income dataset, distributed through OpenML (adult, version 2).
https://archive.ics.uci.edu/dataset/2/adult
License: Creative Commons Attribution 4.0 International (CC BY 4.0).

## Size and task

Raw rows: 48842. Raw columns: 15. Input features: 14.
Training rows after duplicate removal: 34165. Test rows: 14653.
Target: annual income above $50,000 (binary classification).

## Processing and quality

Stratified 70/30 split. Training-only numeric median and categorical mode imputation.
Training-only 1st/99th percentile clipping, followed by standardization of numeric columns.
Exact duplicates removed from training: 24. Test records were not dropped.
Engineered features: capital_net and hours_per_age.
Raw exact duplicate rows: 52.
Conflicting-label feature groups: 5.
Automatic issues among 50 sampled rows: 0.
Manual checking of the 50-row sample is pending; see sample_review.csv.

## Known biases

The historical US census data may reflect unequal economic opportunities and patterns associated with age, race, sex and education. The data may not generalize to current populations or economies.

## Privacy concerns

The dataset contains demographic and financial characteristics. Combinations of attributes may raise re-identification or fairness concerns. This model should not be used as the sole basis of high-stakes decisions.

## Model results

Baseline: accuracy 0.8540, F1 0.6601.
Engineered: accuracy 0.8578, F1 0.6696.

## Reproducibility

Run: python pipeline.py (requires config.yaml alongside the script).
Saved preprocessor: results/preprocessor.joblib.
Saved model: results/model_pipeline.joblib.
Processed dataset: results/processed_data.joblib.
Full generated configuration: results/config.yaml.
SHA256 of processed dataset file: 46f91e99d8e0eeeb75abd6637fa4354b734b029aeedbbe7168e1a3726448370f.