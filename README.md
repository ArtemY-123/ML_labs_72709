
# Lab 4: Data Pipeline and Quality

## 1. Problem Definition

The objective of this project is to develop a reproducible machine learning pipeline using the UCI Adult dataset. The task is to predict whether an individual's annual income exceeds $50,000 based on demographic, educational, occupational, and financial information.

The dataset contains 48,842 observations, 14 input features, and one target variable. This is a binary classification problem, where the output represents income above or below $50,000.

Logistic Regression is used as the classification model. The main evaluation metrics are accuracy and F1-score. Accuracy measures the proportion of correct predictions, while F1-score balances precision and recall.

The project focuses on data quality, reproducibility, and preventing data leakage. All preprocessing parameters are learned from the training data only. The model is evaluated on a separate test set.

Two additional features are introduced to investigate whether feature engineering can improve classification performance.

## 2. Dataset Ingestion

The dataset was downloaded from OpenML using:

`fetch_openml(name="adult", version=2, as_frame=True)`

Dataset characteristics:

| Property | Value |
|---|---|
| Dataset | UCI Adult |
| Observations | 48,842 |
| Columns | 15 |
| Input features | 14 |
| Target | class |
| Task | Binary classification |
| Source | OpenML |

The original dataset is preserved during ingestion.

## 3. Data Validation

Data validation was performed before model training to identify possible quality problems.

The following checks were implemented:

- Data type validation for numerical and categorical features.
- Missing-value detection, including values represented by `?`.
- Numeric range validation.
- Categorical value validation.
- Exact duplicate detection.
- Conflicting-label detection.

The required numerical ranges are:

| Feature | Minimum | Maximum |
|---|---:|---:|
| age | 17 | 90 |
| hours-per-week | 1 | 99 |
| capital-gain | 0 | 99999 |
| capital-loss | 0 | 4356 |

The initial pipeline run passed the implemented range checks.

The complete validation results are saved in `results/validation_report.json`.

## 4. Data Cleaning and Preprocessing

The dataset was divided into training and testing subsets using stratified sampling.

- Training data: 70%
- Testing data: 30%
- Random seed: 42

Stratification preserves the original distribution of the target classes.

### Missing Values

Numerical missing values are replaced using the median calculated from the training data.

Categorical missing values are replaced using the most frequent category from the training data.

### Outlier Treatment

Numerical features are clipped to the 1st and 99th percentiles calculated from the training data.

This limits the influence of extreme numerical values.

### Duplicate Removal

Exact duplicate records are removed from the training dataset only.

The test dataset remains untouched to preserve an unbiased evaluation.

### Data Leakage Prevention

All preprocessing operations that require learned parameters are fitted exclusively on the training dataset.

The fitted transformations are then applied to the test dataset without recalculating their parameters.

The fitted preprocessor is saved as `results/preprocessor.joblib`.

## 5. Feature Engineering

Two new numerical features were created.

### Capital Net

`capital_net = capital-gain - capital-loss`

This feature represents the difference between reported capital gains and capital losses.

### Hours per Age

`hours_per_age = hours-per-week / (age - 16)`

This feature represents weekly working hours relative to the individual's age above 16.

The purpose of feature engineering is to provide additional information that may improve the model's predictive performance.

Two models were trained and compared:

1. Baseline Logistic Regression without derived features.
2. Logistic Regression with the two derived features.

## 6. Label Quality Assessment

A label-quality audit was implemented to identify records with identical input features but conflicting income labels.

The earlier pipeline execution reported 5 groups of duplicate feature records with conflicting labels.

Such records may indicate labeling inconsistencies or different outcomes for individuals with identical recorded attributes.

Additionally, 50 random observations are selected for data-dictionary checks.

The automated checks inspect numerical ranges, categorical values, and consistency between education and education-num.

The selected records are saved in `results/sample_review.csv`.

Manual verification of these 50 records remains to be completed.

## 7. Model Training and Evaluation

Logistic Regression was used to evaluate the effect of feature engineering.

The model was trained on the cleaned training dataset and evaluated on the test dataset.

### Results

The following results were obtained from the initial pipeline execution:

| Model | Accuracy | F1-score |
|---|---:|---:|
| Baseline | 0.8454 | 0.6365 |
| With engineered features | 0.8463 | 0.6468 |

### Analysis

The baseline model achieved an accuracy of 84.54% and an F1-score of 0.6365.

After adding the two engineered features, accuracy increased to 84.63% and F1-score increased to 0.6468.

The accuracy improvement was 0.09 percentage points, while the F1-score increased by 0.0103.

These results suggest that the engineered features provide a small improvement in predictive performance.

Although the increase in accuracy is limited, the higher F1-score indicates a better balance between precision and recall for the positive income class.

The results demonstrate that feature engineering can improve model performance, although the effect depends on the usefulness of the additional information.

## 8. Reproducibility and SHA256 Hash

Reproducibility is supported by the use of:

- Fixed random seed.
- Stratified train/test splitting.
- Scikit-learn preprocessing pipelines.
- Saved model and preprocessing objects.
- YAML configuration.
- SHA256 hashing.

The processed dataset is stored in `results/processed_data.joblib`.

A SHA256 hash is generated to identify the processed dataset file and help verify whether its contents have changed.

The generated hash is stored in `results/config.yaml`.

The original configuration includes the seed, test size, target column, numerical features, and categorical features.

The complete pipeline can be executed with:

`python pipeline.py`

## 9. Data Card

A data card is generated and saved as `results/data_card.md`.

It includes the following information:

- Dataset source and license.
- Dataset size and characteristics.
- Data processing methods.
- Data quality results.
- Known demographic and historical biases.
- Privacy concerns.
- Model evaluation results.
- Reproducibility information.

The Adult dataset contains sensitive demographic attributes, including age, sex, race, and marital status.

These attributes may reflect historical social and economic inequalities.

The model should not be used as the sole basis for high-impact decisions about individuals.

## 10. Conclusion

The project demonstrates the development of a reproducible data pipeline for binary income classification using the UCI Adult dataset.

The workflow includes data ingestion, validation, preprocessing, feature engineering, model training, and evaluation.

Data leakage is prevented by fitting preprocessing operations only on training data and keeping the test set separate.

The addition of capital_net and hours_per_age produced a small improvement in accuracy and F1-score in the initial experiment.

The project also includes data-quality auditing, saved preprocessing objects, configuration files, and SHA256 hashing to support reproducibility.

The remaining manual quality review should be completed before considering the full data-quality assessment finished.

Overall, the experiment demonstrates how structured data processing and feature engineering can contribute to reliable machine learning workflows.

## 11. References

Becker, B., & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository.

https://doi.org/10.24432/C5XW20

OpenML. Adult dataset, version 2.

https://www.openml.org/d/1590

Scikit-learn Developers. Scikit-learn documentation.

https://scikit-learn.org/stable/
