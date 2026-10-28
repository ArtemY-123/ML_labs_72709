from pathlib import Path
import hashlib
import json
import logging
import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

RESULTS = Path("../results")
RESULTS.mkdir(exist_ok=True)

allowed_categories = {
    "workclass": "Private|Self-emp-not-inc|Self-emp-inc|Federal-gov|Local-gov|State-gov|Without-pay|Never-worked",
    "education": "Bachelors|Some-college|11th|HS-grad|Prof-school|Assoc-acdm|Assoc-voc|9th|7th-8th|12th|Masters|1st-4th|10th|Doctorate|5th-6th|Preschool",
    "marital-status": "Married-civ-spouse|Divorced|Never-married|Separated|Widowed|Married-spouse-absent|Married-AF-spouse",
    "occupation": "Tech-support|Craft-repair|Other-service|Sales|Exec-managerial|Prof-specialty|Handlers-cleaners|Machine-op-inspct|Adm-clerical|Farming-fishing|Transport-moving|Priv-house-serv|Protective-serv|Armed-Forces",
    "relationship": "Wife|Own-child|Husband|Not-in-family|Other-relative|Unmarried",
    "race": "White|Asian-Pac-Islander|Amer-Indian-Eskimo|Other|Black",
    "sex": "Female|Male",
    "native-country": "United-States|Cambodia|England|Puerto-Rico|Canada|Germany|Outlying-US(Guam-USVI-etc)|India|Japan|Greece|South|China|Cuba|Iran|Honduras|Philippines|Italy|Poland|Jamaica|Vietnam|Mexico|Portugal|Ireland|France|Dominican-Republic|Laos|Ecuador|Taiwan|Haiti|Columbia|Hungary|Guatemala|Nicaragua|Scotland|Thailand|Yugoslavia|El-Salvador|Trinadad&Tobago|Peru|Hong|Holand-Netherlands"
}

allowed_categories = {
    key: set(value.split("|")) | {"?"}
    for key, value in allowed_categories.items()
}

ranges = {
    "age": (17, 90),
    "hours-per-week": (1, 99),
    "capital-gain": (0, 99999),
    "capital-loss": (0, 4356)
}

education_levels = {
    "Preschool": 1,
    "1st-4th": 2,
    "5th-6th": 3,
    "7th-8th": 4,
    "9th": 5,
    "10th": 6,
    "11th": 7,
    "12th": 8,
    "HS-grad": 9,
    "Some-college": 10,
    "Assoc-voc": 11,
    "Assoc-acdm": 12,
    "Bachelors": 13,
    "Masters": 14,
    "Prof-school": 15,
    "Doctorate": 16
}


class OutlierClipper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        data = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.percentile(data, 1, axis=0)
        self.upper_bounds_ = np.percentile(data, 99, axis=0)
        return self

    def transform(self, X):
        return np.clip(
            np.asarray(X, dtype=float),
            self.lower_bounds_,
            self.upper_bounds_
        )


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, enabled=True):
        self.enabled = enabled

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()

        for col in data.select_dtypes(
            include=["category", "object", "string"]
        ).columns:
            data[col] = data[col].astype("object").map(
                lambda v: np.nan
                if pd.isna(v) or str(v).strip() == "?"
                else str(v).strip()
            )

        if self.enabled:
            gain = pd.to_numeric(data["capital-gain"], errors="coerce")
            loss = pd.to_numeric(data["capital-loss"], errors="coerce")
            hours = pd.to_numeric(data["hours-per-week"], errors="coerce")
            age = pd.to_numeric(data["age"], errors="coerce")

            data["capital_net"] = gain - loss
            data["hours_per_age"] = hours / (age - 16).replace(0, np.nan)

        return data


def validate_schema(df, numeric_features, categorical_features, target):
    expected = numeric_features + categorical_features + [target]
    missing_columns = sorted(set(expected) - set(df.columns))

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    report = {
        "rows": len(df),
        "columns": len(df.columns),
        "types": {},
        "missing_values": {},
        "range_violations": {},
        "invalid_categories": {},
        "exact_duplicate_rows": int(df.duplicated().sum())
    }

    for col in numeric_features:
        numbers = pd.to_numeric(df[col], errors="coerce")

        non_integer = df[col].notna() & (
            numbers.isna() |
            ((numbers % 1 != 0) & numbers.notna())
        )

        report["types"][col] = {
            "actual": str(df[col].dtype),
            "expected": "integer",
            "dtype_valid": bool(pd.api.types.is_integer_dtype(df[col])),
            "invalid_values": int(non_integer.sum())
        }

        report["missing_values"][col] = int(numbers.isna().sum())

        if col in ranges:
            low, high = ranges[col]
            report["range_violations"][col] = int(
                ((numbers < low) | (numbers > high)).sum()
            )

    for col in categorical_features:
        values = df[col].astype("string").str.strip().fillna("?")

        report["types"][col] = {
            "actual": str(df[col].dtype),
            "expected": "category",
            "dtype_valid": bool(
                isinstance(df[col].dtype, pd.CategoricalDtype)
            )
        }

        report["missing_values"][col] = int(values.eq("?").sum())

        report["invalid_categories"][col] = int(
            (~values.isin(allowed_categories[col])).sum()
        )

    labels = df[target].astype("string").str.strip().str.rstrip(".")

    report["types"][target] = {
        "actual": str(df[target].dtype),
        "expected": "category",
        "dtype_valid": bool(
            isinstance(df[target].dtype, pd.CategoricalDtype)
        )
    }

    report["missing_values"][target] = int(labels.isna().sum())

    report["invalid_categories"][target] = int(
        (~labels.isin(["<=50K", ">50K"])).sum()
    )

    (RESULTS / "validation_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    logging.info("Validation report saved")
    logging.info("Range violations: %s", report["range_violations"])
    logging.info("Missing values: %s", report["missing_values"])
    logging.info("Invalid categories: %s", report["invalid_categories"])

    if report["invalid_categories"][target]:
        raise ValueError("Unexpected target labels")

    return report


def audit_labels(df, target, seed):
    features = df.drop(columns=[target]).astype("string")
    features = features.fillna("__MISSING__").replace("?", "__MISSING__")

    keys = pd.util.hash_pandas_object(features, index=False)

    groups = pd.DataFrame({
        "key": keys.to_numpy(),
        "label": df[target].astype("string").str.strip().to_numpy()
    })

    conflicting_keys = groups.groupby("key")["label"].nunique()
    conflicting_keys = conflicting_keys[conflicting_keys > 1].index

    conflict_groups = len(conflicting_keys)
    conflict_rows = int(groups["key"].isin(conflicting_keys).sum())

    sample = df.sample(n=50, random_state=seed).copy()

    def review(row):
        issues = []

        for col, (low, high) in ranges.items():
            value = pd.to_numeric(row[col], errors="coerce")

            if pd.notna(value) and not low <= value <= high:
                issues.append(col)

        for col in allowed_categories:
            value = "?" if pd.isna(row[col]) else str(row[col]).strip()

            if value not in allowed_categories[col]:
                issues.append(col)

        education = str(row["education"]).strip()
        level = pd.to_numeric(row["education-num"], errors="coerce")

        if education in education_levels and pd.notna(level):
            if level != education_levels[education]:
                issues.append("education-num mismatch")

        weight = pd.to_numeric(row["fnlwgt"], errors="coerce")

        if pd.notna(weight) and weight <= 0:
            issues.append("fnlwgt")

        return ", ".join(issues) if issues else "No automated issues"

    sample["dictionary_check"] = sample.apply(review, axis=1)
    sample["manual_review"] = "Pending manual verification"

    sample.to_csv(
        RESULTS / "sample_review.csv",
        index=True,
        index_label="original_row"
    )

    logging.info("Conflicting-label groups: %s", conflict_groups)
    logging.info("Conflicting-label rows: %s", conflict_rows)
    logging.info("50 rows saved for manual review")

    return {
        "conflicting_feature_groups": int(conflict_groups),
        "conflicting_label_rows": conflict_rows,
        "sample_size": 50,
        "automated_sample_issues": int(
            (sample["dictionary_check"] != "No automated issues").sum()
        ),
        "manual_review_status": "Pending manual verification"
    }


def make_pipeline(numeric_features, categorical_features, seed, engineered):
    numbers = numeric_features + (
        ["capital_net", "hours_per_age"] if engineered else []
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", OutlierClipper()),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numbers),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    return Pipeline([
        ("features", FeatureEngineer(enabled=engineered)),
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000, random_state=seed))
    ])


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        original_config = yaml.safe_load(f)

    config = original_config["pipeline"]

    seed = int(config["seed"])
    target = config["target_column"]
    numeric_features = list(config["numeric_features"])
    categorical_features = list(config["categorical_features"])

    logging.info("Fetching UCI Adult dataset...")

    adult = fetch_openml(
        name="adult",
        version=2,
        as_frame=True,
        parser="auto"
    )

    raw = adult.frame.copy(deep=True)

    logging.info(
        "Ingested %s rows and %s columns",
        len(raw),
        len(raw.columns)
    )

    report = validate_schema(
        raw,
        numeric_features,
        categorical_features,
        target
    )

    report["label_quality"] = audit_labels(raw, target, seed)

    (RESULTS / "validation_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    labels = raw[target].astype("string").str.strip().str.rstrip(".")
    y = labels.eq(">50K").astype(int)
    X = raw.drop(columns=[target])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["test_size"],
        random_state=seed,
        stratify=y
    )

    train_data = X_train.copy()
    train_data[target] = y_train

    duplicate_mask = train_data.duplicated()
    removed = int(duplicate_mask.sum())

    X_train = X_train.loc[~duplicate_mask]
    y_train = y_train.loc[~duplicate_mask]

    logging.info(
        "Train: %s, Test: %s, Duplicates removed: %s",
        len(X_train),
        len(X_test),
        removed
    )

    scores = {}

    for name, engineered in [
        ("baseline", False),
        ("engineered", True)
    ]:
        pipeline = make_pipeline(
            numeric_features,
            categorical_features,
            seed,
            engineered
        )

        pipeline.fit(X_train, y_train)
        predicted = pipeline.predict(X_test)

        scores[name] = {
            "accuracy": float(accuracy_score(y_test, predicted)),
            "f1": float(f1_score(y_test, predicted))
        }

        logging.info(
            "%s - Accuracy: %.4f, F1: %.4f",
            name,
            scores[name]["accuracy"],
            scores[name]["f1"]
        )

        if engineered:
            final_pipeline = pipeline

    scores["accuracy_change"] = (
        scores["engineered"]["accuracy"] -
        scores["baseline"]["accuracy"]
    )

    scores["f1_change"] = (
        scores["engineered"]["f1"] -
        scores["baseline"]["f1"]
    )

    (RESULTS / "metrics.json").write_text(
        json.dumps(scores, indent=2),
        encoding="utf-8"
    )

    fitted_preprocessor = Pipeline(final_pipeline.steps[:-1])

    joblib.dump(
        fitted_preprocessor,
        RESULTS / "preprocessor.joblib"
    )

    joblib.dump(
        final_pipeline,
        RESULTS / "model_pipeline.joblib"
    )

    processed = {
        "X_train": fitted_preprocessor.transform(X_train),
        "X_test": fitted_preprocessor.transform(X_test),
        "y_train": y_train.to_numpy(),
        "y_test": y_test.to_numpy()
    }

    processed_file = RESULTS / "processed_data.joblib"
    joblib.dump(processed, processed_file)

    dataset_hash = hashlib.sha256(
        processed_file.read_bytes()
    ).hexdigest()

    updated_config = dict(original_config)

    updated_config["dataset"] = {
        "name": "UCI Adult",
        "openml_version": 2,
        "raw_rows": len(raw),
        "raw_columns": len(raw.columns)
    }

    updated_config["processing"] = {
        "numeric_imputer": "median",
        "categorical_imputer": "most_frequent",
        "clip_percentiles": [1, 99],
        "drop_duplicates": "training_only",
        "derived_features": ["capital_net", "hours_per_age"]
    }

    updated_config["artifacts"] = {
        "preprocessor": "results/preprocessor.joblib",
        "model": "results/model_pipeline.joblib",
        "processed_dataset": "results/processed_data.joblib",
        "sha256": dataset_hash
    }

    (RESULTS / "config.yaml").write_text(
        yaml.safe_dump(updated_config, sort_keys=False),
        encoding="utf-8"
    )

    problem_definition = "\n\n".join([
        "This project develops a reproducible machine learning pipeline to predict whether an individual's annual income exceeds $50,000. The UCI Adult dataset contains 48,842 records and 14 input features, including demographic information, education, occupation, working hours and financial variables. The prediction target is a binary income category: at most $50,000 or more than $50,000 per year.",

        "The classification model is Logistic Regression. Performance is assessed with accuracy and F1-score. Accuracy measures the overall percentage of correct predictions. F1-score balances precision and recall and is particularly useful when the number of high-income and low-income records is unequal. Both measures are recorded on an unseen stratified test set to assess generalization.",

        "Data quality is checked before fitting models by inspecting expected column types, valid numeric ranges, missing values, accepted categorical levels, exact duplicate records and conflicting income labels. Fifty randomly selected records are also prepared for manual verification against the data dictionary. These checks preserve the original dataset and document potential problems instead of silently removing records.",

        "To avoid data leakage, the data is split 70/30 before cleaning or learning preprocessing parameters. Duplicate full records are removed from the training portion only, and the test portion remains untouched. Training-only median and mode imputers handle missing numerical and categorical values. Numeric outliers are clipped using training-set 1st and 99th percentiles. Two derived features, capital_net and hours_per_age, are compared against a baseline model. Fixed random seeds, saved preprocessing objects, a YAML configuration, and a SHA256 hash support reproducibility."
    ])

    (RESULTS / "problem_definition.md").write_text(
        problem_definition,
        encoding="utf-8"
    )

    data_card = "\n".join([
        "# UCI Adult Data Card",
        "",
        "## Source and license",
        "",
        "UCI Machine Learning Repository, Adult Census Income dataset, distributed through OpenML (adult, version 2).",
        "https://archive.ics.uci.edu/dataset/2/adult",
        "License: Creative Commons Attribution 4.0 International (CC BY 4.0).",
        "",
        "## Size and task",
        "",
        f"Raw rows: {len(raw)}. Raw columns: {len(raw.columns)}. Input features: {len(numeric_features) + len(categorical_features)}.",
        f"Training rows after duplicate removal: {len(X_train)}. Test rows: {len(X_test)}.",
        "Target: annual income above $50,000 (binary classification).",
        "",
        "## Processing and quality",
        "",
        "Stratified 70/30 split. Training-only numeric median and categorical mode imputation.",
        "Training-only 1st/99th percentile clipping, followed by standardization of numeric columns.",
        f"Exact duplicates removed from training: {removed}. Test records were not dropped.",
        "Engineered features: capital_net and hours_per_age.",
        f"Raw exact duplicate rows: {report['exact_duplicate_rows']}.",
        f"Conflicting-label feature groups: {report['label_quality']['conflicting_feature_groups']}.",
        f"Automatic issues among 50 sampled rows: {report['label_quality']['automated_sample_issues']}.",
        "Manual checking of the 50-row sample is pending; see sample_review.csv.",
        "",
        "## Known biases",
        "",
        "The historical US census data may reflect unequal economic opportunities and patterns associated with age, race, sex and education. The data may not generalize to current populations or economies.",
        "",
        "## Privacy concerns",
        "",
        "The dataset contains demographic and financial characteristics. Combinations of attributes may raise re-identification or fairness concerns. This model should not be used as the sole basis of high-stakes decisions.",
        "",
        "## Model results",
        "",
        f"Baseline: accuracy {scores['baseline']['accuracy']:.4f}, F1 {scores['baseline']['f1']:.4f}.",
        f"Engineered: accuracy {scores['engineered']['accuracy']:.4f}, F1 {scores['engineered']['f1']:.4f}.",
        "",
        "## Reproducibility",
        "",
        "Run: python pipeline.py (requires config.yaml alongside the script).",
        "Saved preprocessor: results/preprocessor.joblib.",
        "Saved model: results/model_pipeline.joblib.",
        "Processed dataset: results/processed_data.joblib.",
        "Full generated configuration: results/config.yaml.",
        f"SHA256 of processed dataset file: {dataset_hash}."
    ])

    (RESULTS / "data_card.md").write_text(
        data_card,
        encoding="utf-8"
    )

    logging.info("SHA256: %s", dataset_hash)
    logging.info("All Lab 4 artifacts saved in results/")

    for file in sorted(RESULTS.iterdir()):
        logging.info("%s", file)


if __name__ == "__main__":
    main()