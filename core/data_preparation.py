"""Dataset validation, and time-aware splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

from core.data_features import FreightFeatureEngineer


@dataclass
class DataSplit:
    """Container for train/validation partitions."""

    x_train: pd.DataFrame
    x_validation: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series


def training_data_check(
    data_frame: pd.DataFrame,
    target_column: str = "posted_rate",
    outlier_quantile: float = 0.995,
) -> pd.DataFrame:
    """
    Perform basic target validation.

    Invalid target rows are removed because regression training requires a
    finite, non-negative target value.
    """

    if target_column not in data_frame.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found in training data.")

    data_frame[target_column] = pd.to_numeric(
        data_frame[target_column], errors="coerce")

    # Remove invalid target rows: missing or negative values are not usable.
    valid_target_mask = data_frame[target_column].notna() & (
        data_frame[target_column] >= 0)
    filtered_df = data_frame.loc[valid_target_mask].reset_index(drop=True)

    if filtered_df.empty:
        raise ValueError(
            f"No valid rows found for target column '{target_column}'."
        )

    # Optionally remove extreme target outliers for more stable training.
    if outlier_quantile is not None:
        if not (0.0 < outlier_quantile < 1.0):
            raise ValueError("outlier_quantile must be between 0 and 1.")

        upper_limit = filtered_df[target_column].quantile(outlier_quantile)
        clean_mask = filtered_df[target_column] <= upper_limit
        dropped_outliers = int((~clean_mask).sum())

        if dropped_outliers:
            print(
                f"Dropped {dropped_outliers} target outliers (> {upper_limit:.2f}) "
                f"to stabilize RMSE."
            )

        filtered_df = filtered_df.loc[clean_mask].reset_index(drop=True)

    return filtered_df


def _parse_date_series(data_frame: pd.DataFrame, date_column: str) -> pd.Series:
    """Parse a date column safely into pandas datetime values."""
    if date_column not in data_frame.columns:
        return pd.Series(pd.NaT, index=data_frame.index)
    return pd.to_datetime(data_frame[date_column], errors="coerce")


def split_by_time(
    data_frame: pd.DataFrame,
    target_column: str = "posted_rate",
    date_column: str = "date",
    validation_start: Optional[str] = None,
    validation_fraction: float = 0.2,
) -> DataSplit:
    """
    Split data into train/validation using the latest time slice as validation.

    If validation_start is provided, all rows with date >= validation_start are
    used for validation. Otherwise, the most recent validation_fraction of rows
    with valid dates is used.
    """
    if not (0.0 < validation_fraction < 1.0):
        raise ValueError("validation_fraction must be between 0 and 1.")

    if target_column not in data_frame.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    features = data_frame.drop(columns=[target_column]).copy()
    target = data_frame[target_column].copy()
    parsed_dates = _parse_date_series(features, date_column)

    valid_date_mask = parsed_dates.notna()

    if validation_start is not None:
        cutoff = pd.Timestamp(validation_start)
        train_mask = valid_date_mask & (parsed_dates < cutoff)
        validation_mask = valid_date_mask & (parsed_dates >= cutoff)

        if validation_mask.sum() == 0:
            raise ValueError(
                f"No validation rows found for validation_start='{validation_start}'."
            )
        if train_mask.sum() == 0:
            raise ValueError(
                f"No training rows found before validation_start='{validation_start}'."
            )

        print(f"Using time split with validation_start={validation_start}.")
    else:
        if valid_date_mask.sum() == 0:
            raise ValueError(
                "No valid dates found for time-based split. Provide validation_start "
                "or clean the date column."
            )

        ordered_indices = parsed_dates[valid_date_mask].sort_values(
            kind="stable").index
        split_index = int(len(ordered_indices) * (1.0 - validation_fraction))
        split_index = max(1, min(split_index, len(ordered_indices) - 1))

        train_indices = ordered_indices[:split_index]
        validation_indices = ordered_indices[split_index:]

        train_mask = features.index.isin(train_indices)
        validation_mask = features.index.isin(validation_indices)

        print(
            f"Using time split with most recent {validation_fraction:.0%} of dated rows as validation."
        )

    x_train = features.loc[train_mask].reset_index(drop=True)
    x_validation = features.loc[validation_mask].reset_index(drop=True)
    y_train = target.loc[train_mask].reset_index(drop=True)
    y_validation = target.loc[validation_mask].reset_index(drop=True)

    return DataSplit(
        x_train=x_train,
        x_validation=x_validation,
        y_train=y_train,
        y_validation=y_validation,
    )


def split_by_random_holdout(
    data_frame: pd.DataFrame,
    target_column: str = "posted_rate",
    validation_fraction: float = 0.2,
    random_state: int = 42,
) -> DataSplit:
    """
    Create a random holdout split for quick sanity checks only.

    This is useful for debugging model behavior, but it should not be the main
    validation strategy when data is time-ordered.
    """
    if target_column not in data_frame.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    features = data_frame.drop(columns=[target_column]).copy()
    target = data_frame[target_column].copy()

    validation_size = int(len(features) * validation_fraction)
    validation_size = max(1, min(validation_size, len(features) - 1))

    shuffled_indices = features.sample(
        frac=1.0, random_state=random_state).index
    validation_indices = shuffled_indices[:validation_size]
    train_indices = shuffled_indices[validation_size:]

    return DataSplit(
        x_train=features.loc[train_indices].reset_index(drop=True),
        x_validation=features.loc[validation_indices].reset_index(drop=True),
        y_train=target.loc[train_indices].reset_index(drop=True),
        y_validation=target.loc[validation_indices].reset_index(drop=True),
    )


def analyze_dataset(
    data_frame: pd.DataFrame,
    target_column: str = "posted_rate",
    date_column: str = "date",
    high_missing_threshold: float = 0.30,
) -> None:
    """Perform dataset-level quality analysis and print detailed insights."""
    print("\n" + "=" * 80)
    print("                         DATASET ANALYSIS REPORT")
    print("=" * 80)

    n_rows, n_cols = data_frame.shape
    print(f"Shape: {n_rows:,} rows × {n_cols:,} columns")

    # 1. Missing values
    missing_counts = data_frame.isnull().sum()
    if missing_counts.sum() > 0:
        print("\n[Missing Values]")
        missing_summary = pd.DataFrame({
            "missing_count": missing_counts,
            "missing_pct": (missing_counts / n_rows * 100)
        }).sort_values("missing_pct", ascending=False)
        print(missing_summary[missing_summary["missing_count"] > 0].head(
            15).to_string())

        high_missing_cols = missing_summary[missing_summary["missing_pct"]
                                            > high_missing_threshold * 100]
        if not high_missing_cols.empty:
            print(
                f"\n[!] Warning: Columns with >{high_missing_threshold:.0%} missing values:")
            print(high_missing_cols.to_string())
    else:
        print("\n[✓] No missing values detected.")

    # 2. Duplicate rows
    dup_count = data_frame.duplicated().sum()
    print(
        f"\n[Duplicates]: {dup_count:,} duplicate rows ({dup_count / n_rows * 100:.2f}%)")

    # 3. Date Analysis
    if date_column in data_frame.columns:
        parsed_dates = pd.to_datetime(data_frame[date_column], errors="coerce")
        invalid_dates = parsed_dates.isna().sum()
        valid_dates = parsed_dates.dropna()
        print("\n[Date Analysis]")
        print(
            f"  Invalid dates: {invalid_dates:,} ({invalid_dates / n_rows * 100:.2f}%)")
        if not valid_dates.empty:
            print(f"  Borders: {valid_dates.min()}  to  {valid_dates.max()}")

    # 4. Target analysis
    if target_column in data_frame.columns:
        target = pd.to_numeric(data_frame[target_column], errors="coerce")
        valid_target = target.dropna()
        print(f"\n[Target Analysis: '{target_column}']")
        if not valid_target.empty:
            invalid_count = target.isna().sum()
            negative_count = (valid_target < 0).sum()
            zero_count = (valid_target == 0).sum()
            print(
                f"  Invalid/NaN: {invalid_count:,} | Negatives: {negative_count:,} | Zeros: {zero_count:,}"
            )

            print(valid_target.describe().to_frame().T.to_string())

            q1 = valid_target.quantile(0.25)
            q3 = valid_target.quantile(0.75)
            iqr = q3 - q1
            outliers = ((valid_target < (q1 - 1.5 * iqr)) |
                        (valid_target > (q3 + 1.5 * iqr))).sum()
            print(
                f"  Outliers (IQR method): {outliers:,} ({outliers / len(valid_target) * 100:.2f}%)")

            neg_count = (valid_target < 0).sum()
            zero_count = (valid_target == 0).sum()
            if neg_count > 0 or zero_count > 0:
                print(
                    f"  [!] Anomalies -> Negatives: {neg_count}, Zeroes: {zero_count}")

    # 5. Numeric columns review
    numeric_cols = [c for c in data_frame.select_dtypes(
        include=["number"]).columns if c != target_column]
    if numeric_cols:
        print("\n[Numeric Columns Audit]")
        for col in numeric_cols:
            col_data = pd.to_numeric(data_frame[col], errors="coerce").dropna()
            if col_data.empty:
                continue
            neg_count = (col_data < 0).sum()
            zero_count = (col_data == 0).sum()
            print(
                f"  - {col:18} | min={col_data.min():10.2f} | max={col_data.max():10.2f} | Negatives={neg_count:4} | Zeros={zero_count:4}")

    print("=" * 80 + "\n")


class DynamicPreprocessor(BaseEstimator, TransformerMixin):
    """
    Execute feature engineering first, then fit a ColumnTransformer based on
    the columns generated by feature engineering.
    """

    def __init__(
        self,
        id_column: str = "load_id",
        date_column: str = "date",
        drop_columns: Optional[list[str]] = None,
    ) -> None:
        self.id_column = id_column
        self.date_column = date_column
        self.drop_columns = drop_columns

    def fit(self, x: pd.DataFrame, y=None) -> "DynamicPreprocessor":
        self.feature_engineer_ = FreightFeatureEngineer(
            id_column=self.id_column,
            date_column=self.date_column,
        )

        engineered_data = self.feature_engineer_.fit_transform(x, y)

        self.feature_columns_ = engineered_data.columns.tolist()
        # print(self.feature_columns_)
        numeric_columns = engineered_data.select_dtypes(
            include=["number", "bool"]
        ).columns.tolist()

        categorical_columns = [
            column
            for column in engineered_data.columns
            if column not in numeric_columns
        ]

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        self.column_transformer_ = ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, numeric_columns),
                ("categorical", categorical_pipeline, categorical_columns),
            ],
            remainder="drop",
        )

        self.column_transformer_.fit(engineered_data, y)

        return self

    def transform(self, x: pd.DataFrame):
        engineered_data = self.feature_engineer_.transform(x)

        # Keep exactly the feature schema seen during training.
        engineered_data = engineered_data.reindex(
            columns=self.feature_columns_
        )

        return self.column_transformer_.transform(engineered_data)

    def fit_transform(self, x: pd.DataFrame, y=None):
        self.fit(x, y)
        return self.transform(x)


def build_preprocessor() -> DynamicPreprocessor:
    """Create the feature-engineering and encoding preprocessor."""

    return DynamicPreprocessor(
        id_column="load_id",
        date_column="date",
        drop_columns=[],
    )
