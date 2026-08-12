"""Training, evaluation, and persistence utilities for freight-rate regression."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from core.data_preparation import DataSplit, build_preprocessor


@dataclass
class ModelMetrics:
    """Store regression metrics calculated on the internal validation set."""

    mae: float
    rmse: float
    rmsle: float
    r2: float


def _build_regressor(
    loss: str = "squared_error",
    learning_rate: float = 0.025,
    max_iter: int = 900,
    max_leaf_nodes: int = 31,
    min_samples_leaf: int = 40,
    l2_regularization: float = 5.0,
    random_state: int = 42,
) -> HistGradientBoostingRegressor:
    """Create a configured gradient-boosting regressor.

    Notes:
    - squared_error is usually better when the optimization target is RMSE.
    - absolute_error is more robust to outliers and often improves MAE stability.
    """
    return HistGradientBoostingRegressor(
        loss=loss,
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        random_state=random_state,
    )


def build_model_pipeline(
    regressor_loss: str = "squared_error",
) -> Pipeline:
    """Create the complete preprocessing and regression pipeline.

    The regressor is wrapped in TransformedTargetRegressor so the model trains
    on log1p(y) and predictions are automatically mapped back with expm1(y).
    """
    base_regressor = _build_regressor(loss=regressor_loss)

    log_target_regressor = TransformedTargetRegressor(
        regressor=base_regressor,
        func=np.log1p,
        inverse_func=np.expm1
    )

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", log_target_regressor),
        ]
    )


def _calculate_metrics(
    actual_values: pd.Series,
    predicted_values: np.ndarray,
) -> ModelMetrics:
    """Calculate MAE, RMSE, and RMSLE after enforcing non-negative predictions."""
    safe_predictions = np.clip(predicted_values, 0.0, None)
    safe_actual_values = np.clip(
        actual_values.to_numpy(dtype=float), 0.0, None)

    return ModelMetrics(
        mae=float(mean_absolute_error(safe_actual_values, safe_predictions)),
        rmse=float(
            np.sqrt(
                mean_squared_error(
                    safe_actual_values,
                    safe_predictions,
                )
            )
        ),
        rmsle=float(
            np.sqrt(
                mean_squared_error(
                    np.log1p(safe_actual_values),
                    np.log1p(safe_predictions),
                )
            )
        ),
        r2=r2_score(safe_actual_values, safe_predictions)
    )


def _build_sample_weights(
    target_values: pd.Series,
    strategy: str = "none",
) -> np.ndarray | None:
    """Create optional sample weights for training.

    Strategies:
    - none: no weighting
    - target_emphasis: place moderately higher weight on larger target values
      to improve high-value RMSE without making training unstable
    """
    y = np.asarray(target_values, dtype=float)

    if strategy == "none":
        return None

    if strategy == "target_emphasis":
        # Use a gentle log-scaled weighting to avoid over-dominating extreme loads.
        weights = 1.0 + 0.5 * (
            np.log1p(np.clip(y, 0.0, None)) /
            np.log1p(np.nanmax(np.clip(y, 0.0, None)))
        )
        return weights

    raise ValueError(f"Unsupported sample-weight strategy: '{strategy}'")


def analyze_residuals_and_errors(
    validation_df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    top_pct: float = 0.05
) -> None:
    """Perform detailed analysis on prediction residuals and highlight top high-error groups."""
    report = validation_df.copy()
    report["actual"] = np.clip(y_true.to_numpy(dtype=float), 0.0, None)
    report["predicted"] = np.clip(y_pred, 0.0, None)
    report["error"] = report["actual"] - report["predicted"]
    report["abs_error"] = report["error"].abs()
    report["ape"] = report["abs_error"] / report["actual"].clip(lower=1.0)

    total_error_sum = report["abs_error"].sum()
    threshold = report["abs_error"].quantile(1 - top_pct)
    worst_subset = report[report["abs_error"] >= threshold]
    worst_error_sum = worst_subset["abs_error"].sum()

    print("\n" + "=" * 80)
    print(
        f"                 RESIDUALS & ERROR ANALYSIS REPORT (Worst {top_pct:.0%})")
    print("=" * 80)
    print(f"Overall Absolute Error Stats:")
    print(report[["abs_error", "ape"]].describe().to_string())

    print(f"\n[!] The top {top_pct:.0%} largest errors account for "
          f"{(worst_error_sum / total_error_sum) * 100:.2f}% of the total absolute validation error.")
    print(f"Error Threshold for worst {top_pct:.0%}: {threshold:.4f}")

    if "equipment" in report.columns:
        print("\n[Worst Errors Grouped by Equipment]")
        eq_group = worst_subset.groupby("equipment").agg(
            count=("abs_error", "size"),
            mae=("abs_error", "mean"),
            max_error=("abs_error", "max"),
            avg_actual=("actual", "mean")
        ).sort_values("mae", ascending=False)
        print(eq_group.to_string())

    if "distance" in report.columns:
        print("\n[Errors Quantized by Distance Buckets]")
        report["distance_bucket"] = pd.qcut(
            report["distance"], q=5, duplicates="drop"
        )

        dist_group = report.groupby("distance_bucket", observed=False).agg(
            count=("abs_error", "size"),
            mae=("abs_error", "mean"),
            rmse=("error", lambda x: np.sqrt(np.mean(x ** 2))),
            avg_actual=("actual", "mean")
        )
        print(dist_group.to_string())

        print("\n[Errors by Actual Rate Buckets]")
        report["actual_rate_bucket"] = pd.qcut(
            report["actual"], q=5, duplicates="drop"
        )
        rate_group = report.groupby("actual_rate_bucket", observed=False).agg(
            count=("abs_error", "size"),
            mae=("abs_error", "mean"),
            rmse=("error", lambda x: np.sqrt(np.mean(x ** 2))),
            avg_actual=("actual", "mean"),
            avg_predicted=("predicted", "mean"),
        )
        print(rate_group.to_string())

    print("\n[Top 10 Largest Individual Errors]")
    cols = ["load_id", "pickup", "delivery", "equipment", "distance",
            "weight", "market_index", "quote_signal", "actual", "predicted", "error"]
    display_cols = [c for c in cols if c in report.columns]
    print(report.sort_values("abs_error", ascending=False)
          [display_cols].head(10).to_string())
    print("=" * 80 + "\n")


def test_quote_signal_leakage(data_split: DataSplit) -> None:
    """Train a validation model without quote_signal to analyze drop in accuracy (Ablation Test)."""
    print("\n" + "=" * 80)
    print("                 DATA LEAKAGE TEST: ABLATION STUDY")
    print("=" * 80)

    if "quote_signal" not in data_split.x_train.columns:
        print("quote_signal is not in the features. Skipping leakage test.")
        print("=" * 80 + "\n")
        return

    # Create datasets without quote_signal
    x_train_ablation = data_split.x_train.drop(columns=["quote_signal"])
    x_val_ablation = data_split.x_validation.drop(columns=["quote_signal"])

    # Temporarily construct dynamic preprocessor and model for ablation
    pipeline_ablation = build_model_pipeline(regressor_loss="squared_error")

    # We must patch the preprocessor to not expect quote_signal
    # But since DynamicPreprocessor automatically infers columns during fit:
    pipeline_ablation.fit(x_train_ablation, data_split.y_train)
    preds = pipeline_ablation.predict(x_val_ablation)

    metrics_ablation = _calculate_metrics(data_split.y_validation, preds)
    print("Ablation Model (WITHOUT 'quote_signal'):")
    print(
        f"  MAE: {metrics_ablation.mae:.4f} | RMSE: {metrics_ablation.rmse:.4f} | RMSLE: {metrics_ablation.rmsle:.4f} | R2: {metrics_ablation.r2:.4f}")
    print("\n* If the performance drop is minimal, the pipeline is not overly reliant on quote_signal.")
    print("* If performance drops drastically (e.g. R2 drops below 0.70), ensure quote_signal is available at prediction time.")
    print("=" * 80 + "\n")


def train_and_evaluate(
    data_split: DataSplit,
    regressor_loss: str = "squared_error",
    sample_weight_strategy: str = "none",
) -> tuple[Pipeline, ModelMetrics]:
    """
    Train a model on the training partition and evaluate it on the time holdout.

    The validation partition is not used during model fitting.
    """
    model_pipeline = build_model_pipeline(regressor_loss=regressor_loss)

    sample_weights = _build_sample_weights(
        data_split.y_train,
        strategy=sample_weight_strategy,
    )

    fit_params = {}
    if sample_weights is not None:
        fit_params["regressor__sample_weight"] = sample_weights

    model_pipeline.fit(
        data_split.x_train,
        data_split.y_train,
        **fit_params,
    )

    validation_predictions = model_pipeline.predict(
        data_split.x_validation
    )

    metrics = _calculate_metrics(
        actual_values=data_split.y_validation,
        predicted_values=validation_predictions,
    )

    return model_pipeline, metrics


def train_final_model(
    training_data: pd.DataFrame,
    target_column: str = "posted_rate",
    regressor_loss: str = "squared_error",
    sample_weight_strategy: str = "none",
) -> Pipeline:
    """
    Train the final model using all available historical training records.

    This method must only be called after model configuration has been selected
    using the internal time-based validation set.
    """
    if target_column not in training_data.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found."
        )

    training_features = training_data.drop(
        columns=[target_column]
    )
    target_values = training_data[target_column]

    model_pipeline = build_model_pipeline(regressor_loss=regressor_loss)

    sample_weights = _build_sample_weights(
        target_values,
        strategy=sample_weight_strategy,
    )

    fit_params = {}
    if sample_weights is not None:
        fit_params["regressor__sample_weight"] = sample_weights

    model_pipeline.fit(
        training_features,
        target_values,
        **fit_params,
    )

    return model_pipeline


def save_model(
    model_pipeline: Pipeline,
    metrics: ModelMetrics,
    output_path: str,
) -> None:
    """Save the trained model and validation metrics as one versioned artifact."""
    model_path = Path(output_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    artifact: dict[str, Any] = {
        "model": model_pipeline,
        "validation_metrics": asdict(metrics),
        "target_column": "posted_rate",
    }

    joblib.dump(artifact, model_path)


def load_model(model_path: str) -> Pipeline:
    """Load a persisted model artifact and return its prediction pipeline."""
    artifact = joblib.load(model_path)

    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError(
            f"Invalid model artifact: '{model_path}'."
        )

    return artifact["model"]


def predict_rates(
    model_pipeline: Pipeline,
    input_data: pd.DataFrame,
) -> np.ndarray:
    """Generate non-negative freight-rate predictions."""
    predictions = model_pipeline.predict(input_data)
    return np.clip(np.asarray(predictions, dtype=float), 0.0, None)
