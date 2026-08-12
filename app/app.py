"""Main executable for training freight-rate models and generating submission outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from core.data_preparation import training_data_check, split_by_time, analyze_dataset
from core.modeling import (
    analyze_residuals_and_errors,
    test_quote_signal_leakage,
    train_final_model,
    save_model,
    train_and_evaluate,
    predict_rates
)

DECEMBER_BASE_COLUMNS = [
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "date",
]

MODEL_RAW_COLUMNS = [
    "load_id",
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "equipment",
    "weight",
    "date",
    "market_index",
    "quote_signal",
]

DECEMBER_PREFERRED_MONTHS = [10, 1, 2, 9, 8, 3, 7, 4, 6, 5]


def _filter_by_preferred_months(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Keep rows whose calendar month best matches December (time+season)."""
    if frame.empty or "date" not in frame.columns:
        return frame

    parsed_dates = pd.to_datetime(frame["date"], errors="coerce")
    return frame[parsed_dates.dt.month.isin(DECEMBER_PREFERRED_MONTHS)].copy()


def _select_route_history(
    input_data: pd.DataFrame,
    training_data: pd.DataFrame,
    min_samples: int = 25,
) -> pd.DataFrame:
    """
    Select the best reference history for December imputation.

    Expansion happens in a priority-ordered way, and a candidate tier is only
    used if it holds at least `min_samples` rows; otherwise we merge in the
    next fallback tier. Month preference follows both recency and season:
    {10, 1, 2} are closest to December, not {8, 9}.
    """
    required_columns = {"pickup", "delivery", "equipment", "distance"}
    missing_training_columns = required_columns - set(training_data.columns)
    if missing_training_columns:
        raise ValueError(
            "Training data is missing columns required for December fallback: "
            f"{sorted(missing_training_columns)}"
        )

    if input_data.empty:
        raise ValueError("December input is empty.")

    first_row = input_data.iloc[0]

    input_distance = pd.to_numeric(
        pd.Series([first_row["distance"]]), errors="coerce"
    ).iloc[0]

    def _distance_mask(frame: pd.DataFrame, tol: float) -> pd.Series:
        distances = pd.to_numeric(frame["distance"], errors="coerce")
        if pd.isna(input_distance):
            return pd.Series(True, index=frame.index)
        return distances.between(input_distance - tol, input_distance + tol)

    # Priority tiers, most route-specific first.
    tiers = [
        # 1) exact route + equipment
        (
            training_data["pickup"].astype(str).eq(str(first_row["pickup"]))
            & training_data["delivery"].astype(str).eq(str(first_row["delivery"]))
            & training_data["equipment"].astype(str).eq(str(first_row["equipment"]))
        ),
        # 2) exact route (any equipment), narrow distance tolerance
        (
            training_data["pickup"].astype(str).eq(str(first_row["pickup"]))
            & training_data["delivery"].astype(str).eq(str(first_row["delivery"]))
            & _distance_mask(training_data, tol=20)
        ),
        # 3) same equipment + similar distance (wide tolerance)
        (
            training_data["equipment"].astype(
                str).eq(str(first_row["equipment"]))
            & _distance_mask(training_data, tol=40)
        ),
        # 4) same equipment only
        (
            training_data["equipment"].astype(
                str).eq(str(first_row["equipment"]))
        ),
        # 5) nearest available months (no equipment/distance constraint)
        (
            pd.to_datetime(training_data["date"], errors="coerce")
            .dt.month.isin(DECEMBER_PREFERRED_MONTHS)
        ),
        # 6) full training data as final fallback
        pd.Series(True, index=training_data.index),
    ]

    selected = None
    for tier_mask in tiers:
        candidate = training_data.loc[tier_mask].copy()

        # Within the same tier, prefer the shortest relevant months window.
        filtered = _filter_by_preferred_months(candidate)

        if not filtered.empty and len(filtered) >= min_samples:
            selected = filtered
            break

        # If this tier had few rows, merge it into the next tier instead of
        # dropping it entirely (compensates for sparse routes).
        if filtered.empty:
            continue

        # Keep the partial tier but keep searching for a richer candidate.
        partial = filtered
        selected_candidate = None
        for next_mask in tiers[tiers.index(tier_mask) + 1:]:
            next_candidate = training_data.loc[next_mask].copy()
            combined = pd.concat(
                [partial, next_candidate], ignore_index=True
            ).drop_duplicates().reset_index(drop=True)
            combined = _filter_by_preferred_months(combined)

            if len(combined) >= min_samples:
                selected_candidate = combined
                break

        if selected_candidate is not None:
            selected = selected_candidate
            break

    if selected is None or selected.empty:
        selected = training_data.copy()

    return selected


def complete_december_model_input(
    december_data: pd.DataFrame,
    training_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Complete the December data with the raw features required by the model.

    Existing December columns are preserved. Missing values are filled using
    route-specific historical data, preferably from the latest available
    training months.
    """
    result = december_data.copy()

    required_december_columns = set(DECEMBER_BASE_COLUMNS)
    missing_columns = required_december_columns - set(result.columns)

    if missing_columns:
        raise ValueError(
            "December input is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # The December file contains an empty predicted_rate column.
    # It is an output column, not a model input. Remove it before prediction.
    result = result.drop(columns=["predicted_rate"], errors="ignore")

    route_history = _select_route_history(
        input_data=result,
        training_data=training_data,
    )

    # Numeric conversion in the fallback source.
    route_history = route_history.copy()

    for column in (
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "distance",
        "weight",
        "market_index",
        "quote_signal",
    ):
        if column in route_history.columns:
            route_history[column] = pd.to_numeric(
                route_history[column],
                errors="coerce",
            )

    # Fill missing geographic features from the matching route.
    coordinate_columns = [
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
    ]

    for column in coordinate_columns:
        if column not in result.columns:
            if column not in route_history.columns:
                raise ValueError(
                    f"Cannot reconstruct missing coordinate column '{column}'."
                )

            fallback_value = route_history[column].median()

            if pd.isna(fallback_value):
                raise ValueError(
                    f"No valid historical value found for '{column}'."
                )

            result[column] = float(fallback_value)

    # Fill market features route-specifically from recent history.
    for column in ("market_index", "quote_signal"):
        if column not in result.columns:
            if column not in route_history.columns:
                raise ValueError(
                    f"Cannot reconstruct missing model feature '{column}'."
                )

            fallback_value = route_history[column].median()

            if pd.isna(fallback_value):
                # Final numeric fallback, only if route history has no value.
                fallback_value = pd.to_numeric(
                    training_data[column],
                    errors="coerce",
                ).median()

            if pd.isna(fallback_value):
                raise ValueError(
                    f"No valid fallback value found for '{column}'."
                )

            result[column] = float(fallback_value)

    # load_id is not required by the model during inference, because the
    # feature engineer removes it. Add a deterministic ID only if needed.
    if "load_id" not in result.columns:
        result.insert(
            0,
            "load_id",
            [f"DEC-{index + 1:06d}" for index in range(len(result))],
        )

    # Ensure the date has the same general representation as training data.
    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce",
    )

    if result["date"].isna().any():
        invalid_count = int(result["date"].isna().sum())
        raise ValueError(
            f"December input contains {invalid_count} invalid date values."
        )

    # Verify every raw feature needed by the trained pipeline exists.
    missing_model_columns = set(MODEL_RAW_COLUMNS) - set(result.columns)

    if missing_model_columns:
        raise ValueError(
            "Could not complete December model input. Missing columns: "
            f"{sorted(missing_model_columns)}"
        )

    # Return the same raw schema expected by the trained model.
    return result[MODEL_RAW_COLUMNS]


def create_december_predictions(
    model,
    input_path: str,
    output_path: str,
    training_data: pd.DataFrame,
) -> None:
    """
    Generate the seven-column December prediction file.

    The input file already contains a blank predicted_rate column. Its values
    are replaced with model predictions while preserving the original row
    order and output schema.
    """
    original_data = read_csv(Path(input_path))

    expected_columns = DECEMBER_BASE_COLUMNS + ["predicted_rate"]

    missing_columns = set(expected_columns) - set(original_data.columns)
    if missing_columns:
        raise ValueError(
            "December input is missing columns: "
            f"{sorted(missing_columns)}"
        )

    model_input = complete_december_model_input(
        december_data=original_data,
        training_data=training_data,
    )

    predictions = predict_rates(model, model_input)

    if len(predictions) != len(original_data):
        raise ValueError(
            "Number of predictions does not match December input rows."
        )

    output_data = original_data.copy()
    output_data["predicted_rate"] = np.asarray(
        predictions,
        dtype=float,
    )

    # Keep exactly the scorer-required order.
    output_data = output_data[
        DECEMBER_BASE_COLUMNS + ["predicted_rate"]
    ]

    if output_data.isnull().any().any():
        null_columns = output_data.columns[
            output_data.isnull().any()
        ].tolist()

        raise ValueError(
            "December output contains null values in columns: "
            f"{null_columns}"
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_data.to_csv(output_path, index=False)

    print(
        f"Saved December predictions: {output_path} "
        f"({len(output_data):,} rows)"
    )


def create_predictions(model, input_path: str, output_path: str,
                       template_path: str | None = None) -> None:
    """Predict posted rates and save results in template-compatible CSV format."""
    input_data = read_csv(Path(input_path))

    predictions = predict_rates(model, input_data)

    if template_path:
        template = read_csv(Path(template_path))

        # Preserve the required template columns and their original row order.
        prediction_columns = [
            column for column in template.columns if column != "load_id"]
        if len(prediction_columns) != 1:
            raise ValueError(
                "Prediction template must contain 'load_id' and exactly one prediction column."
            )

        if len(template) != len(input_data):
            raise ValueError(
                "Prediction template and input data have different row counts.")

        output_data = template.copy()
        output_data[prediction_columns[0]] = predictions
    else:
        output_data = pd.DataFrame(
            {
                "load_id": input_data["load_id"] if "load_id" in input_data.columns else input_data.index,
                "posted_rate": predictions,
            }
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_data.to_csv(output_path, index=False)
    print(f"Saved predictions: {output_path}")


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        fail(f"file not found: {path}")
    try:
        return pd.read_csv(path)
    except Exception as exc:
        fail(f"could not read {path}: {exc}")


def parse_arguments() -> argparse.Namespace:
    """Parse executable paths and output settings from the command line."""
    parser = argparse.ArgumentParser(
        description="Freight-rate model training and prediction pipeline.")
    parser.add_argument("--train", default="data/train_test.csv")
    parser.add_argument("--validation", default="data/validation.csv")
    parser.add_argument("--validation-template",
                        default="data/validation_predictions_template.csv")
    parser.add_argument("--december-input",
                        default="data/december_chart_inputs.csv")
    parser.add_argument(
        "--model-output", default="artifacts/freight_rate_model.joblib")
    parser.add_argument("--validation-output",
                        default="data/validation_predictions.csv")
    parser.add_argument("--december-output",
                        default="data/december_chart_inputs.csv")
    parser.add_argument(
        "--regressor-loss",
        default="squared_error",
        choices=["squared_error", "absolute_error"],
        help="Loss function for HistGradientBoostingRegressor.",
    )
    parser.add_argument(
        "--sample-weight-strategy",
        default="none",
        choices=["none", "target_emphasis"],
        help="Optional sample weighting strategy used during training.",
    )
    parser.add_argument(
        "--outlier-quantile",
        type=float,
        default=0.995,
        help="Upper target quantile used to trim extreme posted_rate outliers. "
             "Use a value in (0, 1), or disable in code by passing None.",
    )
    parser.add_argument(
        "--top-error-pct",
        type=float,
        default=0.05,
        help="Fraction of worst validation errors to analyze in residual reporting.",
    )
    parser.add_argument(
        "--skip-leakage-test",
        action="store_true",
        help="Skip quote_signal ablation/leakage sensitivity test.",
    )

    return parser.parse_args()


def main() -> None:
    """Train, validate, persist, and use the freight-rate prediction model."""
    arguments = parse_arguments()

    print("=" * 80)
    print("FREIGHT-RATE TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Train file:               {arguments.train}")
    print(f"Validation input file:    {arguments.validation}")
    print(f"Regressor loss:           {arguments.regressor_loss}")
    print(f"Sample weight strategy:   {arguments.sample_weight_strategy}")
    print(f"Target outlier quantile:  {arguments.outlier_quantile}")
    print(f"Top error analysis pct:   {arguments.top_error_pct}")
    print(f"Skip leakage test:        {arguments.skip_leakage_test}")
    print("=" * 80)

    data_frame = read_csv(Path(arguments.train))

    analyze_dataset(data_frame=data_frame,
                    target_column="posted_rate",)

    training_data = training_data_check(
        data_frame=data_frame,
        target_column="posted_rate",
        outlier_quantile=arguments.outlier_quantile,
    )

    data_split = split_by_time(
        data_frame=training_data,
        target_column="posted_rate",
        date_column="date",
        validation_start="2025-10-01",
    )

    eval_model, metrics = train_and_evaluate(
        data_split,
        regressor_loss=arguments.regressor_loss,
        sample_weight_strategy=arguments.sample_weight_strategy,
    )

    print(
        f"Validation metrics | MAE: {metrics.mae:.4f} | "
        f"RMSE: {metrics.rmse:.4f} | RMSLE: {metrics.rmsle:.4f} | R2: {metrics.r2:.4f}"
    )

    # Error validation analysis
    val_predictions = eval_model.predict(data_split.x_validation)

    # Error and residuals analysis
    analyze_residuals_and_errors(
        validation_df=data_split.x_validation,
        y_true=data_split.y_validation,
        y_pred=val_predictions,
        top_pct=arguments.top_error_pct,
    )

    # testing quote_signal leakage
    if not arguments.skip_leakage_test:
        test_quote_signal_leakage(data_split)

    final_model = train_final_model(
        training_data=training_data,
        target_column="posted_rate",
        regressor_loss=arguments.regressor_loss,
        sample_weight_strategy=arguments.sample_weight_strategy,
    )

    save_model(
        model_pipeline=final_model,
        metrics=metrics,
        output_path=arguments.model_output,
    )

    # persisted_model = load_model(arguments.model_output)

    create_predictions(
        model=final_model,
        input_path=arguments.validation,
        output_path=arguments.validation_output,
        template_path=arguments.validation_template,
    )

    create_december_predictions(
        model=final_model,
        input_path=arguments.december_input,
        output_path=arguments.december_output,
        training_data=training_data,
    )

    # if not arguments.skip_score:
    #     run_scoring_script(arguments.score_script)


if __name__ == "__main__":
    main()
