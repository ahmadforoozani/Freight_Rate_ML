"""Integration and contract tests for the freight-rate ML pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.data_features import FreightFeatureEngineer
from core.data_preparation import split_by_time
from core.modeling import (
    ModelMetrics,
    load_model,
    predict_rates,
    save_model,
    train_and_evaluate,
    train_final_model,
)

from app.app import (
    complete_december_model_input,
    create_december_predictions,
    create_predictions,
)


VALIDATION_OUTPUT_COLUMNS = [
    "load_id",
    "predicted_rate",
]

DECEMBER_OUTPUT_COLUMNS = [
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "date",
    "predicted_rate",
]


def _make_training_data(
    rows_count: int = 60,
) -> pd.DataFrame:
    """Create a small deterministic training dataset for fast tests."""
    rows: list[dict] = []

    for index in range(rows_count):
        date = pd.Timestamp("2025-08-01") + pd.Timedelta(days=index)

        distance = 300.0 + (index % 5) * 20.0
        weight = 25000.0 + (index % 4) * 2000.0
        market_index = 0.85 + (index % 6) * 0.05
        quote_signal = 1.70 + (index % 5) * 0.10

        posted_rate = (
            450.0
            + distance * 0.55
            + weight * 0.004
            + market_index * 80.0
            + quote_signal * 20.0
        )

        rows.append(
            {
                "load_id": f"TR-{index + 1:05d}",
                "pickup": "Lexington",
                "delivery": "Fort Wayne",
                "pickup_lat": 38.0406,
                "pickup_lon": -84.5037,
                "delivery_lat": 41.0793,
                "delivery_lon": -85.1394,
                "distance": distance,
                "equipment": "Dry Van",
                "weight": weight,
                "date": date.strftime("%Y-%m-%d"),
                "market_index": market_index,
                "quote_signal": quote_signal,
                "posted_rate": posted_rate,
            }
        )

    return pd.DataFrame(rows)


def _make_december_input(
    rows_count: int = 31,
    include_predicted_rate: bool = True,
) -> pd.DataFrame:
    """Create a December input with intentionally missing model features."""
    dates = pd.date_range("2025-12-01", periods=rows_count, freq="D")

    data = pd.DataFrame(
        {
            "pickup": ["Lexington"] * rows_count,
            "delivery": ["Fort Wayne"] * rows_count,
            "distance": [360.0] * rows_count,
            "equipment": ["Dry Van"] * rows_count,
            "weight": [32000.0] * rows_count,
            "date": dates.strftime("%-m/%-d/%Y"),
        }
    )

    if include_predicted_rate:
        data["predicted_rate"] = np.nan

    return data


class FakeModel:
    """Minimal prediction-compatible model for output contract tests."""

    def __init__(self, prediction: float = 800.0) -> None:
        self.prediction = prediction

    def predict(self, input_data: pd.DataFrame) -> np.ndarray:
        return np.full(
            shape=len(input_data),
            fill_value=self.prediction,
            dtype=float,
        )


def test_feature_engineering_creates_expected_features() -> None:
    """Feature engineering should create core features and remove raw date/id."""
    training_data = _make_training_data(rows_count=1)
    raw_features = training_data.drop(columns=["posted_rate"])

    transformed = FreightFeatureEngineer().fit_transform(raw_features)

    expected_features = {
        "load_month",
        "load_day_of_week",
        "haversine_distance_miles",
        "road_to_air_distance_ratio",
        "distance_x_weight",
        "market_x_quote_signal",
        "log_distance",
        "log_weight",
        "is_dry_van",
    }

    assert expected_features.issubset(transformed.columns)
    assert "load_id" not in transformed.columns
    assert "date" not in transformed.columns

    numeric_values = transformed.select_dtypes(
        include=["number", "bool"]
    ).to_numpy(dtype=float)

    assert not np.isinf(numeric_values).any()


def test_time_split_prevents_future_data_leakage() -> None:
    """Rows on or after the cutoff must never be included in training."""
    data = _make_training_data(rows_count=10)

    split = split_by_time(
        data_frame=data,
        target_column="posted_rate",
        date_column="date",
        validation_start="2025-08-06",
    )

    train_dates = pd.to_datetime(split.x_train["date"])
    validation_dates = pd.to_datetime(split.x_validation["date"])

    assert len(split.x_train) == 5
    assert len(split.x_validation) == 5

    assert train_dates.max() < pd.Timestamp("2025-08-06")
    assert validation_dates.min() >= pd.Timestamp("2025-08-06")

    assert "posted_rate" not in split.x_train.columns
    assert "posted_rate" not in split.x_validation.columns


def test_train_and_predict_end_to_end() -> None:
    """The complete training and prediction flow should run successfully."""
    data = _make_training_data(rows_count=60)

    split = split_by_time(
        data_frame=data,
        target_column="posted_rate",
        date_column="date",
        validation_start="2025-09-15",
    )

    model, metrics = train_and_evaluate(
        data_split=split,
        regressor_loss="squared_error",
        sample_weight_strategy="none",
    )

    predictions = predict_rates(model, split.x_validation)

    assert len(predictions) == len(split.x_validation)
    assert np.isfinite(predictions).all()
    assert (predictions >= 0).all()

    assert np.isfinite(metrics.mae)
    assert np.isfinite(metrics.rmse)
    assert np.isfinite(metrics.rmsle)
    assert np.isfinite(metrics.r2)


def test_saved_model_can_be_loaded_and_used_for_prediction(tmp_path) -> None:
    """A persisted model must remain prediction-compatible after loading."""
    data = _make_training_data(rows_count=60)

    model = train_final_model(
        training_data=data,
        target_column="posted_rate",
        regressor_loss="squared_error",
        sample_weight_strategy="none",
    )

    model_path = tmp_path / "freight_rate_model.joblib"

    save_model(
        model_pipeline=model,
        metrics=ModelMetrics(
            mae=0.0,
            rmse=0.0,
            rmsle=0.0,
            r2=0.0,
        ),
        output_path=str(model_path),
    )

    loaded_model = load_model(str(model_path))

    input_data = data.drop(columns=["posted_rate"]).head(5)
    predictions = predict_rates(loaded_model, input_data)

    assert len(predictions) == 5
    assert np.isfinite(predictions).all()
    assert (predictions >= 0).all()


def test_validation_output_matches_template_schema(tmp_path) -> None:
    """Validation predictions must follow the required template schema."""
    validation_data = pd.DataFrame(
        {
            "load_id": ["VAL-001", "VAL-002", "VAL-003"],
            "pickup": ["A", "B", "C"],
            "delivery": ["X", "Y", "Z"],
        }
    )

    template = pd.DataFrame(
        {
            "load_id": ["VAL-001", "VAL-002", "VAL-003"],
            "predicted_rate": [np.nan, np.nan, np.nan],
        }
    )

    input_path = tmp_path / "validation.csv"
    template_path = tmp_path / "validation-template.csv"
    output_path = tmp_path / "validation-predictions.csv"

    validation_data.to_csv(input_path, index=False)
    template.to_csv(template_path, index=False)

    create_predictions(
        model=FakeModel(prediction=777.25),
        input_path=str(input_path),
        output_path=str(output_path),
        template_path=str(template_path),
    )

    output = pd.read_csv(output_path)

    assert list(output.columns) == VALIDATION_OUTPUT_COLUMNS
    assert len(output) == len(validation_data)
    assert output["load_id"].tolist() == validation_data["load_id"].tolist()
    assert output["predicted_rate"].notna().all()
    assert (output["predicted_rate"] >= 0).all()
    assert output["predicted_rate"].tolist() == [777.25] * 3


def test_december_input_is_completed_with_missing_model_features() -> None:
    """Missing December features must be reconstructed from route history."""
    training_data = _make_training_data(rows_count=60)
    december_data = _make_december_input(include_predicted_rate=True)

    completed = complete_december_model_input(
        december_data=december_data,
        training_data=training_data,
    )

    required_model_columns = {
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
    }

    assert required_model_columns.issubset(completed.columns)
    assert "predicted_rate" not in completed.columns
    assert len(completed) == len(december_data)

    for column in (
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "market_index",
        "quote_signal",
    ):
        assert completed[column].notna().all()

    assert pd.to_datetime(completed["date"]).notna().all()


def test_existing_model_features_are_not_overwritten() -> None:
    """Existing market and quote values should be preserved."""
    training_data = _make_training_data(rows_count=60)

    december_data = _make_december_input(
        rows_count=1,
        include_predicted_rate=True,
    )

    december_data["market_index"] = 1.234
    december_data["quote_signal"] = 2.345

    completed = complete_december_model_input(
        december_data=december_data,
        training_data=training_data,
    )

    assert completed.loc[0, "market_index"] == 1.234
    assert completed.loc[0, "quote_signal"] == 2.345


def test_december_output_matches_required_schema(tmp_path) -> None:
    """December output must contain exactly seven scorer-compatible columns."""
    training_data = _make_training_data(rows_count=60)
    december_data = _make_december_input(rows_count=31)

    input_path = tmp_path / "december-input.csv"
    output_path = tmp_path / "december-predictions.csv"

    december_data.to_csv(input_path, index=False)

    create_december_predictions(
        model=FakeModel(prediction=825.0),
        input_path=str(input_path),
        output_path=str(output_path),
        training_data=training_data,
    )

    output = pd.read_csv(output_path)

    assert list(output.columns) == DECEMBER_OUTPUT_COLUMNS
    assert len(output) == 31

    # The original input fields must remain unchanged.
    assert output["pickup"].eq("Lexington").all()
    assert output["delivery"].eq("Fort Wayne").all()
    assert output["distance"].eq(360).all()
    assert output["equipment"].eq("Dry Van").all()
    assert output["weight"].eq(32000).all()

    assert output["predicted_rate"].notna().all()
    assert np.isfinite(output["predicted_rate"]).all()
    assert (output["predicted_rate"] >= 0).all()
    assert output["predicted_rate"].eq(825.0).all()


def test_december_output_preserves_input_row_order(tmp_path) -> None:
    """Prediction rows must preserve the order of the December input."""
    training_data = _make_training_data(rows_count=60)

    december_data = _make_december_input(rows_count=3)
    december_data["date"] = [
        "12/15/2025",
        "12/01/2025",
        "12/31/2025",
    ]

    input_path = tmp_path / "december-input.csv"
    output_path = tmp_path / "december-predictions.csv"

    december_data.to_csv(input_path, index=False)

    create_december_predictions(
        model=FakeModel(prediction=810.0),
        input_path=str(input_path),
        output_path=str(output_path),
        training_data=training_data,
    )

    output = pd.read_csv(output_path)

    actual_dates = pd.to_datetime(
        output["date"],
        format="mixed",
    ).dt.strftime("%m/%d/%Y").tolist()

    expected_dates = [
        "12/15/2025",
        "12/01/2025",
        "12/31/2025",
    ]

    assert actual_dates == expected_dates


def test_december_input_rejects_missing_required_column() -> None:
    """Invalid December input should fail with a clear error."""
    training_data = _make_training_data(rows_count=60)
    december_data = _make_december_input(rows_count=1)

    december_data = december_data.drop(columns=["equipment"])

    try:
        complete_december_model_input(
            december_data=december_data,
            training_data=training_data,
        )
    except ValueError as error:
        assert "missing required columns" in str(error).lower()
    else:
        raise AssertionError(
            "Expected ValueError for missing December input column."
        )
