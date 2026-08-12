# Freight Rate Prediction

An end-to-end machine-learning pipeline for predicting freight `posted_rate` values from shipment, geographic, temporal, equipment, and market data.

The project trains a `HistGradientBoostingRegressor` pipeline, evaluates it using a time-aware validation split, performs residual and `quote_signal` ablation analysis, persists the final model artifact, and generates prediction files for validation and December inputs.

## Main Capabilities

- Time-aware validation to reduce future-data leakage.
- Feature engineering for temporal, geographic, route, distance, equipment, and market signals.
- Log-transformed target modeling through `TransformedTargetRegressor`.
- Configurable `squared_error` and `absolute_error` regression losses.
- Optional target-emphasis sample weighting for high-value loads.
- Residual analysis for worst prediction errors.
- Ablation test for `quote_signal` dependency.
- Hierarchical fallback strategy for December records with missing model features.
- Model persistence using `joblib`.
- CSV outputs compatible with validation templates and December scoring format.


data_path: dataset/validation-predictions.csv

## Requirements
Python 3.11+
numpy>=1.24,<3.0
pandas>=2.0,<3.0
scikit-learn>=1.4,<2.0
joblib>=1.3,<2.0

## One-Command Run with the default paths:
    python -m app.app

    pytest -v test/test_pipeline.py


## Installation
    pip install -r requirements.txt

## A typical explicit run is
python app.app \
  --train data/train-test.csv \
  --validation data/validation.csv \
  --validation-template data/validation-predictions-template.csv \
  --december-input data/december-chart-inputs.csv \
  --model-output artifacts/freight_rate_model.joblib \
  --validation-output data/validation-predictions.csv \
  --december-output data/december-predictions.csv

## Command-Line Options

Argument	           Default	                    Description
--train	                data/train-test.csv	        Training dataset path.
--validation	        data/validation.csv	        Validation input dataset path.
--validation-template	data/validation-predictions-template.csv	Template used for validation prediction output.
--december-input	    data/december-chart-inputs.csv	December input file.
--model-output	        artifacts/freight_rate_model.joblib	    Path for the persisted model artifact.
--validation-output	    data/validation-predictions.csv	    Validation prediction output path.
--december-output	    data/december-chart-inputs.csv	    December prediction output path.
--regressor-loss	    squared_error	                Regressor loss: squared_error or absolute_error.
--sample-weight-strategy	none	Weighting strategy: none or target_emphasis.
--outlier-quantile	        0.995	Upper target quantile used to remove extreme training-target outliers.
--top-error-pct	            0.05	Fraction of worst validation errors included in residual analysis.
--skip-leakage-test	        disabled	Skip the quote_signal ablation test.


## Generated Outputs

After a successful run, the pipeline produces:

artifacts/freight_rate_model.joblib
data/validation-predictions.csv
data/december-predictions.csv


## Important Note About December Output

By default, both --december-input and --december-output point to:
data/december-chart-inputs.csv

This means the default run overwrites the December input file with completed predictions. 

## Example Experiment

Run the model with absolute-error loss and target-emphasis weighting:

python app/main.py \
  --regressor-loss absolute_error \
  --sample-weight-strategy target_emphasis \
  --december-output data/december-predictions-absolute.csv

Skip the quote_signal ablation test for a faster run:

python app/main.py \
  --skip-leakage-test \
  --december-output data/december-predictions.csv

