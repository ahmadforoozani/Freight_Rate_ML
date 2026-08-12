"""Data cleaning and feature engineering utilities for freight-rate prediction."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FreightFeatureEngineer(BaseEstimator, TransformerMixin):
    """Clean raw freight data and derive robust temporal, spatial, and interaction features."""

    def __init__(self, id_column: str = "load_id", date_column: str = "date") -> None:
        self.id_column = id_column
        self.date_column = date_column

    def fit(self, data_frame: pd.DataFrame, y: Optional[pd.Series] = None) -> "FreightFeatureEngineer":
        """Fit the transformer; retained for scikit-learn pipeline compatibility."""
        return self

    def transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Return a clean feature matrix without identifiers and raw date values."""

        frame = data_frame.copy()

        # Convert common numeric columns safely.
        for column in ("distance", "weight", "market_index", "quote_signal"):
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

        # Preserve data-quality signals before replacing invalid values with NaN.
        # These flags help the model distinguish genuinely missing values from
        # values that were present but invalid in the source system.
        if "weight" in frame.columns:
            frame["has_invalid_weight"] = (
                frame["weight"] < 0).astype("float64")
            frame["has_zero_weight"] = (frame["weight"] == 0).astype("float64")
            frame.loc[frame["weight"] < 0, "weight"] = np.nan

        if "distance" in frame.columns:
            frame["has_invalid_distance"] = (
                frame["distance"] < 0).astype("float64")
            frame["has_zero_distance"] = (
                frame["distance"] == 0).astype("float64")
            frame.loc[frame["distance"] < 0, "distance"] = np.nan

        self._add_date_features(frame)
        self._add_geo_features(frame)
        self._add_interaction_features(frame)

        # Remove raw values that should not be used directly by the estimator.
        frame = frame.drop(
            columns=[self.id_column, self.date_column], errors="ignore")

        # Replace numeric infinity resulting from invalid calculations.
        frame = frame.replace([np.inf, -np.inf], np.nan)
        return frame

    def _add_date_features(self, frame: pd.DataFrame) -> None:
        """Add calendar and cyclical features from the load date."""
        if self.date_column not in frame.columns:
            return

        parsed_date = pd.to_datetime(frame[self.date_column], errors="coerce")
        frame["load_year"] = parsed_date.dt.year
        frame["load_month"] = parsed_date.dt.month
        frame["load_day"] = parsed_date.dt.day
        frame["load_day_of_week"] = parsed_date.dt.dayofweek
        frame["load_week_of_year"] = parsed_date.dt.isocalendar(
        ).week.astype("float64")
        frame["is_weekend"] = (parsed_date.dt.dayofweek >= 5).astype("float64")

        # Generic end-of-month features.
        frame["days_to_month_end"] = (
            parsed_date.dt.days_in_month - parsed_date.dt.day
        ).astype("float64")

        # frame["is_month_end"] = (
        #     frame["days_to_month_end"] == 0
        # ).astype("float64")

        # frame["is_last_2_days_of_month"] = (
        #     frame["days_to_month_end"] <= 1
        # ).astype("float64")

        # Cyclical encoding preserves the proximity of December and January.
        frame["month_sin"] = np.sin(2 * np.pi * frame["load_month"] / 12)
        frame["month_cos"] = np.cos(2 * np.pi * frame["load_month"] / 12)
        frame["weekday_sin"] = np.sin(
            2 * np.pi * frame["load_day_of_week"] / 7)
        frame["weekday_cos"] = np.cos(
            2 * np.pi * frame["load_day_of_week"] / 7)

    def _add_geo_features(self, frame: pd.DataFrame) -> None:
        """Calculate air distance and directional features when coordinates are available."""

        required_coords = ["pickup_lat", "pickup_lon",
                           "delivery_lat", "delivery_lon"]
        if not all(col in frame.columns for col in required_coords):
            return

        lat1 = pd.to_numeric(frame["pickup_lat"], errors="coerce")
        lon1 = pd.to_numeric(frame["pickup_lon"], errors="coerce")
        lat2 = pd.to_numeric(frame["delivery_lat"], errors="coerce")
        lon2 = pd.to_numeric(frame["delivery_lon"], errors="coerce")

        # Coordinates outside valid geographic bounds are invalid.
        lat1 = lat1.where(lat1.between(-90, 90))
        lat2 = lat2.where(lat2.between(-90, 90))
        lon1 = lon1.where(lon1.between(-180, 180))
        lon2 = lon2.where(lon2.between(-180, 180))

        frame["haversine_distance_miles"] = self._haversine(
            lat1, lon1, lat2, lon2)
        frame["latitude_delta"] = (lat2 - lat1).abs()
        frame["longitude_delta"] = (lon2 - lon1).abs()
        frame["pickup_latitude"] = lat1
        frame["pickup_longitude"] = lon1
        frame["delivery_latitude"] = lat2
        frame["delivery_longitude"] = lon2

        if "distance" in frame.columns:

            # A normal road-to-air ratio is generally above 1.0. Extreme values
            # may indicate an unusual route or a possible source-data issue.
            frame["road_to_air_distance_ratio"] = (
                frame["distance"] /
                frame["haversine_distance_miles"].replace(0, np.nan)
            )

            # Retain the existing coarse route classes.
            frame["is_short_haul"] = (frame["distance"] < 50).astype("float64")
            frame["is_medium_haul"] = (
                frame["distance"].between(50, 250)
            ).astype("float64")
            frame["is_long_haul"] = (frame["distance"] > 250).astype("float64")

            # Fine-grained thresholds are based on the error analysis:
            # prediction uncertainty rises notably for long-distance shipments.
            frame["is_over_500_miles"] = (
                frame["distance"] >= 500
            ).astype("float64")
            frame["is_over_1000_miles"] = (
                frame["distance"] >= 1000
            ).astype("float64")
            frame["is_over_1800_miles"] = (
                frame["distance"] >= 1800
            ).astype("float64")

            # Smooth nonlinear distance signal, useful for high-value long-haul loads.
            frame["distance_squared"] = frame["distance"] ** 2

        # if "pickup" in frame.columns and "delivery" in frame.columns:
        #     frame["route_id"] = frame["pickup"].astype(
        #         str) + "_" + frame["delivery"].astype(str)

    def _add_interaction_features(self, frame: pd.DataFrame) -> None:
        """Create domain-relevant feature interactions without using the target."""

        # 1. Base Interactions (Distance & Weight)
        if {"distance", "weight"}.issubset(frame.columns):
            frame["distance_x_weight"] = frame["distance"] * frame["weight"]
            frame["weight_per_distance"] = frame["weight"] / \
                frame["distance"].replace(0, np.nan)

            # Flag for heavy loads to help model differentiate high-margin/heavy tiers
            frame["is_heavy_load"] = (
                frame["weight"] > 35000).astype("float64")

        # 2. Market & Signal Interactions
        if {"market_index", "distance"}.issubset(frame.columns):
            frame["market_x_distance"] = frame["market_index"] * \
                frame["distance"]

        if {"market_index", "quote_signal"}.issubset(frame.columns):
            frame["market_x_quote_signal"] = frame["market_index"] * \
                frame["quote_signal"]
            # Modeling non-linear market pressure
            frame["market_strain"] = (
                frame["market_index"] * frame["quote_signal"]) ** 2

        # 3. Log transformations for skewed distributions
        if "distance" in frame.columns:
            frame["log_distance"] = np.log1p(frame["distance"].clip(lower=0))
        if "weight" in frame.columns:
            frame["log_weight"] = np.log1p(frame["weight"].clip(lower=0))

        # 4. Long Haul Interactions
        if {"market_index", "distance"}.issubset(frame.columns):
            frame["market_x_long_haul"] = (
                frame["market_index"] * (frame["distance"] >= 1000).astype("float64"))
            frame["market_x_very_long_haul"] = (
                frame["market_index"] * (frame["distance"] >= 1800).astype("float64"))

        # 5. Equipment-Specific Interactions
        if "equipment" in frame.columns:
            equipment_text = frame["equipment"].astype("string").str.lower()

            frame["is_reefer"] = equipment_text.str.contains(
                r"reefer|refrigerated", regex=True, na=False).astype("float64")
            frame["is_dry_van"] = equipment_text.str.contains(
                r"dry[\s-]*van", regex=True, na=False).astype("float64")
            frame["is_flatbed"] = equipment_text.str.contains(
                r"flat[\s-]*bed", regex=True, na=False).astype("float64")

            # Enhanced interactions for specialized equipment
            if "distance" in frame.columns:
                frame["reefer_x_distance"] = frame["is_reefer"] * \
                    frame["distance"]
                frame["reefer_x_very_long_haul"] = frame["is_reefer"] * \
                    (frame["distance"] >= 1800).astype("float64")

            if {"market_strain", "is_reefer"}.issubset(frame.columns):
                # Interaction between high market demand and specialized equipment cost
                frame["reefer_x_market_strain"] = frame["is_reefer"] * \
                    frame["market_strain"]

            if {"weight_per_distance", "is_reefer"}.issubset(frame.columns):
                # Cost density for refrigerated heavy loads
                frame["reefer_x_weight_per_distance"] = frame["is_reefer"] * \
                    frame["weight_per_distance"]

    @staticmethod
    def _haversine(
        latitude_1: pd.Series,
        longitude_1: pd.Series,
        latitude_2: pd.Series,
        longitude_2: pd.Series,
    ) -> pd.Series:
        """Calculate great-circle distance in Miles between two coordinate pairs."""
        earth_radius_km = 6371.0088

        latitude_1 = np.radians(latitude_1)
        longitude_1 = np.radians(longitude_1)
        latitude_2 = np.radians(latitude_2)
        longitude_2 = np.radians(longitude_2)

        latitude_delta = latitude_2 - latitude_1
        longitude_delta = longitude_2 - longitude_1

        haversine_term = (
            np.sin(latitude_delta / 2) ** 2
            + np.cos(latitude_1) * np.cos(latitude_2) *
            np.sin(longitude_delta / 2) ** 2
        )
        dist_km = 2 * earth_radius_km * \
            np.arcsin(np.sqrt(haversine_term.clip(0, 1)))
        dist_miles = dist_km * 0.621371
        return dist_miles
