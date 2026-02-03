"""
Model loading and prediction service.
Handles:
- Model loading
- Feature alignment
- Prediction
- Confidence intervals
"""

from __future__ import annotations
import pickle
import json
import pandas as pd
from pathlib import Path


class PredictionService:
    """
    Service for loading model and making predictions.
    """

    def __init__(
        self,
        model_path: str = "models/revenue_model.pkl",
        feature_info_path: str = "models/feature_info.json",
        ci_stats_path: str = "models/training_stats.json",
    ):
        self.model = None
        self.feature_info = None
        self.feature_names = None
        self.residual_std = None

        self.model_path = Path(model_path)
        self.feature_info_path = Path(feature_info_path)
        self.ci_stats_path = Path(ci_stats_path)

        self.load_model()
        self.load_feature_info()
        self.load_ci_stats()

    
    # LOADERS

    def load_model(self):
        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            print(f"✅ Model loaded from {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def load_feature_info(self):
        try:
            with open(self.feature_info_path, "r") as f:
                self.feature_info = json.load(f)

            self.feature_names = self.feature_info["feature_names"]
            print(f"✅ Feature info loaded from {self.feature_info_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load feature info: {e}")

    def load_ci_stats(self):
        try:
            with open(self.ci_stats_path, "r") as f:
                stats = json.load(f)

            self.residual_std = float(stats["residual_std"])
            print(f"✅ CI stats loaded from {self.ci_stats_path}")
        except Exception:
            self.residual_std = None
            print("⚠️ CI stats not found — CI disabled")

    # PREDICTION

    def _prepare_dataframe(self, input_data: dict) -> pd.DataFrame:
        """
        Align user input with training features.
        Missing features are filled with 0.
        """
        df = pd.DataFrame([input_data])

        missing_features = set(self.feature_names) - set(df.columns)
        for feature in missing_features:
            df[feature] = 0.0

        return df[self.feature_names]

    def predict(self, input_data: dict) -> float:
        df = self._prepare_dataframe(input_data)
        prediction = self.model.predict(df)[0]
        return float(prediction)

    def predict_with_ci(self, input_data: dict, confidence: float = 0.95):
        prediction = self.predict(input_data)

        if self.residual_std is None:
            return prediction, prediction, prediction

        z = 1.96  # 95%
        margin = z * self.residual_std

        return (
            float(prediction),
            float(prediction - margin),
            float(prediction + margin),
        )
        
    # METADATA & EXPLANATIONS

    def get_model_info(self) -> dict:
        return {
            "model_name": self.feature_info.get("model_name", "Linear Regression"),
            "metrics": self.feature_info.get("metrics", {}),
            "feature_count": len(self.feature_names),
            "features": self.feature_names,
        }

    def get_feature_explanations(self) -> dict:
        """
        Human-readable explanations for frontend usage.
        """
        return {
            "transaction_qty": "Number of products sold during the day",
            "unit_price": "Average selling price per product",
            "revenue_lag1": "Revenue from the previous day",
            "revenue_lag7": "Revenue from 7 days ago, same weekday",
            "revenue_rolling3": "Average revenue over the last 3 days",
            "revenue_rolling7": "Average revenue over the last 7 days",
            "month": "Month of the year (1–12)",
            "day": "Day of the month",
            "day_of_week": "Day of the week (0=Monday)",
            "is_weekend": "1 if weekend, else 0",
        }


# SINGLETON

_prediction_service: PredictionService | None = None


def get_prediction_service() -> PredictionService:
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
    return _prediction_service
