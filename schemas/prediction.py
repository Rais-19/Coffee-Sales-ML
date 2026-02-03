"""
Pydantic schemas for API request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional,List


class PredictionInput(BaseModel):
    
    transaction_qty: int = Field(..., ge=1, description="Number of items in transaction")
    unit_price: float = Field(..., gt=0, description="Average unit price")
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    day: int = Field(..., ge=1, le=31, description="Day of month")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday)")
    is_weekend: int = Field(..., ge=0, le=1, description="Weekend flag (0 or 1)")
    revenue_lag1: float = Field(..., description="Revenue from previous day")
    revenue_lag7: float = Field(..., description="Revenue from 7 days ago")
    revenue_rolling3: float = Field(..., description="3-day rolling average revenue")
    revenue_rolling7: float = Field(..., description="7-day rolling average revenue")
    
    # Store location (one-hot encoded) - adjust based on your model
    store_location_Hells_Kitchen: Optional[int] = Field(0, ge=0, le=1)
    store_location_Lower_Manhattan: Optional[int] = Field(0, ge=0, le=1)
    




    
class Config:
    json_schema_extra = {
        "predicted_revenue": 3520.75,
        "model_name": "XGBoost"
    }


class ConfidenceInterval(BaseModel):
    lower: float = Field(..., description="Lower bound of confidence interval")
    upper: float = Field(..., description="Upper bound of confidence interval")
    confidence_level: str = Field(..., description="Confidence level (e.g., 95%)")
    
    
class PredictionOutput(BaseModel):
    """ output schema for revenue prediction"""
    success: bool
    predicted_revenue: float
    confidence_interval: ConfidenceInterval
    model_name: str
    timestamp: str
class ErrorResponse(BaseModel):
    """
    Error response schema.
    """
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    
from typing import List

class BatchPredictionInput(BaseModel):
    predictions: List[PredictionInput] = Field(
        ..., 
        min_items=1, 
        max_items=100,
        description="List of prediction inputs (max 100)"
    )


class BatchPredictionResult(BaseModel):
    index: int
    predicted_revenue: float
    confidence_interval: ConfidenceInterval
    success: bool
    error: str | None

class BatchPredictionOutput(BaseModel):
    success: bool
    total_predictions: int
    successful_predictions: int
    failed_predictions: int
    results: List[BatchPredictionResult]
    model_name: str
    timestamp: str
class ModelInfo(BaseModel):
    model_name: str
    metrics: dict
    feature_count: int
    features: list
class FeatureInfo(BaseModel):
    name: str
    description: str
    type: str
class FeaturesResponse(BaseModel):
    features: list
    total_features: int
    feature_groups: dict

