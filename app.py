"""
FastAPI application for coffee sales revenue prediction.

Design choice:
- Advanced features (lags & rolling) are provided by the user
- API validates, aligns, predicts, and explains results
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from services.prediction_service import get_prediction_service
from schemas.prediction import PredictionInput, PredictionOutput


app = FastAPI(
    title="Coffee Sales Revenue Prediction API",
    description="Revenue prediction using Linear Regression with confidence intervals",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

prediction_service = None


# STARTUP


@app.on_event("startup")
async def startup_event():
    global prediction_service
    prediction_service = get_prediction_service()
    print("✅ Prediction service ready")


# HEALTH

@app.get("/")
async def health():
    return {
        "status": "healthy",
        "model_loaded": prediction_service is not None,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# PREDICTION

@app.post("/predict", response_model=PredictionOutput)
def predict(user_input: PredictionInput):
    if prediction_service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    input_dict = user_input.model_dump()

    pred, lower, upper = prediction_service.predict_with_ci(input_dict)

    return {
        "success": True,
        "predicted_revenue": pred,
        "confidence_interval": {
            "lower": lower,
            "upper": upper,
            "confidence_level": "95%",
        },
        "explanations": {
            "prediction_meaning": (
                "This is the estimated daily revenue based on the provided inputs."
            ),
            "confidence_interval": (
                "There is a 95% probability that the true revenue lies within this range."
            ),
            "important_note": (
                "Lag and rolling features were provided by the user to reflect historical trends."
            ),
        },
        "model_name": prediction_service.get_model_info()["model_name"],
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# FEATURE EXPLANATIONS

@app.get("/explain/features")
def explain_features():
    if prediction_service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "feature_explanations": prediction_service.get_feature_explanations()
    }
