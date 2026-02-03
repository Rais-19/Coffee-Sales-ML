Coffee Sales Revenue Prediction
Project Overview

This project is an end-to-end machine learning system that predicts daily coffee shop revenue using historical sales patterns and business features.
It covers the full ML lifecycle: exploratory data analysis, feature engineering, model training, API deployment, and a user-friendly frontend.

The system is designed as a portfolio-ready example of how a data science model is transformed into a usable product.

Objectives
Primary Goal

Predict daily revenue for a coffee shop using structured input features, while providing a confidence interval to quantify prediction uncertainty.

Secondary Goals

Demonstrate clean ML engineering practices

Expose the model through a production-style API

Provide a simple, understandable frontend for non-technical users

Exploratory Data Analysis (EDA) – Final Goal

The EDA phase aimed to:

Understand revenue behavior over time

Identify seasonality, trends, and weekly patterns

Measure the impact of:

Transaction volume

Unit price

Store location

Calendar effects (weekday vs weekend)

Justify the creation of:

Lag features (previous revenue values)

Rolling averages (short-term trends)

The outcome of EDA directly informed feature selection and feature engineering, ensuring that the model captures real business dynamics rather than noise.

Feature Engineering

Key engineered features include:

Calendar features (month, day, weekday, weekend flag)

Lagged revenue values (1-day and 7-day)

Rolling averages (3-day and 7-day)

One-hot encoded store locations

Lag and rolling features are explicitly provided by the user to keep the system stateless and transparent.

Model

Algorithm: Linear Regression

Reasoning:

High interpretability

Stable behavior for business forecasting

Easy explanation to non-technical stakeholders

Evaluation metrics stored during training:

R² score

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

Confidence Intervals

The API returns a 95% confidence interval, computed using the standard deviation of training residuals.

Purpose:

Communicate uncertainty

Prevent overconfidence in point predictions

Align predictions with real-world decision-making
Future Improvements

Planned extensions for future versions:

Automatic lag & rolling feature computation using stored history

Time-series models (ARIMA, XGBoost, LSTM)

Multi-store forecasting dashboard

Model retraining pipeline

Authentication and rate limiting

Deployment to cloud (Docker + AWS/GCP)