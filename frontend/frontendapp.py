import streamlit as st
import requests
from datetime import date
import os
# CONFIG

API_URL = os.getenv("API_URL", "https://coffee-sales-ml.onrender.com/")
st.set_page_config(
    page_title="Coffee Revenue Predictor ",
    page_icon="",
    layout="centered"
)

# HEADER

st.title(" Coffee Shop Revenue Predictor")
st.markdown(
    """
    Predict **daily revenue** using historical trends and pricing data.

    This tool uses a **machine learning model** trained on real coffee shop sales.
    """
)

st.divider()

# SIDEBAR – EDUCATION

with st.sidebar:
    st.header(" How this works")

    st.markdown(
        """
        ###  What does the model use?

        The model looks at:
        - **Sales today**
        - **Prices**
        - **Recent revenue trends** (lags & rolling averages)

        ###  Why lag & rolling features?

        These help the model understand:
        - Momentum 
        - Seasonality 
        - Short-term trends 
        """
    )

    st.markdown("---")

    st.markdown(
        """
        ###  Important note

        Lag & rolling values must come from **your own records**.
        If you don’t have history yet, you may start with **0**.
        """
    )

# INPUT FORM
st.subheader(" Enter Daily Information")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        transaction_qty = st.number_input(
            " Number of items sold",
            min_value=0,
            value=50,
            help="Total products sold today"
        )

        unit_price = st.number_input(
            " Average unit price",
            min_value=0.0,
            value=4.5,
            step=0.1,
            help="Average price per item"
        )

    with col2:
        store_location = st.selectbox(
            " Store location",
            ["Hells_Kitchen", "Lower_Manhattan"]
        )

        selected_date = st.date_input(
            " Date",
            value=date.today()
        )

    st.divider()
    st.subheader(" Historical Revenue (Advanced)")

    st.caption("These values describe **past performance**. If unsure, start with 0.")

    col3, col4 = st.columns(2)

    with col3:
        revenue_lag1 = st.number_input(
            "Revenue yesterday (lag 1)",
            value=0.0,
            help="Revenue from the previous day"
        )

        revenue_rolling3 = st.number_input(
            "Avg revenue (last 3 days)",
            value=0.0,
            help="Average revenue of last 3 days"
        )

    with col4:
        revenue_lag7 = st.number_input(
            "Revenue 7 days ago (lag 7)",
            value=0.0,
            help="Revenue from same day one week ago(if today is monday: revenue of last monday)"
        )

        revenue_rolling7 = st.number_input(
            "Avg revenue (last 7 days)",
            value=0.0,
            help="Average of last 7 days"
        )

    submit = st.form_submit_button(" Predict Revenue")

# PREDICTION

if submit:
    payload = {
        "transaction_qty": transaction_qty,
        "unit_price": unit_price,
        "month": selected_date.month,
        "day": selected_date.day,
        "day_of_week": selected_date.weekday(),
        "is_weekend": int(selected_date.weekday() >= 5),
        "revenue_lag1": revenue_lag1,
        "revenue_lag7": revenue_lag7,
        "revenue_rolling3": revenue_rolling3,
        "revenue_rolling7": revenue_rolling7,
        "store_location_Hells_Kitchen": int(store_location == "Hells_Kitchen"),
        "store_location_Lower_Manhattan": int(store_location == "Lower_Manhattan"),
    }

    with st.spinner(" Predicting revenue..."):
        response = requests.post(f"{API_URL}/predict", json=payload)

    if response.status_code == 200:
        result = response.json()

        st.success(" Prediction successful")

        st.metric(
            label="Estimated Revenue",
            value=f"${result['predicted_revenue']:.2f}"
        )

        ci = result["confidence_interval"]

        st.markdown(
            f"""
            ### Confidence Interval (95%)

            The true revenue is likely between:

            **${ci['lower']:.2f} — ${ci['upper']:.2f}**
            """
        )

        st.info(
            """
             **How to interpret this**

            - The main value is the **best estimate**
            - The interval shows **uncertainty**
            - Wider intervals mean more variability in past data
            """
        )

    else:
        st.error(" Prediction failed")
        st.code(response.text)

