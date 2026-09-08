import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

st.set_page_config(page_title="Shipping Delay Predictor", page_icon="📦", layout="centered")

st.title("📦 Shipping Delay Predictor")
st.markdown("Enter shipment details to predict if a delivery will be delayed.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        warehouse_block = st.selectbox("Warehouse Block", ["A", "B", "C", "D", "F"])
        mode_of_shipment = st.selectbox("Mode of Shipment", ["Flight", "Ship", "Road"])
        customer_care_calls = st.number_input("Customer Care Calls", min_value=0, value=4, step=1)
        customer_rating = st.slider("Customer Rating", min_value=1, max_value=5, value=4)
        cost_of_product = st.number_input("Cost of Product ($)", min_value=1.0, value=100.0, step=1.0)

    with col2:
        prior_purchases = st.number_input("Prior Purchases", min_value=0, value=3, step=1)
        product_importance = st.selectbox("Product Importance", ["low", "medium", "high"])
        discount_offered = st.number_input("Discount Offered (%)", min_value=0.0, max_value=99.0, value=10.0, step=1.0)
        weight_in_gms = st.number_input("Weight (grams)", min_value=1.0, value=500.0, step=10.0)
        gender = st.selectbox("Gender", ["M", "F"])

    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    payload = {
        "Warehouse_block": warehouse_block,
        "Mode_of_Shipment": mode_of_shipment,
        "Customer_care_calls": customer_care_calls,
        "Customer_rating": customer_rating,
        "Cost_of_the_Product": cost_of_product,
        "Prior_purchases": prior_purchases,
        "Product_importance": product_importance,
        "Discount_offered": discount_offered,
        "Weight_in_gms": weight_in_gms,
        "Gender": gender,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json()
            prediction = result["prediction"]
            probability = result["delay_probability"]

            if prediction == "Delayed":
                st.error(f"⚠️ Prediction: **{prediction}**")
            else:
                st.success(f"✅ Prediction: **{prediction}**")

            st.metric("Delay Probability", f"{probability:.2%}")
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the API. Make sure the backend is running at " + API_URL)
