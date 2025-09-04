import streamlit as st
import requests

st.set_page_config(page_title="Sales Prediction", page_icon="📊", layout="centered")
st.title("📈 Sales Prediction App")

# API URL
API_URL = "http://127.0.0.1:8002/docs#/default"  # Replace with your deployed FastAPI URL

# Input fields
st.header("Enter Your Company Sales Data")
priceeach = st.number_input("Price Each", min_value=0.0, step=0.01)
quantityordered = st.number_input("Quantity Ordered", min_value=1, step=1)
productline = st.text_input("Product Line")
productcode = st.text_input("Product Code")
customername = st.text_input("Customer Name")
country = st.text_input("Country")

# Prediction
if st.button("Predict Sales"):
    if all([productline, productcode, customername, country]):
        payload = {
            "priceeach": priceeach,
            "quantityordered": int(quantityordered),
            "productline": productline,
            "productcode": productcode,
            "customername": customername,
            "country": country
        }
        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            prediction = response.json().get("prediction")
            st.success(f"Predicted Sales: {prediction:.2f}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please fill in all fields.")
