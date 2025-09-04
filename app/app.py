import streamlit as st
import requests

st.set_page_config(page_title="ForecastX", page_icon="📊", layout="centered")

# Title
st.title("ForecastX")

# Subtitle
st.subheader("Company Sales Prediction App")

# Input fields with placeholders
priceeach = st.number_input("Product Price($)", min_value=0.0, placeholder=50.0)
quantityordered = st.number_input("Quantity Ordered", min_value=1, placeholder=10)
productline = st.text_input("Product Line", placeholder="Motorcycles")
productcode = st.text_input("Product Code", placeholder="S10_1949")
customername = st.text_input("Customer Name", placeholder="Alpha Cognac")
country = st.text_input("Country", placeholder="USA")

st.info(
    "Notice: The supervised learning models used in this app were trained on a particular company's data. "
    "Prediction accuracy will be best for that company's activities. "
    "We can retrain the model with your company's historical sales CSV containing the fields above to predict your own sales.")

# Predict button
if st.button("Predict"):
    data = {
        "priceeach": priceeach,
        "quantityordered": quantityordered,
        "productline": productline,
        "productcode": productcode,
        "customername": customername,
        "country": country
    }

    # Call API
    try:
        response = requests.post("http://localhost:8002/predict", json=data)

        # Check for a successful response (status code 200)
        if response.status_code == 200:
            prediction_data = response.json()
            # **UPDATED LINE**
            # Access the key 'sales_prediction($)' from the API response
            prediction = prediction_data.get('sales_prediction($)')
            if prediction is not None:
                st.success(f"Predicted Sales: ${prediction:.2f}")
            else:
                st.error("Prediction failed: 'sales_prediction($)' key not found in the API response.")
        else:
            # Handle API errors with other status codes
            try:
                error_details = response.json().get('detail', 'Unknown error')
                st.error(f"Prediction failed: {error_details}")
            except requests.exceptions.JSONDecodeError:
                st.error(f"Prediction failed with status code {response.status_code}. Server returned a non-JSON response.")

    except requests.exceptions.ConnectionError:
        st.error("API call error: Connection to the backend server failed. Is the server running?")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# Footer
st.markdown("<hr><p style='text-align:center; color:gray;'>Developed by Tolulope Oyejide AI/ML Labs</p>",unsafe_allow_html=True)