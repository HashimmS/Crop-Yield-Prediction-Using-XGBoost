import streamlit as st
import pandas as pd
import joblib  # For loading trained ML models
import numpy as np

# Load the trained model (Ensure 'crop_yield_model.pkl' exists in the same directory)
@st.cache_resource
def load_model():
    return joblib.load("crop_yield_model.pkl")

model = load_model()

# Streamlit UI
st.title("🌾 Crop Yield Prediction App")
st.write("Enter the required details to predict the crop yield.")

# Sidebar inputs
st.sidebar.header("Enter Inputs for Prediction")
year = st.sidebar.number_input("Year", min_value=2000, max_value=2030, step=1)
rainfall = st.sidebar.number_input("Average Rainfall (mm/year)", min_value=0.0)
pesticides = st.sidebar.number_input("Pesticides (tonnes)", min_value=0.0)
temperature = st.sidebar.number_input("Average Temperature (°C)", min_value=-10.0, max_value=50.0)

# Prediction button
if st.sidebar.button("Predict Yield"):
    input_data = pd.DataFrame({
        "Country_Encoded": [0],  # Placeholder, update if needed
        "Item_Encoded": [0],     # Placeholder, update if needed
        "Pesticides": [pesticides],
        "Avg_Temp": [temperature],
        "Rainfall": [rainfall]
    })

    # Make prediction
    prediction = model.predict(input_data)

    # Display result
    st.success(f"🌱 Predicted Crop Yield: **{prediction[0]:.2f} hg/ha**")

# Optional: Add dataset upload option
uploaded_file = st.file_uploader("Upload a CSV file for batch predictions", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Dataset Preview:")
    st.write(data.head())
    
    # Ensure dataset has required columns
    required_columns = ["Year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]
    if all(col in data.columns for col in required_columns):
        # Rename columns to match trained model
        data = data.rename(columns={
            "pesticides_tonnes": "Pesticides",
            "avg_temp": "Avg_Temp",
            "average_rain_fall_mm_per_year": "Rainfall"
        })

        # Add placeholders for missing categorical features if needed
        data["Country_Encoded"] = 0  # Placeholder (update if necessary)
        data["Item_Encoded"] = 0  # Placeholder (update if necessary)

        # Reorder columns to match the trained model's feature order
        feature_order = ["Country_Encoded", "Item_Encoded", "Pesticides", "Avg_Temp", "Rainfall"]
        data = data[feature_order]

        # Make predictions
        predictions = model.predict(data)
        data["Predicted_Yield"] = predictions
        st.write("### Predictions:")
        st.write(data)
    else:
        st.error("Uploaded CSV must contain columns: Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp")

# Footer
st.write("Developed for Crop Yield Prediction using XGBoost 🚀")
