import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load models
dtr = pickle.load(open('models/dtr_model.pkl', 'rb'))
preprocessor = pickle.load(open('models/preprocesser.pkl', 'rb'))

# Page configuration
st.set_page_config(
    page_title="Crop Yield Prediction Dashboard",
    page_icon="🌾",
    layout="wide"
)

# Custom UI styling
st.markdown("""
<style>
.stApp {
    background-color: #f4f9f4;
}

h1 {
    color: #2e7d32;
}

.sidebar .sidebar-content {
    background-color: #e8f5e9;
}

.stButton>button {
    background-color: #2e7d32;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}

.result-box {
    background-color: #e8f5e9;
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    color: #1b5e20;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("🌾 Crop Yield Prediction Dashboard")

st.markdown("""
This machine learning dashboard predicts **agricultural crop yield** using environmental and farming inputs like rainfall, pesticide usage, temperature, crop type and region.
""")

# Sidebar inputs
st.sidebar.header("🌱 Enter Farming Inputs")

year = st.sidebar.number_input("Year", 1900, 2100, 2024)

average_rainfall = st.sidebar.number_input(
    "Average Rainfall (mm/year)", 0.0, 5000.0, 1000.0
)

pesticides_tonnes = st.sidebar.number_input(
    "Pesticides Used (tonnes)", 0.0, 1000.0, 50.0
)

avg_temp = st.sidebar.number_input(
    "Average Temperature (°C)", -50.0, 60.0, 25.0
)

area = st.sidebar.selectbox("Area", [
    "India","Brazil","USA","China","Australia","Argentina",
    "Canada","France","Germany","Pakistan","Thailand","Spain"
])

item = st.sidebar.selectbox("Crop Type", [
    "Maize","Rice, paddy","Wheat","Potatoes",
    "Soybeans","Cassava","Sweet potatoes"
])

# Dashboard metrics
st.subheader("📊 Current Input Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Rainfall", f"{average_rainfall} mm")
col2.metric("Temperature", f"{avg_temp} °C")
col3.metric("Pesticides", f"{pesticides_tonnes} t")

# Prediction button
if st.sidebar.button("Predict Crop Yield 🌾"):

    features = pd.DataFrame(
        [[year, average_rainfall, pesticides_tonnes, avg_temp, area, item]],
        columns=[
            'Year',
            'average_rain_fall_mm_per_year',
            'pesticides_tonnes',
            'avg_temp',
            'Area',
            'Item'
        ]
    )

    transformed_features = preprocessor.transform(features)

    prediction = dtr.predict(transformed_features)

    # Result display
    st.markdown("### 🌱 Predicted Agricultural Yield")

    st.markdown(
        f'<div class="result-box">{prediction[0]:.2f} Tonnes</div>',
        unsafe_allow_html=True
    )

    # Simple visualization
    st.subheader("📈 Yield Prediction Trend")

    values = [prediction[0]-2, prediction[0]-1, prediction[0]]

    years = ["Past Estimate", "Recent Estimate", "Predicted"]

    fig, ax = plt.subplots()

    ax.plot(years, values, marker="o")

    ax.set_ylabel("Yield (Tonnes)")

    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("👩‍💻 Developed by **Rishita V Patil** | Crop Yield Prediction ML Project")