import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import base64

# ===================================
# PAGE CONFIG
# ===================================
st.set_page_config(
    page_title="Bike Sharing Prediction",
    page_icon="🚲",
    layout="centered"
)

# ===================================
# LOAD CSS
# ===================================
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ===================================
# LOAD MODEL + SCALER
# ===================================
model = joblib.load("gradient_boosting_model.pkl")
scaler = joblib.load("scaler.pkl")

# ===================================
# SESSION STATE
# ===================================
if 'selected_year' not in st.session_state:
    st.session_state.selected_year = 2012

if 'selected_month' not in st.session_state:
    st.session_state.selected_month = 6

# ===================================
# HEADER IMAGE (BIKE LOGO)
# ===================================
with open("static/bike.png", "rb") as f:
    bike_b64 = base64.b64encode(f.read()).decode()

st.markdown(
    f"""
    <div class='title-container'>
        <img src="data:image/png;base64,{bike_b64}" class="header-banner-img"/>
    </div>
    """,
    unsafe_allow_html=True
)


# ===================================
# YEAR / MONTH SELECTORS
# ===================================
col1, col2 = st.columns(2)

with col1:
    yr = st.selectbox("Year", list(range(2011, 2032)), index=1)
    st.session_state.selected_year = yr

with col2:
    mnth = st.selectbox("Month", list(range(1, 13)), index=5)
    st.session_state.selected_month = mnth

# number of days per month
days_in_month = {
    1: 31, 2: 28, 3: 31, 4: 30,
    5: 31, 6: 30, 7: 31, 8: 31,
    9: 30, 10: 31, 11: 30, 12: 31
}

# leap year correction
if mnth == 2 and (
    (yr % 4 == 0 and yr % 100 != 0) or (yr % 400 == 0)
):
    days_in_month[2] = 29

# ===================================
# FORM
# ===================================
with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        season = st.selectbox(
            "Season",
            [1, 2, 3, 4],
            format_func=lambda x: {
                1: "Spring",
                2: "Summer",
                3: "Fall",
                4: "Winter"
            }[x]
        )

    with col2:
        day = st.selectbox(
            "Day",
            list(range(1, days_in_month[mnth] + 1))
        )

    col3, col4, col5 = st.columns(3)

    with col3:
        hr = st.slider("Hour", 0, 23, 12)

    with col4:
        holiday = st.selectbox("Holiday", [0, 1], format_func=lambda x: "Yes" if x else "No")

    with col5:
        weekday = st.selectbox(
            "Weekday",
            list(range(7)),
            format_func=lambda x: ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"][x]
        )

    col6, col7 = st.columns(2)

    with col6:
        workingday = st.selectbox("Working Day", [0, 1], format_func=lambda x: "Yes" if x else "No")

    with col7:
        weathersit = st.selectbox(
            "Weather Situation",
            [1, 2, 3, 4],
            format_func=lambda x: {
                1: "Clear / Partly Cloudy",
                2: "Mist / Cloudy",
                3: "Light Snow / Light Rain",
                4: "Heavy Rain / Ice / Snow + Fog"
            }[x]
        )
    col8, col9 = st.columns(2)

    with col8:
        temp = st.number_input("Temperature (°C)", value=20.0)

    with col9:
        atemp = st.number_input("Feeling Temperature (°C)", value=20.0)

    col10, col11 = st.columns(2)

    with col10:
        hum = st.number_input("Humidity (%)", value=50.0)

    with col11:
        windspeed = st.number_input("Wind Speed (km/h)", value=10.0)

    # convert year
    yr_model = yr - 2011

    predict_button = st.form_submit_button("Predict Bike Count")

# ===================================
# PREDICTION LOGIC
# ===================================
if predict_button:

    # input dataframe
    input_data = pd.DataFrame({
        'season': [season],
        'yr': [yr_model],
        'mnth': [mnth],
        'hr': [hr],
        'holiday': [holiday],
        'weekday': [weekday],
        'workingday': [workingday],
        'weathersit': [weathersit],
        'temp': [temp],
        'atemp': [atemp],
        'hum': [hum],
        'windspeed': [windspeed],
        'day': [day]
    })

    # scaling (IMPORTANT)
    scale_cols = ['temp', 'atemp', 'hum', 'windspeed']
    input_data[scale_cols] = scaler.transform(input_data[scale_cols])

    # 1. Get prediction in LOG scale
    prediction_log = model.predict(input_data)[0]
    
    # 2. CONVERT BACK TO REAL COUNT 
    # This turns the log value back into a positive bike count
    prediction = np.expm1(prediction_log)
    # 3. Apply the safety check and rounding
    # This ensures 0.0001 becomes 0 and anything weird stays at 0
    display_number = int(max(0, round(prediction)))
    # result UI
    st.markdown(
        f"""
        <div class='prediction-result-container'>
            <div class='result-label'>Estimated Rentals</div>
            <div class='result-number'>{display_number}</div>
            <div class='result-unit'>bikes / hour</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.write("")
    # optional insight
    st.info("Prediction generated using Gradient Boosting Regressor trained on UCI Bike Sharing dataset")