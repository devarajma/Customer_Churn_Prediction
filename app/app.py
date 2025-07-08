
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
# from streamlit_extras.metric_cards import style_metric_cards

# Load model and dependencies
model = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")
scaler = joblib.load("scaler.pkl")

# App configuration
st.set_page_config(page_title="Telco Churn Predictor", layout="wide")
st.title("📞 Telco Customer Churn Prediction")
st.markdown("""
<style>
    .big-font {
        font-size:20px !important;
        font-weight: 500;
    }
    .stButton > button {
        background-color: #ff4b4b;
        color: white;
        padding: 0.75em 2em;
        border-radius: 8px;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3771/3771423.png", width=120)
st.sidebar.header("Customer Profile")

def get_user_input():
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.sidebar.selectbox("Has Partner?", ["No", "Yes"])
    dependents = st.sidebar.selectbox("Has Dependents?", ["No", "Yes"])
    tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
    phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_sec = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
    backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    stream_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    stream_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charge = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    total_charge = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 2500.0)

    return pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': 1 if senior == "Yes" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet,
        'OnlineSecurity': online_sec,
        'OnlineBackup': backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': stream_tv,
        'StreamingMovies': stream_movies,
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment,
        'MonthlyCharges': monthly_charge,
        'TotalCharges': total_charge
    }])

def preprocess_input(df):
    df_encoded = pd.get_dummies(df)
    for col in set(model_columns) - set(df_encoded.columns):
        df_encoded[col] = 0
    df_encoded = df_encoded[model_columns]
    df_encoded[['MonthlyCharges', 'TotalCharges']] = scaler.transform(df_encoded[['MonthlyCharges', 'TotalCharges']])
    return df_encoded

input_df = get_user_input()

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Customer Summary")
    st.dataframe(input_df.T, use_container_width=True)

with col2:
    st.subheader("Prediction Result")
    if st.button("Predict Churn"):
        processed = preprocess_input(input_df)
        prediction = model.predict(processed)
        prob = model.predict_proba(processed)[0][1]

        if prediction[0] == 1:
            st.error(f"❌ Customer is likely to churn.")
        else:
            st.success(f"✅ Customer is likely to stay.")

        # style_metric_cards()
        st.metric(label="Churn Probability", value=f"{prob:.2%}", delta=None)
        st.progress(prob)

st.markdown("---")
st.markdown("### Insights")
st.markdown("You can adjust inputs on the sidebar to simulate different customer scenarios and see how behavior impacts churn.")
