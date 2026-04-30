import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Set page configuration
st.set_page_config(page_title="Credit Approval Predictor", layout="wide")

st.title("🎯 Credit Approval Predictor")
st.write("Predict credit card approval using machine learning (Random Forest - 95.5% Accuracy)")

# Load model and scaler
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("Model files not found. Please run train_model.py first.")
    st.stop()

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Enter Applicant Information")
    
    age = st.slider("Age", min_value=18, max_value=80, value=35)
    income = st.number_input("Annual Income ($)", min_value=20000, max_value=150000, value=50000, step=1000)
    years_at_job = st.slider("Years at Current Job", min_value=0, max_value=50, value=5)
    credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=700)
    existing_cards = st.slider("Existing Credit Cards", min_value=0, max_value=10, value=2)

with col2:
    st.subheader("📊 Model Performance")
    st.metric("Model Accuracy", "95.50%")
    st.metric("Model Type", "Random Forest")
    st.metric("Precision", "86.67%")
    st.metric("Recall", "98.11%")

# Make prediction
if st.button("🔮 Predict Credit Approval", use_container_width=True):
    # Prepare input data
    input_data = np.array([[age, income, years_at_job, credit_score, existing_cards]])
    
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    # Display result
    st.divider()
    st.subheader("📈 Prediction Result")
    
    if prediction == 1:
        st.success(f"✅ APPROVED", icon="✅")
        st.write(f"**Approval Probability:** {probability[1]*100:.2f}%")
    else:
        st.error(f"❌ NOT APPROVED", icon="❌")
        st.write(f"**Rejection Probability:** {probability[0]*100:.2f}%")
    
    # Show input summary
    st.divider()
    st.subheader("📋 Application Summary")
    
    summary_df = pd.DataFrame({
        'Attribute': ['Age', 'Annual Income', 'Years at Job', 'Credit Score', 'Existing Credit Cards'],
        'Value': [f"{age} years", f"${income:,}", f"{years_at_job} years", f"{credit_score}", f"{existing_cards}"]
    })
    
    st.table(summary_df)

st.divider()
st.info("💡 This model is trained on historical credit approval data. Predictions are for demonstration purposes.")
