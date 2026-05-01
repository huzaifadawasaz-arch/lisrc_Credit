import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set page configuration
st.set_page_config(page_title="Credit Approval Predictor", layout="wide")

st.title("🎯 Credit Approval Predictor")
st.write("Predict credit card approval using machine learning (Random Forest)")

@st.cache_data
def load_data():
    return pd.read_csv('credit (3) (1).csv')

@st.cache_resource
def build_model():
    df = load_data()
    X = df.drop('approved', axis=1)
    y = df['approved']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }

    return model, metrics

model, metrics = build_model()

# Create layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Enter Applicant Information")
    age = st.slider("Age", min_value=18, max_value=80, value=35)
    income = st.number_input("Annual Income ($)", min_value=20000.0, max_value=150000.0, value=50000.0, step=1000.0)
    years_at_job = st.slider("Years at Current Job", min_value=0, max_value=50, value=5)
    credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=700)
    existing_cards = st.slider("Existing Credit Cards", min_value=0, max_value=10, value=2)

with col2:
    st.subheader("📊 Model Performance")
    st.metric("Model Accuracy", f"{metrics['accuracy']*100:.2f}%")
    st.metric("Precision", f"{metrics['precision']*100:.2f}%")
    st.metric("Recall", f"{metrics['recall']*100:.2f}%")
    st.metric("F1 Score", f"{metrics['f1_score']*100:.2f}%")

if st.button("🔮 Predict Credit Approval", use_container_width=True):
    input_data = np.array([[age, income, years_at_job, credit_score, existing_cards]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.divider()
    st.subheader("📈 Prediction Result")

    if prediction == 1:
        st.success("✅ APPROVED", icon="✅")
        st.write(f"**Approval Probability:** {probability[1]*100:.2f}%")
    else:
        st.error("❌ NOT APPROVED", icon="❌")
        st.write(f"**Rejection Probability:** {probability[0]*100:.2f}%")

    st.divider()
    st.subheader("📋 Application Summary")
    summary_df = pd.DataFrame({
        'Attribute': ['Age', 'Annual Income', 'Years at Job', 'Credit Score', 'Existing Credit Cards'],
        'Value': [f"{age} years", f"${income:,.0f}", f"{years_at_job} years", f"{credit_score}", f"{existing_cards}"]
    })
    st.table(summary_df)

st.divider()
st.info("💡 This app trains the model at startup from the included dataset. No external pickle files are required.")
