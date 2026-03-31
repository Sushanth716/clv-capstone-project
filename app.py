import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("E:/CLV_Project/model/clv_model.pkl", "rb"))

st.title("💰 CLV Predictor")

# Inputs
recency = st.number_input("Recency", min_value=0)
frequency = st.number_input("Frequency", min_value=1)
monetary = st.number_input("Monetary", min_value=0.0)

if st.button("Predict"):

    # 🔥 Feature Engineering (IMPORTANT)
    avg_order_value = monetary / frequency
    purchase_frequency = frequency / (recency + 1)
    lifespan = recency + 1
    monetary_frequency = monetary * frequency
    recency_frequency = recency * frequency

    input_data = pd.DataFrame({
        "Recency": [recency],
        "Frequency": [frequency],
        "Monetary": [monetary],
        "AvgOrderValue": [avg_order_value],
        "PurchaseFrequency": [purchase_frequency],
        "Lifespan": [lifespan],
        "Monetary_Frequency": [monetary_frequency],
        "Recency_Frequency": [recency_frequency]
    })

    prediction = model.predict(input_data)[0]

    if prediction == 0:
        st.error("Low Value Customer")
    elif prediction == 1:
        st.warning("Medium Value Customer")
    else:
        st.success("High Value Customer")
        st.markdown("---")
st.write("### 📊 Customer Insights Dashboard")
df = pd.read_csv("E:/CLV_Project/data/feature_engineered_data.csv")
st.bar_chart(df["Frequency"])
import shap

explainer = shap.Explainer(model, df.drop(["Customer ID", "CLV_Segment"], axis=1))
shap_values = explainer(df.drop(["Customer ID", "CLV_Segment"], axis=1))

st.write("### Feature Importance")
st.pyplot(shap.summary_plot(shap_values, df.drop(["Customer ID", "CLV_Segment"], axis=1)))