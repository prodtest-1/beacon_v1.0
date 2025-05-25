import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
import spacy
import re
import tensorflow as tf
from transformers import pipeline

# ---------------------------
# LOAD MODELS
# ---------------------------
# Load NLP
nlp_spacy = spacy.load("en_core_web_sm")
ner_hf = pipeline("ner", grouped_entities=True)

# Load or define XGBoost (mock for now)
def predict_xgboost(inputs):
    model = xgb.XGBClassifier()
    # Placeholder dummy logic
    return np.random.choice([0, 1]), np.random.uniform(0.6, 0.95)

# Load or define LSTM (mock)
def predict_lstm(sequence):
    return np.random.uniform(0, 1)

# Load or define autoencoder (mock)
def detect_anomaly(user_pattern):
    return np.random.uniform(0, 0.05) > 0.02  # True = Anomaly

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Smart Subscription Tracker", layout="centered")

st.title("📊 Smart Subscription Manager with AI")

st.sidebar.header("🔍 Select Feature")
feature = st.sidebar.radio("Choose", [
    "1️⃣ Predict Cancellation Risk",
    "2️⃣ Detect Anomalies (Autoencoder)",
    "3️⃣ Time Series Behavior (LSTM)",
    "4️⃣ Extract Subscription from Email/Text"
])

# ---------------------------
# 1. XGBoost Predictor
# ---------------------------
if feature.startswith("1"):
    st.header("📈 Cancellation Risk Predictor")

    monthly_cost = st.slider("Monthly Cost ($)", 1, 50, 10)
    days_since_login = st.slider("Days Since Last Login", 0, 100, 30)
    duration = st.slider("Subscription Duration (months)", 1, 36, 6)
    has_reminder = st.selectbox("Has Reminder Set?", ["No", "Yes"]) == "Yes"
    logins_last_month = st.slider("Logins Last Month", 0, 50, 5)
    session_time = st.slider("Average Session Length (min)", 1, 120, 10)
    category = st.selectbox("Subscription Type", ["entertainment", "utility"])

    # Predict
    input_vec = [monthly_cost, days_since_login, duration, int(has_reminder), logins_last_month, session_time, 0 if category == "entertainment" else 1]
    label, prob = predict_xgboost(input_vec)

    st.success(f"🧠 Predicted Risk: {'⚠️ Likely to Forget' if label else '✅ Will Remember'} ({prob:.2%} confidence)")

# ---------------------------
# 2. Anomaly Detector
# ---------------------------
elif feature.startswith("2"):
    st.header("🕵️ Detect Anomalous Behavior (Autoencoder)")

    pattern = st.text_input("Enter 6 months login pattern (comma-separated)", "9,8,10,9,0,0")
    user_vector = np.array([int(x.strip()) for x in pattern.split(",")])

    is_anomaly = detect_anomaly(user_vector)
    st.warning("⚠️ Unusual behavior detected!") if is_anomaly else st.success("✅ Behavior appears normal.")

# ---------------------------
# 3. LSTM Sequence Forecast
# ---------------------------
elif feature.startswith("3"):
    st.header("📊 Time Series User Behavior")

    sequence = st.text_input("Enter last 6 months of login activity", "10,8,5,3,1,0")
    seq_array = np.array([int(x.strip()) for x in sequence.split(",")]).reshape(1, 6, 1)

    prob = predict_lstm(seq_array)
    st.info(f"🧠 Likelihood of Forgetting Next Month: {prob:.2%}")

# ---------------------------
# 4. NLP Subscription Extractor
# ---------------------------
elif feature.startswith("4"):
    st.header("📩 Subscription Extractor from Text")

    sample = st.text_area("Paste email or receipt text", """
        Hello! You've successfully subscribed to Netflix. You’ll be billed $15.99 on June 1st, 2025.
    """)

    st.subheader("🧠 spaCy NER:")
    doc = nlp_spacy(sample)
    for ent in doc.ents:
        st.write(f"{ent.text} → {ent.label_}")

    st.subheader("🤗 Transformers NER:")
    entities = ner_hf(sample)
    for e in entities:
        st.write(f"{e['word']} → {e['entity_group']} (score: {e['score']:.2f})")

    st.subheader("💡 Pattern Matches")
    price_match = re.search(r"\$\d+\.\d+", sample)
    date_match = re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w* \d{1,2}", sample)

    st.write("💵 Price:", price_match.group() if price_match else "Not found")
    st.write("📅 Renewal Date:", date_match.group() if date_match else "Not found")
