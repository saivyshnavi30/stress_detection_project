import streamlit as st
from src.preprocess import load_data, clean_data
from src.feature_engineering import extract_features, label_stress
from src.train_model import train_model
from src.realtime_simulation import simulate_realtime

st.title("🎮 Real-Time Stress Detection System")

# Load data
df = load_data("data/sample_data.csv")
df = clean_data(df)

st.subheader("📊 Raw Data")
st.write(df)

# Feature extraction
features = extract_features(df)

# Labels
labels = label_stress(df)
labels = labels.iloc[-len(features):]

# Train model
model = train_model(features, labels)

st.subheader("⚙️ Features")
st.write(features)

st.subheader("📈 Stress Levels")
st.line_chart(labels)

st.subheader("📉 Heart Rate & HRV")
st.line_chart(features[['HR', 'HRV']])

# Real-time simulation
if st.button("▶ Run Real-Time Simulation"):
    simulate_realtime(model, features)