import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="SafeStep AI", page_icon="🧠", layout="centered")

st.title(" SafeStep AI")
st.subheader("AI-Based Wearable Fall Risk Prediction System")

st.markdown("---")

# -------------------------------
# CREATE SIMULATED DATASET
# -------------------------------

np.random.seed(42)

data = pd.DataFrame({
    "AccelerationX": np.random.uniform(-15, 15, 5000),
    "AccelerationY": np.random.uniform(-15, 15, 5000),
    "AccelerationZ": np.random.uniform(-15, 15, 5000)
})

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------

# Acceleration magnitude
data["Magnitude"] = np.sqrt(
    data["AccelerationX"]**2 +
    data["AccelerationY"]**2 +
    data["AccelerationZ"]**2
)

# Motion instability features
data["Variance"] = data["Magnitude"].rolling(window=5).var().fillna(0)
data["Jerk"] = data["Magnitude"].diff().fillna(0)

# -------------------------------
# CREATE RISK LABELS
# -------------------------------

def label_risk(mag):
    if mag > 18:
        return 2   # fall
    elif mag > 12:
        return 1   # instability
    else:
        return 0   # normal

data["label"] = data["Magnitude"].apply(label_risk)

# -------------------------------
# TRAIN MACHINE LEARNING MODEL
# -------------------------------

X = data[["Magnitude", "Variance", "Jerk"]]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# -------------------------------
# USER INPUT SECTION
# -------------------------------

st.header("📡 Sensor Input Simulation")

ax = st.number_input("Acceleration X", value=0.0)
ay = st.number_input("Acceleration Y", value=0.0)
az = st.number_input("Acceleration Z", value=0.0)

# -------------------------------
# PREDICTION BUTTON
# -------------------------------

if st.button("Analyze Movement"):

    # calculate features from user input
    magnitude = np.sqrt(ax**2 + ay**2 + az**2)
    variance = magnitude * 0.1
    jerk = magnitude * 0.05

    sample = pd.DataFrame(
        [[magnitude, variance, jerk]],
        columns=["Magnitude", "Variance", "Jerk"]
    )

    prediction = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]

    st.markdown("---")
    st.subheader("📊 AI Motion Risk Analysis")

    # -------------------------------
    # DISPLAY RESULT
    # -------------------------------

    if prediction == 0:
        st.success("🟢 Normal Movement")
        st.write("Risk Level: Low")

    elif prediction == 1:
        st.warning("🟡 Instability Detected")
        st.write("Risk Level: Moderate")
        st.info("Please stabilize movement")

    else:
        st.error("🔴 HIGH FALL RISK")
        st.write("Risk Level: Critical")
        st.warning("🚨 Emergency Alert Triggered")
        st.info("📞 Caregiver Notified")