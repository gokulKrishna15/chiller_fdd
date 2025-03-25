
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, fbeta_score, accuracy_score, confusion_matrix

# 🔹 Set Page Config (Full Screen, No Sidebar)
st.set_page_config(page_title="Fault Detection Report", page_icon="📊", layout="wide")

# 🔹 Apply Minimalist Custom Styling
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(to right, #1E1E1E, #2C3E50);
            color: white;
        }
        h1 {
            text-align: center;
            font-size: 42px;
            font-weight: bold;
            color: #00C6FF; /* Light Blue */
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.6);
        }
        h2, h3 {
            text-align: center;
            font-size: 28px;
            color: #F39C12; /* Professional Gold */
        }
        .dataframe {
            background-color: white;
            color: black;
            border-radius: 8px;
            padding: 10px;
        }
        .metric-box {
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            padding: 15px;
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.1);
            margin-bottom: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# 🔹 Load CNN Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cnn_fault_detection_model.h5")

model = load_model()

# 🔹 Load Scaler & Encoder
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

fault_names = list(encoder.classes_)

st.title("📊 AI-Powered Fault Detection Report")
st.markdown("---")

try:
    df = pd.read_csv("preprocessed_data.csv")
    
    if "Fault_Type" in df.columns:
        actual_faults = df["Fault_Type"].values
        df = df.drop(columns=["Fault_Type"])
    else:
        st.error("🚨 'Fault_Type' column not found in dataset!")
        st.stop()

    # 🔹 Apply Scaling & Reshape for CNN
    X = scaler.transform(df.values)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # 🔹 Predict Faults
    predictions = model.predict(X)
    predicted_labels = np.argmax(predictions, axis=1)
    predicted_faults = np.array([fault_names[i] for i in predicted_labels])

    # 🔹 Compute Performance Metrics
    accuracy = accuracy_score(actual_faults, predicted_faults)
    precision = precision_score(actual_faults, predicted_faults, average='macro')
    recall = recall_score(actual_faults, predicted_faults, average='macro')
    f2_score = fbeta_score(actual_faults, predicted_faults, beta=2, average='macro')
    
    mismatches = sum(predicted_faults != actual_faults)
    total_samples = len(actual_faults)
    mismatch_percentage = (mismatches / total_samples) * 100

    # 🔹 Display Key Metrics
    col1, col2, col3 = st.columns(3)
    col1.markdown(f'<div class="metric-box">⚠️ Mismatch %: {mismatch_percentage:.2f}</div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="metric-box">✅ Precision: {precision:.4f}</div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="metric-box">📊 Recall: {recall:.4f}</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col4, col5 = st.columns(2)
    col4.markdown(f'<div class="metric-box">🎯 Accuracy: {accuracy:.4f}</div>', unsafe_allow_html=True)
    col5.markdown(f'<div class="metric-box">🚀 F2 Score: {f2_score:.4f}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 🔹 Confusion Matrix
    st.markdown("### 🎯 Confusion Matrix")
    cm = confusion_matrix(actual_faults, predicted_faults, labels=fault_names)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=fault_names, yticklabels=fault_names, ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    st.pyplot(fig)
    
    st.markdown("---")
    
    # 🔹 Sample Predictions
    df["Actual Fault"] = actual_faults
    df["Predicted Fault"] = predicted_faults
    st.markdown("### 🔍 Sample Predictions (First 20)")
    st.dataframe(df.head(20))
    
    # 🔹 Display Sample Mismatches
    mismatched_df = df[df["Actual Fault"] != df["Predicted Fault"]]
    if not mismatched_df.empty:
        st.markdown("### ❌ Sample Mismatches (First 20)")
        st.dataframe(mismatched_df.head(20))
    else:
        st.success("✅ No mismatches found!")

except FileNotFoundError:
    st.error("🚨 'preprocessed_data.csv' not found! Please check the file location.")


