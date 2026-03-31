import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Diet Consultant",
    page_icon="🥗",
    layout="centered"
)

# --- CUSTOM CSS FOR HIGH-END UI ---
st.markdown("""
    <style>
    .stApp {
        background-color: black;
    }

    /* Input Header Styling */
    .input-header {
        color: #1b5e20;
        font-weight: 800;
        padding-bottom: 10px;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 20px;
    }

    /* Modern Button */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.5em;
        background-color: #2e7d32;
        color: white !important;
        font-weight: bold;
        font-size: 18px;
        border: none;
        transition: 0.3s;
        box-shadow: 0 4px 15px rgba(46, 125, 50, 0.2);
    }
    .stButton>button:hover {
        background-color: #1b5e20;
        box-shadow: 0 6px 20px rgba(27, 94, 32, 0.4);
        transform: translateY(-2px);
    }

    /* Prediction Result Card - Fixed for Black Text */
    .prediction-box {
        padding: 30px;
        background-color: #ffffff;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 8px solid #2e7d32;
        margin-top: 30px;
    }

    .result-label {
        color: #444444 !important;
        font-size: 1.1em;
        font-weight: 600;
        margin-bottom: 5px;
    }

    .result-value {
        color: #000000 !important; /* Pure Black for visibility */
        font-size: 3.2em !important;
        font-weight: 900 !important;
        margin: 10px 0;
        line-height: 1.1;
    }

    .metrics-summary {
        color: #666;
        font-size: 0.9em;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)


# --- LOAD BALANCED ASSETS ---
@st.cache_resource
def load_assets():
    try:
        # These must be the files exported after your SMOTE re-training
        model = pickle.load(open("Final_Diet_Model.pkl", "rb"))
        scaler = pickle.load(open("Final_Scaler.pkl", "rb"))
        labels = pickle.load(open("Diet_Labels.pkl", "rb"))
        return model, scaler, labels
    except FileNotFoundError:
        return None, None, None


model, scaler, labels = load_assets()

# --- HEADER SECTION ---
st.markdown("<h1 style='text-align: center; color: #1b5e20;'>🥗 AI Personal Diet Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Optimized Recommendation Engine (Balanced Version)</p>",
            unsafe_allow_html=True)

if model is None:
    st.error("⚠️ **Error:** Model files not found! Ensure you ran the Export Cell in your Jupyter Notebook.")
    st.stop()

# --- INPUT FORM ---
with st.container():
    st.markdown("<div class='input-header'>📋 Your Biological Metrics</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        age = st.number_input("Age", min_value=1, max_value=110, value=30)
        gender = st.selectbox("Gender", options=[(1, "Male"), (0, "Female")], format_func=lambda x: x[1])[0]
        height = st.number_input("Height (cm)", min_value=50, max_value=250, value=175)
        weight = st.number_input("Weight (kg)", min_value=10, max_value=250, value=80)

    with col2:
        activity = st.selectbox("Activity Level",
                                options=[(0, "Sedentary"), (1, "Moderate"), (2, "Active")],
                                format_func=lambda x: x[1])[0]
        sugar = st.number_input("Sugar Level (mg/dL)", min_value=40, max_value=500, value=100)
        cholesterol = st.number_input("Cholesterol Level", min_value=80, max_value=500, value=190)

# --- PREDICTION LOGIC ---
if st.button("Generate My Diet Plan"):
    # 1. Feature Order must match exactly:
    columns = ['Age', 'Gender', 'Height_cm', 'Weight_kg', 'Activity_Level', 'Sugar_Level', 'Cholesterol']

    # 2. Convert to DataFrame (prevents UserWarnings)
    input_df = pd.DataFrame([[age, gender, height, weight, activity, sugar, cholesterol]], columns=columns)

    # 3. Scale using the production scaler
    input_scaled = scaler.transform(input_df)
    input_scaled_df = pd.DataFrame(input_scaled, columns=columns)

    # 4. Get Prediction
    prediction_idx = model.predict(input_scaled_df)[0]
    final_diet = labels[prediction_idx]

    # 5. Display the result
    st.markdown(f"""
        <div class="prediction-box">
            <p class="result-label">The AI suggests a:</p>
            <h1 class="result-value">{final_diet}</h1>
            <p class="metrics-summary">Calculated based on your specific metabolic profile.</p>
            <hr style="border: 0.5px solid #eee; margin: 20px 0;">
            <p style="color: #2e7d32; font-size: 0.85em;"><b>Note:</b> This model was trained on balanced data with 93% accuracy. Please consult a doctor for medical diets.</p>
        </div>
    """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown(
    "<br><p style='text-align: center; color: #bbb; font-size: 0.7em;'>Diet Prediction System v2.1 | Powered by SMOTE & Random Forest</p>",
    unsafe_allow_html=True)