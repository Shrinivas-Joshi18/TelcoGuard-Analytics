import streamlit as st
import pandas as pd
import joblib

# --- Load Model and Scaler ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('models/fraud_detection_model.joblib')
        scaler = joblib.load('models/scaler.joblib')
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_assets()
if not model or not scaler:
    st.error("Model or scaler files not found. Please ensure they are in the 'models' directory.")
    st.stop()

# --- Function to load local CSS ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Apply CSS ---
local_css("style.css")

# --- Page Configuration ---
st.set_page_config(page_title="TelcoGuard Analytics", page_icon="🛡️", layout="wide")

# --- UI LAYOUT ---
st.title("🛡️ TelcoGuard Fraud Analytics")

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("Transaction Details")
    transaction_type = st.selectbox("Type", ("CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"))
    amount = st.number_input("Amount", min_value=0.0, format="%.2f")
    oldbalanceOrg = st.number_input("Sender's Balance Before", min_value=0.0, format="%.2f")
    newbalanceOrig = st.number_input("Sender's Balance After", min_value=0.0, format="%.2f")
    predict_button = st.button("Analyze Transaction", type="primary", use_container_width=True)

# --- Main Page Logic & Result Display ---
if predict_button:
    # Create DataFrame, scale, and predict
    input_data = pd.DataFrame({
        'step': [1], 'amount': [amount], 'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrig': [newbalanceOrig], 'oldbalanceDest': [0], 'newbalanceDest': [0],
        'type_CASH_OUT': [1 if transaction_type == 'CASH_OUT' else 0],
        'type_DEBIT': [1 if transaction_type == 'DEBIT' else 0],
        'type_PAYMENT': [1 if transaction_type == 'PAYMENT' else 0],
        'type_TRANSFER': [1 if transaction_type == 'TRANSFER' else 0],
    })
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)

    # Display results inside our custom styled "card"
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.subheader("Risk Assessment Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction[0] == 1:
            st.error("High Risk of Fraud Detected 🚨")
        else:
            st.success("Low Risk of Fraud ✅")
            
    with col2:
        confidence = prediction_proba[0][prediction[0]] * 100
        st.metric(label="Model Confidence", value=f"{confidence:.2f}%")
        
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Enter transaction details in the sidebar and click 'Analyze Transaction' to see the risk assessment.")