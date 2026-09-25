import streamlit as st
import pandas as pd
import joblib
from etherscan_helper import get_wallet_features

st.set_page_config(page_title="Web3 AI Fraud Detector", page_icon="🛡️")

st.title("🛡️ Web3 AI Fraud & Anomaly Detector")
st.write("Analyze any Ethereum wallet address in real-time to assess risk using Machine Learning.")

# Load trained model and features
model = joblib.load('fraud_model.pkl')
features_list = joblib.load('model_features.pkl')

wallet_address = st.text_input("Enter Ethereum Wallet Address:", placeholder="0x...")

if st.button("Analyze Risk"):
    if wallet_address:
        with st.spinner("Fetching on-chain transactions from Etherscan..."):
            features_df = get_wallet_features(wallet_address)
            
            if features_df is not None:
                # Predict probability
                prediction = model.predict(features_df[features_list])[0]
                proba = model.predict_proba(features_df[features_list])[0][1]
                
                st.subheader("Results")
                if prediction == 1:
                    st.error(f"🚨 HIGH RISK DETECTED! Probability of Fraud: {proba*100:.1f}%")
                else:
                    st.success(f"✅ LOW RISK / SAFE WALLET. Probability of Fraud: {proba*100:.1f}%")
                
                st.subheader("Calculated Wallet Metrics")
                st.dataframe(features_df.T.rename(columns={0: "Value"}))
    else:
        st.warning("Please enter a valid wallet address.")