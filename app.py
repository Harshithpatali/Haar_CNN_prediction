import streamlit as st
import pandas as pd
import numpy as np
import pywt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the saved model
model = load_model("nifty_lstm_model.h5")

# Title
st.title("📈 NIFTY Next-Day Trend Predictor (LSTM)")

# Upload CSV
uploaded_file = st.file_uploader("Upload your normalized NIFTY OHLCV CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.write("Data Preview:", df.head())

    # Extract and scale features
    features = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Create last sequence for prediction
    window_size = 32
    if len(features_scaled) < window_size:
        st.error("Not enough data for a full sequence (need at least 32 rows).")
    else:
        last_seq = features_scaled[-window_size:]
        input_seq = np.expand_dims(last_seq, axis=0)  # (1, 32, 5)

        # Predict
        prediction = model.predict(input_seq)
        pred_class = np.argmax(prediction)
        confidence = prediction[0][pred_class]

        # Output result
        label = "📉 Downtrend" if pred_class == 0 else "📈 Uptrend"
        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: `{confidence:.2%}`")

        # Optional chart
        st.line_chart(df[['Close']].tail(100).reset_index(drop=True))

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using LSTM and Streamlit")
