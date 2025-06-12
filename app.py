# app.py
import streamlit as st
import numpy as np
import joblib

# Load model and preprocessors
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# Webpage layout
st.title("Tweet Emotion Detection App")
st.markdown("🔍 *Enter a tweet and this app will predict the emotion behind it.*")

# Input from user
tweet = st.text_area("Enter your tweet below:")

# Predict button
if st.button("Predict Emotion"):
    if tweet.strip() != "":
        # Feature Engineering
        tfidf_text = vectorizer.transform([tweet]).toarray()
        numeric = scaler.transform([[len(tweet), len(tweet.split()), int('?' in tweet), int('!' in tweet)]])
        features = np.hstack((tfidf_text, numeric))

        # Prediction
        prediction = model.predict(features)
        emotion = le.inverse_transform(prediction)[0]

        st.success(f"Predicted Emotion: **{emotion}**")
    else:
        st.warning("Please enter a tweet before clicking Predict.")

