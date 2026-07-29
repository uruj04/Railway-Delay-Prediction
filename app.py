import streamlit as st
import pandas as pd
import joblib
# load dataset
train = pd.read_csv("ir_train.csv")

train[["train_number"]].drop_duplicates().to_csv(
    "train_numbers.csv", index=False
)
# creating list of train numbers
train_numbers = sorted(train["train_number"].unique())
# Load model and feature names
model = joblib.load("railway_delay_model.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(
    page_title="Indian Railway Delay Predictor",
    page_icon="🚆",
    layout="centered"
)

st.title("🚆 Indian Railway Delay Predictor")
st.write("Predict the expected train delay.")

# User Inputs
train_number = st.selectbox(
    "Select Train Number",
    train_numbers
)



journey_date = st.date_input("Journey Date")

if st.button("Predict Delay"):

    # Create input dataframe
    input_data = pd.DataFrame(columns=feature_names)
    input_data.loc[0] = 0

    # Fill user inputs
    input_data["train_number"] = train_number
    input_data["year"] = journey_date.year
    input_data["month"] = journey_date.month
    input_data["day"] = journey_date.day
    input_data["weekday"] = journey_date.weekday()

    # Predict
    prediction = model.predict(input_data)[0]

    # Convert minutes to HH:MM:SS
    total_seconds = int(prediction * 60)

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    st.success(
        f"Predicted Delay: {hours:02d}:{minutes:02d}:{seconds:02d}"
    )
