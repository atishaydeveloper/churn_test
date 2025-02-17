import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open("churn_model.pkl", "rb") as file:
    model = pickle.load(file)

# Title of the web app
st.title("📊 Customer Churn Prediction App")

# File uploader for CSV dataset
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read CSV file
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview:")
    st.dataframe(data.head())

    # Check if data has necessary columns (Adjust this based on your model)
    required_columns = ["feature1", "feature2", "feature3"]  # Replace with actual feature names
    if all(col in data.columns for col in required_columns):
        # Make predictions
        predictions = model.predict(data[required_columns])

        # Show predictions
        data["Churn Prediction"] = predictions
        st.write("### Prediction Results:")
        st.dataframe(data)

        # Download the results
        st.download_button(
            label="📥 Download Predictions",
            data=data.to_csv(index=False),
            file_name="churn_predictions.csv",
            mime="text/csv"
        )
    else:
        st.error("⚠️ The uploaded CSV does not have the required columns!")

