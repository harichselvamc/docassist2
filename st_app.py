import streamlit as st
import numpy as np
import pickle

# Load the trained model and scaler
model_file_path = 'random_forest_model.pkl'
scaler_file_path = 'scaler.pkl'
with open(model_file_path, 'rb') as model_file:
    loaded_model = pickle.load(model_file)
with open(scaler_file_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define the recommendation function
def recommend(source):
    if source == 1:
        return "Maintain a balanced diet and consult a doctor for regular checkups."
    elif source == 0:
        return "Your health parameters are stable. Continue with your current lifestyle."
    else:
        return "Consider a detailed medical examination for potential issues."

# Streamlit UI
st.title("Health Prediction System")
st.write("Enter your health parameters to get a prediction and recommendation.")

# Input form
with st.form("input_form"):
    HAEMATOCRIT = st.number_input("HAEMATOCRIT", min_value=0.0, max_value=100.0, step=0.1)
    HAEMOGLOBINS = st.number_input("HAEMOGLOBINS", min_value=0.0, max_value=20.0, step=0.1)
    ERYTHROCYTE = st.number_input("ERYTHROCYTE", min_value=0.0, max_value=10.0, step=0.1)
    LEUCOCYTE = st.number_input("LEUCOCYTE", min_value=0.0, max_value=20.0, step=0.1)
    THROMBOCYTE = st.number_input("THROMBOCYTE", min_value=0.0, max_value=500.0, step=1.0)
    MCH = st.number_input("MCH", min_value=0.0, max_value=50.0, step=0.1)
    MCHC = st.number_input("MCHC", min_value=0.0, max_value=50.0, step=0.1)
    MCV = st.number_input("MCV", min_value=0.0, max_value=150.0, step=0.1)
    AGE = st.number_input("AGE", min_value=0, max_value=120, step=1)
    
    submitted = st.form_submit_button("Submit")

# Prediction and result display
if submitted:
    # Prepare the input data
    input_data = np.array([[HAEMATOCRIT, HAEMOGLOBINS, ERYTHROCYTE, LEUCOCYTE, THROMBOCYTE, MCH, MCHC, MCV, AGE]])
    input_scaled = scaler.transform(input_data)

    # Make predictions
    prediction = loaded_model.predict(input_scaled)[0]
    recommendation = recommend(prediction)

    # Display results
    st.subheader("Prediction Result")
    st.write(f"Prediction: {'At Risk' if prediction == 1 else 'Healthy'}")
    st.write(f"Recommendation: {recommendation}")
