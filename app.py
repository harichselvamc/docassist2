from flask import Flask, request, render_template
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained stacking model and scaler
model_file_path = 'stacking_model.pkl'  # Path to the saved model file
scaler_file_path = 'scaler.pkl'  # Path to the saved scaler file

# Load the model and scaler
best_model = joblib.load(model_file_path)
scaler = joblib.load(scaler_file_path)

# Define the recommendation function based on predicted source
def recommend(source):
    if source == 1:
        return "Maintain a balanced diet and consult a doctor for regular checkups."
    elif source == 0:
        return "Your health parameters are stable. Continue with your current lifestyle."
    else:
        return "Consider a detailed medical examination for potential issues."

# Define the routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get input data from the form
            input_data = [
                float(request.form['HAEMATOCRIT']),
                float(request.form['HAEMOGLOBINS']),
                float(request.form['ERYTHROCYTE']),
                float(request.form['LEUCOCYTE']),
                float(request.form['THROMBOCYTE']),
                float(request.form['MCH']),
                float(request.form['MCHC']),
                float(request.form['MCV']),
                float(request.form['AGE'])
            ]

            # Transform the input data using the loaded scaler
            input_data = np.array([input_data])  # Reshape for a single prediction
            input_scaled = scaler.transform(input_data)

            # Make prediction using the best model
            prediction = best_model.predict(input_scaled)[0]

            # Get recommendation based on the prediction
            recommendation = recommend(prediction)

            # Display results on the same page
            return render_template('index.html', 
                                   prediction='At Risk' if prediction == 1 else 'Healthy', 
                                   recommendation=recommendation)

        except Exception as e:
            return render_template('index.html', prediction="Error", recommendation=str(e))

    # Initial GET request to display the form
    return render_template('index.html')

# # Run the app
# if __name__ == '__main__':
#     app.run(debug=True)
