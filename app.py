from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load('diabetes_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')

# Initialize Flask app
app = Flask(__name__)

# Define the recommendation system
def recommend_plan(screentime, bmi, age):
    recommendations = []
    if screentime > 240:
        recommendations.append("Reduce screentime to less than 4 hours daily.")
        recommendations.append("Take breaks every 30 minutes of screen use.")
    if bmi > 25:
        recommendations.append("Incorporate daily 30-minute brisk walking or yoga.")
    elif bmi < 18.5:
        recommendations.append("Focus on a calorie-rich diet with protein shakes.")
    if age > 50:
        recommendations.append("Include light exercises and regular health check-ups.")
    return recommendations

# Define a prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Extract input data from the request
    data = request.json
    features = [
        data['Pregnancies'],
        data['Glucose'],
        data['BloodPressure'],
        data['SkinThickness'],
        data['Insulin'],
        data['BMI'],
        data['DiabetesPedigreeFunction'],
        data['Age'],
        data['Avg_Usage'],
        data['Avg_Notifications'],
        data['Avg_Times_Opened']
    ]
    
    # Scale features and make a prediction
    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features)
    probability = model.predict_proba(scaled_features)[0][1]
    
    # Generate recommendations
    recommendations = recommend_plan(
        data['Avg_Usage'], data['BMI'], data['Age']
    )
    
    # Return the result
    result = {
        'Diabetes_Risk': int(prediction[0]),
        'Probability': probability,
        'Recommendations': recommendations
    }
    return jsonify(result)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
