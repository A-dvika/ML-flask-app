from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('iris_svc_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract input features from the request
    features = [data['sepalLength'], data['sepalWidth'], data['petalLength'], data['petalWidth']]

    # Reshape the input for model prediction
    prediction = model.predict([features])

    # Convert prediction to species name
    species = ['setosa', 'versicolor', 'virginica']
    result = species[prediction[0]]

    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
