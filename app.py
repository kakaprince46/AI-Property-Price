import os
from flask import Flask, render_template, request
import joblib
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize Firebase
cred = credentials.Certificate("firebase-key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load trained model
model = joblib.load('model/model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get and validate form data
        location = request.form.get('location', '').strip()
        size_sqft = request.form.get('size_sqft', '').strip()
        bedrooms = request.form.get('bedrooms', '').strip()
        bathrooms = request.form.get('bathrooms', '').strip()
        amenities = request.form.get('amenities', '').strip()

        # Validate and convert inputs
        if not location or not size_sqft or not bedrooms or not bathrooms:
            return "Missing required form fields", 400

        data = {
            'location': location,
            'size_sqft': float(size_sqft),
            'bedrooms': int(bedrooms),
            'bathrooms': int(bathrooms),
            'amenities': amenities
        }

        # Create DataFrame with same structure as training data
        input_data = pd.DataFrame([data])

        # Make prediction
        predicted_price = model.predict(input_data)[0]

        # Save to Firebase
        doc_ref = db.collection('predictions').document()
        doc_ref.set({
            **data,
            'predicted_price': float(predicted_price),
            'timestamp': firestore.SERVER_TIMESTAMP
        })

        return render_template('result.html',
                               prediction=round(predicted_price, 2),
                               **data)

    except ValueError as ve:
        return f"Invalid input: {ve}", 400
    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
