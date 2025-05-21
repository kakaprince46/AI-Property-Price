import os
import json
from flask import Flask, render_template, request
import joblib
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Firebase config from environment variables
firebase_config = {
    "type": os.environ.get("FIREBASE_TYPE"),
    "project_id": os.environ.get("FIREBASE_PROJECT_ID"),
    "private_key_id": os.environ.get("FIREBASE_PRIVATE_KEY_ID"),
    "private_key": os.environ.get("FIREBASE_PRIVATE_KEY").replace('\\n', '\n'),
    "client_email": os.environ.get("FIREBASE_CLIENT_EMAIL"),
    "client_id": os.environ.get("FIREBASE_CLIENT_ID"),
    "auth_uri": os.environ.get("FIREBASE_AUTH_URI"),
    "token_uri": os.environ.get("FIREBASE_TOKEN_URI"),
    "auth_provider_x509_cert_url": os.environ.get("FIREBASE_AUTH_PROVIDER_CERT_URL"),
    "client_x509_cert_url": os.environ.get("FIREBASE_CLIENT_CERT_URL")
}

# Initialize Firebase
cred = credentials.Certificate(firebase_config)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load trained model using dynamic path
model_path = os.path.join(os.path.dirname(__file__), 'model/model.pkl')
model = joblib.load(model_path)

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
