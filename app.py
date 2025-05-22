import os
import json
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Debug Firebase configuration
print("\n=== Checking Firebase Configuration ===")
print("Project ID:", os.environ.get("FIREBASE_PROJECT_ID"))
print("Client Email:", os.environ.get("FIREBASE_CLIENT_EMAIL"))
print("Private Key Present:", bool(os.environ.get("FIREBASE_PRIVATE_KEY")))
print("="*40 + "\n")

# Handle Firebase private key formatting robustly
private_key = os.environ.get("FIREBASE_PRIVATE_KEY", "").strip()
private_key = private_key.strip('"').strip("'")  # Remove any accidental surrounding quotes

# Ensure proper PEM format
if private_key and not private_key.startswith('-----BEGIN PRIVATE KEY-----'):
    if '\\n' in private_key:
        private_key = private_key.replace('\\n', '\n')
    private_key = f"-----BEGIN PRIVATE KEY-----\n{private_key}\n-----END PRIVATE KEY-----"

# Debug print the beginning of the key
print("Private key starts with:", private_key[:50] + "...")

firebase_config = {
    "type": "service_account",
    "project_id": os.environ.get("FIREBASE_PROJECT_ID"),
    "private_key_id": os.environ.get("FIREBASE_PRIVATE_KEY_ID"),
    "private_key": private_key,
    "client_email": os.environ.get("FIREBASE_CLIENT_EMAIL"),
    "client_id": os.environ.get("FIREBASE_CLIENT_ID"),
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": os.environ.get("FIREBASE_CLIENT_CERT_URL")
}

# Initialize Firebase
try:
    cred = credentials.Certificate(firebase_config)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firebase initialized successfully!")
except Exception as e:
    print(f"Firebase initialization failed: {str(e)}")
    raise

# Verify and load model
model_path = os.path.join(os.path.dirname(__file__), 'model/model.pkl')
print("\n=== Checking Model ===")
print("Model path:", model_path)

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

try:
    model = joblib.load(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Model loading failed: {str(e)}")
    raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get and validate form data
        required_fields = ['location', 'size_sqft', 'bedrooms', 'bathrooms']
        data = {field: request.form.get(field, '').strip() for field in required_fields}
        amenities = request.form.get('amenities', '').strip()

        # Validate inputs
        if not all(data.values()):
            return jsonify({"error": "All fields are required"}), 400

        # Convert numerical fields
        try:
            data['size_sqft'] = float(data['size_sqft'])
            data['bedrooms'] = int(data['bedrooms'])
            data['bathrooms'] = int(data['bathrooms'])
        except ValueError as e:
            return jsonify({"error": f"Invalid number format: {str(e)}"}), 400

        # Prepare data for model
        input_data = pd.DataFrame([{
            'location': data['location'],
            'size_sqft': data['size_sqft'],
            'bedrooms': data['bedrooms'],
            'bathrooms': data['bathrooms'],
            'amenities': amenities
        }])

        # Make prediction
        predicted_price = model.predict(input_data)[0]

        # Save prediction to Firebase
        doc_ref = db.collection('predictions').document()
        doc_ref.set({
            **data,
            'amenities': amenities,
            'predicted_price': float(predicted_price),
            'timestamp': firestore.SERVER_TIMESTAMP
        })

        return render_template('result.html',
                            prediction=round(predicted_price, 2),
                            **data,
                            amenities=amenities)

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
