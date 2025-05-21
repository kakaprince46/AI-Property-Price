import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load data
data = pd.read_csv('property_data.csv')

# Preprocessing
features = data.drop('price', axis=1)
target = data['price']

# Identify categorical and numerical columns
categorical_cols = ['location', 'amenities']
numerical_cols = ['size_sqft', 'bedrooms', 'bathrooms']

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Create model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train model
model.fit(features, target)

# Save model
joblib.dump(model, 'model/model.pkl')
print("Model trained and saved!")
