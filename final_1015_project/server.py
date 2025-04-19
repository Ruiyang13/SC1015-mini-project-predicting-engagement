from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import joblib
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load your trained model
try:
    model = joblib.load('trained_model.joblib')
    ohe = joblib.load('ohe_encoder.joblib')
    print("Loaded existing model and encoder")
except:
    print("No existing model found. Training a new one...")
    try:
        # Load and prepare data exactly as in your notebook
        df = pd.read_csv('TikTok - Posts.csv')
        
        # Convert create_time to datetime and extract weekday name
        df['Timestamp'] = pd.to_datetime(df['create_time'])
        df['Weekday'] = df['Timestamp'].dt.day_name()
        
        # Convert weekday names to numbers (Monday=0, Sunday=6)
        weekday_map = {
            'Monday': 0,
            'Tuesday': 1,
            'Wednesday': 2,
            'Thursday': 3,
            'Friday': 4,
            'Saturday': 5,
            'Sunday': 6
        }
        df['weekday'] = df['Weekday'].map(weekday_map)
        
        # Select features exactly as in your notebook
        clean_categorical_cols = ['weekday', 'is_verified']
        clean_numeric_cols = ['play_count', 'profile_followers', 'video_duration']
        
        # Prepare categorical features
        df_cat = df[clean_categorical_cols]
        ohe = OneHotEncoder()
        ohe.fit(df_cat)
        df_cat_ohe = pd.DataFrame(ohe.transform(df_cat).toarray(), 
                                columns=ohe.get_feature_names_out(df_cat.columns))
        
        # Prepare numeric features
        df_num = df[clean_numeric_cols]
        
        # Combine features
        X = pd.concat([df_num, df_cat_ohe], axis=1)
        y = np.log1p(df['digg_count'])  # Using digg_count as target
        
        # Split data exactly as in your notebook
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train model with exact same parameters
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Save the model and encoder
        joblib.dump(model, 'trained_model.joblib')
        joblib.dump(ohe, 'ohe_encoder.joblib')
        print("Model and encoder trained and saved successfully")
        
        # Print performance metrics exactly as in your notebook
        print("\nTraining Performance:")
        print("MSE:", mean_squared_error(y_train, model.predict(X_train)))
        print("R²:", r2_score(y_train, model.predict(X_train)))
        
        print("\nTest Performance:")
        print("MSE:", mean_squared_error(y_test, model.predict(X_test)))
        print("R²:", r2_score(y_test, model.predict(X_test)))
        
    except Exception as e:
        print("Error training model:", str(e))
        raise e

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json
        
        # Prepare numeric features
        numeric_features = np.array([[
            data['play_count'],
            data['profile_followers'],
            data['video_duration']
        ]])
        
        # Prepare categorical features
        categorical_features = np.array([[
            data['weekday'],
            data['is_verified']
        ]])
        
        # One-hot encode categorical features
        cat_encoded = ohe.transform(categorical_features).toarray()
        
        # Combine features
        features = np.hstack([numeric_features, cat_encoded])
        
        # Make prediction
        log_prediction = model.predict(features)[0]
        prediction = np.exp(log_prediction) - 1
        
        return jsonify({'prediction': float(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) 