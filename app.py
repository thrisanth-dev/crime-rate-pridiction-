import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from datetime import datetime
import os
import sys

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Model and Data Persistence ---
MODEL_FILE = 'crime_prediction_rf_model.pkl'
SCALER_FILE = 'scaler.pkl'
ENCODERS_FILE = 'encoders.pkl'
METRICS_FILE = 'metrics.pkl'

# Global storage for dataset and trained components
df = None
rf_model = None
scaler = None
le_dict = {}
model_metrics = {}

# --- Utility Functions ---

def load_and_prepare_data():
    """Load data, preprocess, and train models (run once on server start)."""
    global df, rf_model, scaler, le_dict, model_metrics

    # Check for the CSV file in the root directory
    if not os.path.exists("crime_dataset_india.csv"):
        print("Error: crime_dataset_india.csv not found.")
        print("Please ensure the CSV file is in the same directory as app.py.")
        # Exit the application if data is missing
        return False

    try:
        df = pd.read_csv("crime_dataset_india.csv")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return False

    # --- Preprocessing & Feature Engineering ---
    df = df.rename(columns={
        'Date of Occurrence': 'Date',
        'Crime Description': 'Primary Type',
        'City': 'City',
        'Crime Domain': 'Crime Domain'
    })

    city_to_state = {
        'Ahmedabad': 'Gujarat', 'Chennai': 'Tamil Nadu', 'Ludhiana': 'Punjab',
        'Pune': 'Maharashtra', 'Delhi': 'Delhi', 'Kolkata': 'West Bengal',
        'Hyderabad': 'Telangana', 'Bengaluru': 'Karnataka', 'Jaipur': 'Rajasthan',
        'Lucknow': 'Uttar Pradesh', 'Mumbai': 'Maharashtra'
    }
    df['State'] = df['City'].map(city_to_state)

    df = df.dropna(subset=['Date', 'Primary Type', 'City', 'State', 'Victim Gender', 'Crime Domain', 'Weapon Used'])
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
    df = df.dropna(subset=['Date'])
    df['Victim Age'] = pd.to_numeric(df['Victim Age'], errors='coerce')

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Hour'] = df['Date'].dt.hour
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Latitude'] = 0.0 # Placeholder
    df['Longitude'] = 0.0 # Placeholder
    df['Case Closed'] = df['Case Closed'].map({'Yes': 1, 'No': 0})

    # Encode categorical variables
    categorical_cols = ['Primary Type', 'Crime Domain', 'Victim Gender', 'City', 'State', 'Weapon Used']
    for col in categorical_cols:
        le = LabelEncoder()
        # Handle new or missing categories during transformation by making sure only strings are encoded
        df[f'{col} Encoded'] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

    features = [f'{col} Encoded' for col in categorical_cols] + \
               ['Year', 'Month', 'Day', 'Hour', 'DayOfWeek', 'Latitude', 'Longitude']
    X = df[features]
    y = df['Case Closed']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # --- Model Training ---
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=500),
        "SVM": SVC(kernel='rbf', probability=True)
    }

    model_metrics = {'Accuracy': {}, 'F1-Score': {}}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        model_metrics['Accuracy'][name] = round(accuracy_score(y_test, y_pred), 4)
        model_metrics['F1-Score'][name] = round(f1_score(y_test, y_pred, zero_division=0), 4)
        if name == "Random Forest":
            rf_model = model

    print("Models trained successfully. Random Forest selected as predictor.")

    # Save components
    with open(MODEL_FILE, 'wb') as f: pickle.dump(rf_model, f)
    with open(SCALER_FILE, 'wb') as f: pickle.dump(scaler, f)
    with open(ENCODERS_FILE, 'wb') as f: pickle.dump(le_dict, f)
    with open(METRICS_FILE, 'wb') as f: pickle.dump(model_metrics, f)
    
    return True

def load_components():
    """Load trained components if they exist, otherwise train."""
    global df, rf_model, scaler, le_dict, model_metrics
    
    if os.path.exists(MODEL_FILE) and os.path.exists("crime_dataset_india.csv"):
        try:
            # Load ML components
            with open(MODEL_FILE, 'rb') as f: rf_model = pickle.load(f)
            with open(SCALER_FILE, 'rb') as f: scaler = pickle.load(f)
            with open(ENCODERS_FILE, 'rb') as f: le_dict = pickle.load(f)
            with open(METRICS_FILE, 'rb') as f: model_metrics = pickle.load(f)
            
            # Reload dataset and apply necessary columns for analysis
            df = pd.read_csv("crime_dataset_india.csv")
            df = df.rename(columns={'Date of Occurrence': 'Date', 'Crime Description': 'Primary Type', 'City': 'City', 'Crime Domain': 'Crime Domain'})
            city_to_state = {'Ahmedabad': 'Gujarat', 'Chennai': 'Tamil Nadu', 'Ludhiana': 'Punjab', 'Pune': 'Maharashtra', 'Delhi': 'Delhi', 'Kolkata': 'West Bengal', 'Hyderabad': 'Telangana', 'Bengaluru': 'Karnataka', 'Jaipur': 'Rajasthan', 'Lucknow': 'Uttar Pradesh', 'Mumbai': 'Maharashtra'}
            df['State'] = df['City'].map(city_to_state)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
            df['Year'] = df['Date'].dt.year
            df['Hour'] = df['Date'].dt.hour
            
            print("ML components loaded from disk.")
            return True
        except Exception as e:
            print(f"Error loading saved components ({e}). Retraining...")
            return load_and_prepare_data()
    else:
        print("No saved components found or CSV missing. Starting training...")
        return load_and_prepare_data()

# --- Flask Routes ---

@app.route('/')
def home():
    """Serves the main HTML page from the 'templates' folder."""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict_endpoint():
    """API endpoint for single prediction requests."""
    if rf_model is None:
        return jsonify({"error": "Model not trained or loaded."}), 500

    data = request.get_json()
    
    try:
        # Extract features and preprocess
        City = data['City']
        Primary_Type = data['Primary_Type']
        Victim_Gender = data['Victim_Gender']
        
        Victim_Age = int(data['Victim_Age'])
        Weapon_Used = data['Weapon_Used']
        
        # Temporal Feature Extraction
        dt_obj = datetime.strptime(f"{data['Date_Occurred']} {data['Time_Occurred']}", '%Y-%m-%d %H:%M:%S')
        Year = dt_obj.year
        Month = dt_obj.month
        Day = dt_obj.day
        Hour = dt_obj.hour
        DayOfWeek = dt_obj.weekday()
        Latitude = 0.0
        Longitude = 0.0

        city_to_state = {'Ahmedabad': 'Gujarat', 'Chennai': 'Tamil Nadu', 'Ludhiana': 'Punjab', 'Pune': 'Maharashtra', 'Delhi': 'Delhi', 'Kolkata': 'West Bengal', 'Hyderabad': 'Telangana', 'Bengaluru': 'Karnataka', 'Jaipur': 'Rajasthan', 'Lucknow': 'Uttar Pradesh', 'Mumbai': 'Maharashtra'}
        State = city_to_state.get(City, 'Unknown') 
        
        # Encoding categorical features
        # Note: If a category is missing in the encoder, it will raise a KeyError/ValueError
        City_Encoded = le_dict['City'].transform([City])[0]
        Primary_Type_Encoded = le_dict['Primary Type'].transform([Primary_Type])[0]
        Victim_Gender_Encoded = le_dict['Victim Gender'].transform([Victim_Gender])[0]
        Weapon_Used_Encoded = le_dict['Weapon Used'].transform([Weapon_Used])[0]
        State_Encoded = le_dict['State'].transform([State])[0]
        
        # Find Crime Domain for the given Primary Type (Crucial step matching training data)
        crime_domain_val = df.loc[df['Primary Type'] == Primary_Type, 'Crime Domain'].iloc[0] if not df.loc[df['Primary Type'] == Primary_Type, 'Crime Domain'].empty else "Violent Crime"
        Crime_Domain_Encoded = le_dict['Crime Domain'].transform([crime_domain_val])[0]

        # Prepare input array (MUST match training feature order!)
        input_data = np.array([[
            Primary_Type_Encoded, 
            Crime_Domain_Encoded,
            Victim_Gender_Encoded, 
            City_Encoded, 
            State_Encoded,
            Year, Month, Day, Hour, DayOfWeek, Latitude, Longitude
        ]])
        
        # Scale and Predict
        input_scaled = scaler.transform(input_data)
        prediction_proba = rf_model.predict_proba(input_scaled)[0][1] 
        prediction = rf_model.predict(input_scaled)[0]
        
        status = "CLOSED" if prediction == 1 else "OPEN"

        return jsonify({
            "status": status,
            "confidence": round(prediction_proba * 100, 2),
            "predicted_class": int(prediction)
        })

    except ValueError as ve:
        return jsonify({"error": f"Prediction failed due to unknown category in input data. Detail: {ve}"}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/api/analysis', methods=['GET'])
def analysis_endpoint():
    """API endpoint to fetch data for dashboard plots and dropdown menus."""
    if df is None:
        return jsonify({"error": "Data not loaded."}), 500

    # 1. Dashboard Metrics and Global Plots Data
    gender_dist = df['Victim Gender'].value_counts().to_dict()
    hourly_trend = df['Hour'].value_counts().sort_index().to_dict()
    top_10_states = df['State'].value_counts().head(10).to_dict()
    top_10_crimes = df['Primary Type'].value_counts().head(10).to_dict()
    metrics_data = model_metrics

    # 2. Interactive Analysis Data (Unique values for dropdowns)
    unique_cities = sorted(df['City'].dropna().unique().tolist())
    unique_primary_types = sorted(df['Primary Type'].dropna().unique().tolist())
    unique_weapons = sorted(df['Weapon Used'].dropna().unique().tolist())
    unique_genders = sorted(df['Victim Gender'].dropna().unique().tolist())
    unique_years = sorted([int(y) for y in df['Year'].dropna().unique().tolist()], reverse=True)
    
    return jsonify({
        "dashboard_data": {
            "gender_dist": gender_dist,
            "hourly_trend": hourly_trend,
            "top_10_states": top_10_states,
            "top_10_crimes": top_10_crimes,
            "model_metrics": metrics_data
        },
        "dropdown_data": {
            "cities": unique_cities,
            "primary_types": unique_primary_types,
            "weapons": unique_weapons,
            "genders": unique_genders,
            "years": unique_years,
        }
    })

@app.route('/api/city_analysis', methods=['POST'])
def city_analysis_endpoint():
    """API endpoint for specific city/year analysis."""
    if df is None:
        return jsonify({"error": "Data not loaded."}), 500
    
    data = request.get_json()
    city = data.get('city', '')
    year = data.get('year')

    if not city or not year:
         return jsonify({"error": "City and Year are required."}), 400

    city_data = df[df['City'] == city]
    
    yearly_city_trend = city_data.groupby('Year').size().to_dict()
    top5_crimes = city_data['Primary Type'].value_counts().head(5).to_dict()
    
    year_data = city_data[city_data['Year'] == year]
    summary = f"No data found for {city} in {year}."
    if not year_data.empty:
        top_crime = year_data['Primary Type'].value_counts().idxmax()
        total_count = year_data['Primary Type'].value_counts().max()
        summary = f"In {city} in {year}, the most frequent crime is **{top_crime}** with {int(total_count)} cases."

    return jsonify({
        "summary": summary,
        "yearly_city_trend": yearly_city_trend,
        "top5_crimes": top5_crimes
    })


# --- Main Execution Block ---
if __name__ == '__main__':
    if load_components():
        print("\n--- Starting Flask Server ---")
        print("Access the dashboard at: http://127.0.0.1:5000/")
        # Note: debug=True is good for development, but remove for production
        app.run(debug=True)
    else:
        print("\n--- Failed to initialize application. Check data file. ---")
