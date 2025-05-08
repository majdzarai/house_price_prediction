from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define API Key
API_KEY_1 = "your-secret-api-key-123456"  # 🔐 Change this to a secure key!

# Input limits
limits = {
    'bedrooms': {'min': 0, 'max': 20},
    'bathrooms': {'min': 0, 'max': 10},
    'sqft_living': {'min': 200, 'max': 10000},
    'sqft_lot': {'min': 200, 'max': 100000},
    'floors': {'min': 1, 'max': 4},
    'waterfront': {'min': 0, 'max': 1},
    'view': {'min': 0, 'max': 4},
    'condition': {'min': 1, 'max': 5},
    'grade': {'min': 1, 'max': 13},
    'sqft_above': {'min': 0, 'max': 10000},
    'sqft_basement': {'min': 0, 'max': 5000},
    'sqft_living15': {'min': 200, 'max': 10000},
    'sqft_lot15': {'min': 200, 'max': 100000},
    'lat': {'min': 47.0, 'max': 47.8},
    'long': {'min': -122.5, 'max': -121.0},
    'house_age': {'min': 0, 'max': 200},
    'years_since_renovated': {'min': 0, 'max': 100}
}

@app.route('/')
def home():
    return render_template('index.html', limits=limits)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = []
        for key in limits.keys():
            value = float(request.form[key])
            if not (limits[key]['min'] <= value <= limits[key]['max']):
                error = f"❌ Erreur : la valeur de '{key}' doit être entre {limits[key]['min']} et {limits[key]['max']}."
                return render_template("index.html", limits=limits, error=error)
            features.append(value)

        features_scaled = scaler.transform([features])
        log_prediction = model.predict(features_scaled)
        prediction = np.expm1(log_prediction[0])

        return render_template('result.html', prediction=round(prediction, 2))

    except Exception as e:
        return render_template("index.html", limits=limits, error=f"Erreur : {str(e)}")

# 🔐 API endpoint with API Key authentication
@app.route('/api/predict', methods=['POST'])
def api_predict():
    api_key = request.headers.get("x-api-key")
    if api_key != API_KEY_1:

        return jsonify({"error": "Unauthorized - Invalid API Key"}), 401

    try:
        data = request.json
        if not data:
            return jsonify({"error": "Missing JSON data"}), 400

        features = []
        for key in limits.keys():
            if key not in data:
                return jsonify({"error": f"Missing key: {key}"}), 400
            value = float(data[key])
            if not (limits[key]['min'] <= value <= limits[key]['max']):
                return jsonify({"error": f"Value for '{key}' out of bounds."}), 400
            features.append(value)

        features_scaled = scaler.transform([features])
        log_prediction = model.predict(features_scaled)
        prediction = np.expm1(log_prediction[0])

        return jsonify({"prediction": float(round(prediction, 2))})


    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
