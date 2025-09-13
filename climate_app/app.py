from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

# -----------------------------
# Helper to convert predictions into text
# -----------------------------
def interpret_weather(predictions):
    """
    Convert model predictions into a human-readable summary.
    predictions: list/array of predicted values (temp, co2, sea, precip, humidity, wind)
    Returns: string summary
    """
    try:
        temp, co2, sea, precip, hum, wind = predictions

        # Temperature category
        if temp < 10:
            temp_text = "Cold"
        elif temp < 25:
            temp_text = "Moderate"
        else:
            temp_text = "Hot"

        # Precipitation category
        if precip < 5:
            precip_text = "Dry"
        elif precip < 20:
            precip_text = "Light Rain"
        else:
            precip_text = "Heavy Rain"

        # Wind category
        if wind < 10:
            wind_text = "Calm wind"
        elif wind < 20:
            wind_text = "Breezy"
        else:
            wind_text = "Windy"

        summary = (
            f"Weather Summary: {temp_text} day with CO₂={co2:.1f}, "
            f"Sea Level={sea:.1f}, {precip_text}, Humidity={hum:.1f}%, {wind_text}."
        )
        return summary
    except Exception:
        return "Could not interpret weather prediction."

# -----------------------------
# Load all models dynamically
# -----------------------------
MODEL_DIR = "models"
models = {}

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

for file in os.listdir(MODEL_DIR):
    if file.endswith(".h5"):
        model_name = file.replace(".h5", "")
        models[model_name] = tf.keras.models.load_model(
            os.path.join(MODEL_DIR, file), compile=False
        )

print("✅ Loaded models:", list(models.keys()))

# Print input shapes for debugging
for name, model in models.items():
    print(f"Model '{name}' input shape: {model.input_shape}")

# -----------------------------
# Home route
# -----------------------------
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# -----------------------------
# Predict route
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # -----------------------------
        # Handle HTML form submission
        # -----------------------------
        if request.form:
            model_name = request.form.get('model_name')
            try:
                features = [
                    float(request.form.get('temperature')),
                    float(request.form.get('co2')),
                    float(request.form.get('sea_level')),
                    float(request.form.get('precipitation')),
                    float(request.form.get('humidity')),
                    float(request.form.get('wind_speed')),
                ]
            except (TypeError, ValueError):
                return render_template(
                    'index.html',
                    prediction=None,
                    model=model_name,
                    error="All feature values must be numbers."
                )
        else:
            # -----------------------------
            # Handle JSON input
            # -----------------------------
            data = request.get_json(force=True, silent=True)
            if not data:
                return jsonify({"error": "Invalid or empty JSON"}), 400

            model_name = data.get("model_name")
            features = data.get("features")

            if not model_name or not features:
                return jsonify({"error": "Missing model_name or features"}), 400

            if not isinstance(features, list) or not all(isinstance(x, (int, float)) for x in features):
                return jsonify({"error": "Features must be a list of numbers"}), 400

        # -----------------------------
        # Check model exists
        # -----------------------------
        if model_name not in models:
            msg = f"Model '{model_name}' not found. Available models: {list(models.keys())}"
            if request.form:
                return render_template('index.html', prediction=None, model=model_name, error=msg)
            return jsonify({"error": msg}), 400

        model = models[model_name]

        # -----------------------------
        # Predict depending on model type
        # -----------------------------
        summary = None

        if model_name in ['lstm_model', 'gru_model', 'rnn_model','ann_model']:
            # Aggregate predictions per input feature
            aggregated_predictions = []
            summaries = []
            for i, value in enumerate(features):
                X_single = np.array([value], dtype=np.float32).reshape(1, 1, 1)
                y_pred = model.predict(X_single)
                pred_list = y_pred.tolist()[0]
                aggregated_predictions.append(pred_list)
                summaries.append(interpret_weather(pred_list))
            predictions = aggregated_predictions
            # Combine all summaries into one string
            summary = " | ".join(summaries)

        else:
            # For ANN or Dense models
            X = np.array(features, dtype=np.float32).reshape(1, -1)
            y_pred = model.predict(X)
            pred_list = y_pred.tolist()[0]
            predictions = pred_list
            summary = interpret_weather(pred_list)

        # -----------------------------
        # Return response
        # -----------------------------
        if request.form:
            return render_template(
                'index.html',
                prediction=predictions,
                summary=summary,
                model=model_name
            )
        return jsonify({
            "model": model_name,
            "input_features": features,
            "predictions": predictions,
            "summary": summary
        })

    except Exception as e:
        msg = str(e)
        if request.form:
            return render_template('index.html', prediction=None, model='unknown', error=msg)
        return jsonify({"error": msg}), 400

# -----------------------------
# Run app
# -----------------------------
if __name__ == '__main__':
    print("Starting Flask app on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
