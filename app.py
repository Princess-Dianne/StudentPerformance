from flask import Flask, request, jsonify, send_from_directory
import pickle
import pandas as pd
from Orange.data.pandas_compat import table_from_frame
from flask_cors import CORS
import os

# Flask setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = BASE_DIR  # all files in this folder
app = Flask(__name__, static_folder=STATIC_DIR)
CORS(app)

# Load the model
MODEL_PATH = os.path.join(STATIC_DIR, "StudentPerformance.pkcls")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Serve the HTML page
@app.route("/")
def home():
    return send_from_directory(STATIC_DIR, "index.html")

# Serve other static files if any
@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json.get("rows", [])
    if not data:
        return jsonify({"error": "No data provided"}), 400

    df = pd.DataFrame(data)
    table = table_from_frame(df)
    raw_pred = model(table)
    class_labels = model.domain.class_var.values
    final_pred = [class_labels[int(c)] for c in raw_pred]

    return jsonify({"prediction": final_pred})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
   