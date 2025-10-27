import os
import joblib
import pandas as pd
import json
import yaml # Import yaml
from flask import Flask, request, jsonify
import scipy.special as sp 
import logging

# --- 1. Create Flask App ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# --- 2. Define Paths from params.yaml ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    with open(os.path.join(ROOT_DIR, "params.yaml"), 'r') as f:
        params = yaml.safe_load(f)
    
    MODEL_PATH = os.path.join(ROOT_DIR, params['select_best']['final_model_path'])
    SCALER_PATH = os.path.join(ROOT_DIR, params['train']['scaler_path'])
    IMPUTATION_PATH = os.path.join(ROOT_DIR, params['data_preprocessing']['imputation_path'])
    OHE_COLUMNS_PATH = os.path.join(ROOT_DIR, params['data_preprocessing']['ohe_columns_path'])
    LAMBDA_PATH = os.path.join(ROOT_DIR, params['feature_engineering']['boxcox_lambda_path'])

except Exception as e:
    app.logger.error(f"--- FATAL: Could not load params.yaml. {e} ---")
    params = None

# --- 3. Load Artifacts at Startup ---
model_artifacts = {}
try:
    if params is None:
        raise FileNotFoundError("params.yaml not found or failed to load.")

    app.logger.info(f"--- Loading model from {MODEL_PATH} ---")
    model_artifacts["model"] = joblib.load(MODEL_PATH)
    
    app.logger.info(f"--- Loading scaler from {SCALER_PATH} ---")
    model_artifacts["scaler"] = joblib.load(SCALER_PATH)
    
    app.logger.info(f"--- Loading imputation values from {IMPUTATION_PATH} ---")
    with open(IMPUTATION_PATH, 'r') as f:
        model_artifacts["imputation_values"] = json.load(f)
        
    app.logger.info(f"--- Loading OHE columns from {OHE_COLUMNS_PATH} ---")
    with open(OHE_COLUMNS_PATH, 'r') as f:
        ohe_cols = json.load(f)
        model_artifacts["ohe_columns"] = ohe_cols
        app.logger.info(f"--- Loaded {len(ohe_cols)} OHE columns ---")
    
    app.logger.info(f"--- Loading Box-Cox lambdas from {LAMBDA_PATH} ---")
    with open(LAMBDA_PATH, 'r') as f:
        model_artifacts["boxcox_lambdas"] = json.load(f)
        app.logger.info(f"--- Loaded {len(model_artifacts['boxcox_lambdas'])} lambda values ---")
    
    # Check model feature count against OHE columns
    model_features = model_artifacts["model"].n_features_in_
    if len(model_artifacts["ohe_columns"]) != model_features:
        app.logger.warning(f"--- WARNING: Loaded {len(ohe_cols)} columns, but model expects {model_features}. ---")
        
    app.logger.info("--- All artifacts loaded successfully ---")

except Exception as e:
    app.logger.error(f"--- FATAL ERROR: Failed to load artifacts. {e} ---")
    app.logger.error("--- API cannot start without all artifacts. Run the DVC pipeline. ---")
    model_artifacts = None

# --- 4. Define Endpoints ---
@app.route("/", methods=["GET"])
def read_root():
    if model_artifacts:
        return jsonify({"status": "ok", "message": "API is running and artifacts are loaded. POST to /predict."})
    else:
        return jsonify({"status": "error", "message": "API is down. Artifacts failed to load. Check server logs."}), 500

@app.route("/predict", methods=["POST"])
def predict():
    if not model_artifacts:
        return jsonify({"error": "Model artifacts not loaded. Check server logs."}), 500

    try:
        data_json = request.json
        if not data_json:
            return jsonify({"error": "No input JSON received."}), 400

        # --- 3. Replicate Preprocessing & Feature Engineering ---
        
        # a. Clean column names (app.py had this, good)
        cleaned_data = {}
        for key, value in data_json.items():
            cleaned_key = key.replace('\n', '').strip()
            cleaned_data[cleaned_key] = value

        df = pd.DataFrame([cleaned_data])
        
        # b. Imputation
        imputation_map = model_artifacts["imputation_values"]
        df.fillna(imputation_map, inplace=True)
        
        # --- c. Box-Cox Transformation ---
        lambda_map = model_artifacts["boxcox_lambdas"]
        for col, lmbda in lambda_map.items():
            if col in df.columns:
                val = df.at[0, col]
                if val > 0:
                    df.at[0, col] = sp.boxcox(val, lmbda)
                else:
                    df.at[0, col] = 0.0 # Match transformation logic
        
        # d. One-Hot Encoding
        df_encoded = pd.get_dummies(df, columns=['Entity'], drop_first=True)
        
        # e. Align columns
        final_columns = model_artifacts["ohe_columns"]
        df_aligned = df_encoded.reindex(columns=final_columns, fill_value=0)
        
        # --- f. Scaling (NEW STEP) ---
        df_scaled = model_artifacts["scaler"].transform(df_aligned)
        
        # --- 4. Make Prediction ---
        prediction = model_artifacts["model"].predict(df_scaled)
        
        return jsonify({"prediction": prediction[0]})
    
    except Exception as e:
        import traceback
        app.logger.error(f"--- PREDICTION ERROR: {e} ---")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": f"Failed to make prediction. {e}"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))