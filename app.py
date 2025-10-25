import os
import joblib
import pandas as pd
import json
from flask import Flask, request, jsonify
import scipy.special as sp 

# --- 1. Create Flask App ---
app = Flask(__name__)

# --- 2. Define Paths ---
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

MODEL_PATH = os.path.join(ROOT_DIR, "model", "model.pkl") 
IMPUTATION_PATH = os.path.join(ROOT_DIR, "imputation_values.json")
OHE_COLUMNS_PATH = os.path.join(ROOT_DIR, "ohe_columns.json")
LAMBDA_PATH = os.path.join(ROOT_DIR, "boxcox_lambdas.json") 

# --- 3. Load Artifacts at Startup ---
model_artifacts = {}
try:
    print(f"--- Loading model from {MODEL_PATH} ---")
    model_artifacts["model"] = joblib.load(MODEL_PATH)
    
    print(f"--- Loading imputation values from {IMPUTATION_PATH} ---")
    with open(IMPUTATION_PATH, 'r') as f:
        model_artifacts["imputation_values"] = json.load(f)
        
    print(f"--- Loading OHE columns from {OHE_COLUMNS_PATH} ---")
    with open(OHE_COLUMNS_PATH, 'r') as f:
        ohe_cols = json.load(f)
        model_artifacts["ohe_columns"] = ohe_cols
        print(f"--- Loaded {len(ohe_cols)} OHE columns ---")

    
    print(f"--- Loading Box-Cox lambdas from {LAMBDA_PATH} ---")
    with open(LAMBDA_PATH, 'r') as f:
        model_artifacts["boxcox_lambdas"] = json.load(f)
        print(f"--- Loaded {len(model_artifacts['boxcox_lambdas'])} lambda values ---")
    

   
    model_features = model_artifacts["model"].n_features_in_
    if len(model_artifacts["ohe_columns"]) != model_features:
        print(f"--- WARNING: Loaded {len(ohe_cols)} columns, but model expects {model_features}. ---")
        

    print("--- All artifacts loaded successfully ---")

except FileNotFoundError as e:
    print(f"--- FATAL ERROR: Missing artifact file. {e} ---")
    print("--- API cannot start without all artifacts. Please run preprocessing and feature engineering. ---")
    model_artifacts = None
except Exception as e:
    print(f"--- FATAL ERROR: Failed to load artifacts. {e} ---")
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

        # --- 3. Replicate Preprocessing ---
        
        # a. Clean column names in the input data
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
                # Box-Cox only works on positive values
                if val > 0:
                    df.at[0, col] = sp.boxcox(val, lmbda)
                else:
                    df.at[0, col] = 0.0
        

        # d. One-Hot Encoding
        df_encoded = pd.get_dummies(df, columns=['Entity'], drop_first=True)
        
        # e. Align columns
        final_columns = model_artifacts["ohe_columns"]
        df_final = df_encoded.reindex(columns=final_columns, fill_value=0)
        
        # --- 4. Make Prediction ---
        prediction = model_artifacts["model"].predict(df_final)
        
        # --- 5. Format and return the prediction ---
        return jsonify({"prediction": prediction[0]})
    
    except Exception as e:
        import traceback
        print(f"--- PREDICTION ERROR: {e} ---")
        print(traceback.format_exc())
        return jsonify({"error": f"Failed to make prediction. {e}"}), 400

if __name__ == "__main__":
    app.run(port=5000)