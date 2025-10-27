import os
import numpy as np
import pandas as pd
import pickle
import json
import yaml
import logging
import argparse
from dvclive import Live
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

# --- Logging setup ---
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger('train')
logger.setLevel('DEBUG')
log_file_path = os.path.join(log_dir,'train.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def load_data(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"Error loading data from {path}: {e}")
        raise

def get_model(model_type: str, params: dict):
    """Factory function to instantiate a model from its name and params."""
    if model_type == 'Ridge':
        return Ridge(**params)
    if model_type == 'Lasso':
        return Lasso(**params)
    if model_type == 'RandomForestRegressor':
        return RandomForestRegressor(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, p: int) -> dict:
    """Calculates regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    n = len(y_true)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2_score': float(r2),
        'adj_r2_score': float(adj_r2)
    }
    return metrics

def train(config_path: str):
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        split_params = config['data_split']
        train_params = config['train']

        logger.info("Loading training and testing data")
        train_df = load_data(split_params['train_path'])
        test_df = load_data(split_params['test_path'])

        target_col = split_params['target_col']
        X_train = train_df.drop(target_col, axis=1)
        y_train = train_df[target_col]
        X_test = test_df.drop(target_col, axis=1)
        y_test = test_df[target_col]

        logger.info("Scaling data")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save the scaler for app.py
        scaler_path = train_params['scaler_path']
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"Scaler saved to {scaler_path}")
        
        # Create output dirs
        os.makedirs(train_params['models_dir'], exist_ok=True)
        os.makedirs(train_params['reports_dir'], exist_ok=True)

        logger.info("Starting model experiments...")
        for exp_name, exp_config in train_params['experiments'].items():
            logger.info(f"--- Running experiment: {exp_name} ---")
            
            with Live(dir=os.path.join(train_params['reports_dir'], exp_name), save_dvc_exp=True) as live:
                model_type = exp_config['type']
                model_params = exp_config['params']
                
                live.log_params(model_params)
                
                logger.info(f"Training {model_type}...")
                model = get_model(model_type, model_params)
                model.fit(X_train_scaled, y_train)
                
                logger.info("Evaluating model...")
                y_pred = model.predict(X_test_scaled)
                metrics = evaluate_model(y_test, y_pred, p=X_test.shape[1])
                
                for metric_name, value in metrics.items():
                    live.log_metric(metric_name, value)
                logger.info(f"Metrics: {metrics}")

                # Save the model
                model_path = os.path.join(train_params['models_dir'], f"{exp_name}.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"Model saved to {model_path}")
        
        logger.info("All experiments complete.")

    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        raise e

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    train(config_path=args.config)