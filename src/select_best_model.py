import os
import json
import yaml
import shutil
import logging
import argparse
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Logging setup ---
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger('select_best_model')
logger.setLevel('DEBUG')
log_file_path = os.path.join(log_dir,'select_best_model.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# --- NEW: Evaluation function (copied from train.py) ---
def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, p: int) -> dict:
    """Calculates regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    n = len(y_true)
    # Handle case where n - p - 1 is zero
    if (n - p - 1) == 0:
        adj_r2 = r2 # or np.nan, or just r2
    else:
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2_score': float(r2),
        'adj_r2_score': float(adj_r2)
    }
    return metrics
# --- End of new function ---

def select_model(config_path: str):
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        train_params = config['train']
        select_params = config['select_best']
        split_params = config['data_split'] # NEW
        
        reports_dir = train_params['reports_dir']
        models_dir = train_params['models_dir']
        primary_metric = train_params['primary_metric']
        metric_goal = train_params['metric_goal']
        
        final_model_path = select_params['final_model_path']
        summary_path = select_params['best_model_summary_path']

        # 1. Load metrics from all NEW experiments
        all_metrics = []
        if os.path.exists(reports_dir) and os.listdir(reports_dir):
            for exp_name in os.listdir(reports_dir):
                metric_file = os.path.join(reports_dir, exp_name, 'metrics.json')
                if os.path.exists(metric_file):
                    with open(metric_file, 'r') as f:
                        metrics = json.load(f)
                        metrics['model_name'] = exp_name
                        all_metrics.append(metrics)
        else:
            logger.warning(f"Reports directory is empty or missing: {reports_dir}")
        
        # 2. --- NEW: Evaluate the CURRENT production model ---
        try:
            logger.info(f"Loading new test data from {split_params['test_path']}")
            test_df = pd.read_csv(split_params['test_path'])
            X_test = test_df.drop(split_params['target_col'], axis=1)
            y_test = test_df[split_params['target_col']]
            
            logger.info(f"Loading scaler from {train_params['scaler_path']}")
            with open(train_params['scaler_path'], 'rb') as f:
                scaler = pickle.load(f)
            
            X_test_scaled = scaler.transform(X_test)
            
            logger.info(f"Loading existing model from {final_model_path}")
            with open(final_model_path, 'rb') as f:
                existing_model = pickle.load(f)
                
            logger.info("Re-evaluating existing model on new test data...")
            y_pred = existing_model.predict(X_test_scaled)
            existing_metrics = evaluate_model(y_test, y_pred, p=X_test.shape[1])
            existing_metrics['model_name'] = 'existing_best_model'
            
            all_metrics.append(existing_metrics)
            logger.info(f"Existing model scores: {existing_metrics}")

        except FileNotFoundError:
            logger.warning(f"No existing model found at {final_model_path}. Will select from new models only.")
        except Exception as e:
            logger.error(f"Error re-evaluating existing model: {e}. Will select from new models only.")
        # --- End of new evaluation logic ---

        if not all_metrics:
            logger.error("No metrics found from new experiments or existing model. Aborting.")
            raise FileNotFoundError("No models could be compared.")

        # 3. Compare ALL models (new + existing)
        metrics_df = pd.DataFrame(all_metrics)
        logger.info(f"Comparing all {len(metrics_df)} models:\n{metrics_df}")

        if metric_goal == 'maximize':
            best_model_row = metrics_df.loc[metrics_df[primary_metric].idxmax()]
        elif metric_goal == 'minimize':
            best_model_row = metrics_df.loc[metrics_df[primary_metric].idxmin()]
        else:
            raise ValueError(f"Invalid metric_goal: {metric_goal}. Must be 'maximize' or 'minimize'.")

        best_model_name = best_model_row['model_name']
        best_metrics = best_model_row.to_dict()
        
        logger.info(f"And the winner is: {best_model_name}")
        logger.info(f"Winning metrics: {best_metrics}")

        # 4. --- NEW: Update model *only if* a new one wins ---
        if best_model_name == 'existing_best_model':
            logger.info("The existing model is still the best. No changes will be made.")
        else:
            logger.info(f"New model '{best_model_name}' is better. Promoting to production.")
            best_model_src = os.path.join(models_dir, f"{best_model_name}.pkl")
            os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
            shutil.copy(best_model_src, final_model_path)
            logger.info(f"Best model copied to {final_model_path}")
        # --- End of new promotion logic ---

        # 5. Save summary of the winner
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(best_metrics, f, indent=4)
        logger.info(f"Best model summary saved to {summary_path}")

    except Exception as e:
        logger.error(f"An error occurred during model selection: {e}")
        raise e

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    select_model(config_path=args.config)
