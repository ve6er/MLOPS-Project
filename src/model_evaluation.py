import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import yaml
from dvclive import Live
import mlflow

#Logs directory
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

#Logging Configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

#Load Param File
def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML Error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected Error: %s', e)
        raise

#Load Model
def load_model(file_path: str):
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logger.error('Model not found at %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error %s', e)
        raise

#Load Test Data
def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug('Test Data Loaded from %s', file_path)
        return df
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except pd.errors.ParserError as e:
        logger.error('Error reading file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error %s', e)
        raise

#Evaluation of the Ridge Regression model
def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    try:
        y_pred = model.predict(X_test)
        
        # Regression metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Adjusted R2
        n = len(y_test)
        p = X_test.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        metrics_dict = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2_score': float(r2),
            'adj_r2_score': float(adj_r2)
        }
        
        logger.debug('Model Evaluation - MSE: %f, RMSE: %f, MAE: %f, R2: %f, Adj R2: %f', 
                    mse, rmse, mae, r2, adj_r2)
        return metrics_dict
    except Exception as e:
        logger.error('Unexpected error %s', e)
        raise

#Save metrics
def save_metrics(metrics: dict, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Unexpected error occurred %s', e)
        raise

def main():
    mlflow.set_tracking_uri(f"file://{os.path.abspath('mlruns')}")
    mlflow.set_experiment("Sustainable-Energy-Ridge-Regression")
    
    try:
        with mlflow.start_run(run_name='evaluate'):
            # Load parameters and model
            params = load_params(params_path='params.yaml')
            model = load_model('./model/model.pkl')
            
            # Load test data
            test_data = load_data('data/preprocessed/global_sustainable_energy_preprocessed.csv')
            
            # Separate features and target
            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values
            
            # Evaluate model
            metrics = evaluate_model(model, X_test, y_test)
            
            # Log model with MLflow
            mlflow.sklearn.log_model(model, "ridge_regression_model")
            
            # Log metrics with MLflow
            mlflow.log_metrics(metrics)
            
            # Log parameters
            if 'ridge' in params:
                mlflow.log_params(params['ridge'])
            
            # Save metrics to file
            save_metrics(metrics, 'reports/metrics.json')
            mlflow.log_artifact('reports/metrics.json')
            
            logger.info('Model evaluation completed successfully')
            
        # Experiment Tracking Using DVClive
        with Live(save_dvc_exp=True) as live:
            live.log_metric('mse', metrics['mse'])
            live.log_metric('rmse', metrics['rmse'])
            live.log_metric('mae', metrics['mae'])
            live.log_metric('r2_score', metrics['r2_score'])
            live.log_metric('adj_r2_score', metrics['adj_r2_score'])
            live.log_params(params)
              
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        print(f'Error: {e}')
        raise
        
if __name__ == "__main__":
    main()