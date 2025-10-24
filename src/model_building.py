import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
import yaml
import mlflow
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

#Logs directory

log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

#Log configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


#Load Parameter File
def load_params(params_path: str) -> dict:
    try:
        with open(params_path,'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s',file)
        return params
    except FileNotFoundError:
        logger.error('File Not Found at %s',params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected Error:%s',e)
        raise

#Load Data
def load_csv(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s',file_path)
        return df
    except Exception as e:
        logger.error('Unexpected Error %s',e)
        raise
    except FileNotFoundError:
        logger.debug('File not found at %s',file_path)
        raise
    except pd.errors.ParseError as e:
        logger.debug('Cannot parse file: %s',e)
        raise

#training the model
def train_model(X_train: np.ndarray, y_train: np.ndarray, params):
    try:
        if X_train.shape[0]!=y_train.shape[0]:
            raise ValueError("Dimensions mismatch with X_train and y_train")
        else:
            logger.debug(f"Initializing model parameters from params")
            reg = Ridge()

            logger.debug('Model training starred with %d samples',X_train.shape[0])
            reg.fit(X_train,y_train)
            
            alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
            alphas = list(params["alphas"])
            store_cv_values = params["store_cv_values"]
            ridge_cv_model = RidgeCV(alphas=alphas, store_cv_results=store_cv_values)
            ridge_cv_model.fit(X_train, y_train)

            print(f"Optimal lambda: {ridge_cv_model.alpha_}")
            logger.debug('Training Completed')
            
            return reg
    except ValueError as e:
        logger.error("Value error encountered during model training %s", e)
    except Exception as e:
        logger.error("Unexpected error encountered %s",e)
    
#Save Model
def save_model(model, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        
        with open(file_path,'wb') as file:
            pickle.dump(model,file)
        logger.debug('Model saved to %s',file_path)
    except FileNotFoundError:
        logger.debug('File Not Found at %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected Error %s',e)
        raise

#Main
def main():
	mlflow.set_tracking_uri(f"file://{os.path.abspath('mlruns')}")

	mlflow.set_experiment("Renewable Energy Share Prediction")
logger.debug(f"path: {os.path.abspath('mlruns')}")     
try:
        with mlflow.start_run(run_name='train'):
            params = load_params('params.yaml')['model_building']
            train_data = load_csv('data/preprocessed/global_sustainable_energy_preprocessed2.csv')
            X_train = train_data.iloc[:,:-1].values
            y_train = train_data.iloc[:,-1].values

            clf = train_model(X_train, y_train, params)
            if not os.path.exists('model'):
                os.makedirs('model')
            model_save_path = 'model/model.pkl'
            save_model(clf, model_save_path)
            mlflow.sklearn.log_model(clf, f"{mlflow.active_run().info.run_id}")

except Exception as e:
        logger.error('Failed to build model %s',e)
        print(f'Error{e}')
        

if __name__ == "__main__":
    main()