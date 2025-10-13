import pandas as pd
from scipy.stats import boxcox
import os
import logging
import json
import yaml
import argparse
from sklearn.model_selection import train_test_split

# --- Logging setup ---
log_dir = '../logs'
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')
log_file_path = os.path.join(log_dir,'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def feature_engineering(config_path):
    try:
        # Load parameters from the YAML file
        with open(os.path.join("..", config_path)) as f:
            config = yaml.safe_load(f)

        fe_params = config['feature_engineering']
        split_params = config['data_split']

        input_path = os.path.join("..", fe_params['input_path'])
        lambda_path = os.path.join("..", fe_params['boxcox_lambda_path'])
        arr_col = fe_params['cols_to_transform']
        train_path = os.path.join("..", split_params['train_path'])
        test_path = os.path.join("..", split_params['test_path'])
        test_size = split_params['test_size']
        random_state = split_params['random_state']
        target_col = split_params['target_col']

        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)

        # 1. Applying Box-Cox Transformation
        logger.info("Applying Box-Cox transformation")
        boxcox_lambdas = {}
        for col in arr_col:
            mask = df[col] > 0
            if mask.sum() > 0:
                transformed_values, fitted_lambda = boxcox(df.loc[mask, col])
                df.loc[mask, col] = transformed_values
                boxcox_lambdas[col] = fitted_lambda

        with open(lambda_path, 'w') as f:
            json.dump(boxcox_lambdas, f)
        logger.info(f"Saved Box-Cox lambdas to {lambda_path}")

        # 2. Splitting the data into training and testing sets
        logger.info(f"Splitting data with test_size={test_size}")
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state
        )

        # Create directories for train/test data if they don't exist
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        os.makedirs(os.path.dirname(test_path), exist_ok=True)

        logger.info(f"Saving training data to {train_path}")
        train_df.to_csv(train_path, index=False)

        logger.info(f"Saving testing data to {test_path}")
        test_df.to_csv(test_path, index=False)

        logger.info("Feature engineering and data splitting complete.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise e

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    feature_engineering(config_path=args.config)
