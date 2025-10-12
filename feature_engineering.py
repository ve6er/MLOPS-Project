import pandas as pd
from scipy.stats import boxcox
import os
import logging
import json
import yaml
import argparse

# --- Logging setup ---
log_dir = '../logs' # Correct path to save logs
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
        # Go up one directory to find the config file
        with open(os.path.join("..", config_path)) as f:
            config = yaml.safe_load(f)

        params = config['feature_engineering']
        # Adjust paths to be relative to the main project folder
        input_path = os.path.join("..", params['input_path'])
        output_path = os.path.join("..", params['output_path'])
        lambda_path = os.path.join("..", params['boxcox_lambda_path'])
        arr_col = params['cols_to_transform']

        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)

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

        logger.info(f"Saving feature engineered data to {output_path}")
        df.to_csv(output_path, index=False)
        logger.info("Feature engineering complete.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise e

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    feature_engineering(config_path=args.config)
