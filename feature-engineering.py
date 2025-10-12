import pandas as pd
from scipy.stats import boxcox
import os
import logging
import json

#Logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

#Setting up logger
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

#Setup console Logger
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

#Adding Logs to File
log_file_path = os.path.join(log_dir,'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def feature_engineering(input_path, output_path):
    """
    This function takes the preprocessed data, applies feature engineering techniques,
    and saves the resulting data.
    """
    try:
        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)

        # Columns to apply box-cox transformation
        arr_col = [
            'Electricity from fossil fuels (TWh)',
            'Electricity from nuclear (TWh)',
            'Electricity from renewables (TWh)',
            'Financial flows to developing countries (US $)',
            'Primary energy consumption per capita (kWh/person)',
            'Renewable-electricity-generating-capacity-per-capita',
            'Renewables (% equivalent primary energy)',
            'Value_co2_emissions_kt_by_country',
            'gdp_growth',
            'gdp_per_capita'
        ]

        logger.info("Applying Box-Cox transformation")
        boxcox_lambdas = {}
        for col in arr_col:
            # Applying boxcox only to positive values
            mask = df[col] > 0
            if mask.sum() > 0:
                transformed_values, fitted_lambda = boxcox(df.loc[mask, col])
                df.loc[mask, col] = transformed_values
                boxcox_lambdas[col] = fitted_lambda

        # Save the lambdas to a file
        with open('boxcox_lambdas.json', 'w') as f:
            json.dump(boxcox_lambdas, f)

        logger.info(f"Saving feature engineered data to {output_path}")
        df.to_csv(output_path, index=False)
        logger.info("Feature engineering complete.")

    except Exception as e:
        logger.error(f"An error occurred during feature engineering: {e}")
        raise e

if __name__ == '__main__':
    # This assumes the preprocessed data is saved in 'data/preprocessed_data.csv'
    # and the output should be saved in 'data/featured_data.csv'
    if not os.path.exists('data'):
        os.makedirs('data')
    feature_engineering('data/preprocessed_data.csv', 'data/featured_data.csv')