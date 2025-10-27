import pandas as pd
import os 
import logging
import json
import yaml

# --- Logging setup ---
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')
log_file_path = os.path.join(log_dir,'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
# --- End Logging ---

# --- Column Definitions ---
# Define columns centrally so fix_types and group_values can share them
MEAN_COLUMNS = [
    'Access to electricity (% of population)', 
    'Access to clean fuels for cooking', 
    'Primary energy consumption per capita (kWh/person)', 
    'Energy intensity level of primary energy (MJ/$2017 PPP GDP)', 
    'gdp_growth', 
    'gdp_per_capita', 
    'Density\\n(P/Km2)'
]
ZERO_COLUMNS = [
    'Electricity from fossil fuels (TWh)', 
    'Electricity from nuclear (TWh)', 
    'Electricity from renewables (TWh)', 
    'Renewable energy share in the total final energy consumption (%)', 
    'Low-carbon electricity (% electricity)', 
    'Renewables (% equivalent primary energy)', 
    'Value_co2_emissions_kt_by_country', 
    'Financial flows to developing countries (US $)', 
    'Renewable-electricity-generating-capacity-per-capita'
]

# Global dictionary to store imputation values
imputer_values = {}

# --- Preprocessing Functions ---

def drop_cols(df: pd.DataFrame):  
    logger.debug("Dropping irrelevant columns")
    return df.drop(['Land Area(Km2)', 'Latitude', 'Longitude'], axis=1, errors='ignore')

def fix_types(df: pd.DataFrame):
    logger.debug("Starting type conversion")
    
    # Combine all columns that should be numeric
    numeric_cols = MEAN_COLUMNS + ZERO_COLUMNS
    
    for col in numeric_cols:
        if col in df.columns:
            # Use errors='coerce' to turn bad strings (like the one in the log) into NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle 'Year' specifically
    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        # We must fill NaNs before converting to integer
        # Using 0 as a placeholder for any missing/bad years
        df['Year'] = df['Year'].fillna(0).astype(int)
    
    logger.debug("Type conversion finished")
    return df

def group_values(df: pd.DataFrame):
    logger.debug("Starting grouping and imputation")
    global imputer_values # Ensure we're modifying the global dict

    data_filled = df.copy()
    
    # This line will now work, as all columns are numeric (or NaN)
    data_filled = data_filled.groupby('Entity').transform(lambda x: x.fillna(x.mean(numeric_only=True))).combine_first(df)

    data_filled['Entity'] = df['Entity']
    data_filled['Year'] = df['Year']

    # Impute any remaining NaNs (e.g., for entities with no data) using the global mean
    for col in MEAN_COLUMNS:
        if col in data_filled.columns:
            mean_val = data_filled[col].mean()
            imputer_values[col] = mean_val # Store global mean for API
            data_filled[col] = data_filled[col].fillna(mean_val)

    for col in ZERO_COLUMNS:
        if col in data_filled.columns:
            imputer_values[col] = 0.0 # Store zero for API
            data_filled[col] = data_filled[col].fillna(0)

    logger.debug("Imputation finished")
    return data_filled

def encode_data(df: pd.DataFrame):
    logger.debug("Starting one-hot encoding")
    data_encoded = pd.get_dummies(df, columns=['Entity'], drop_first=True)
    
    # Get all column names *after* encoding
    ohe_columns = data_encoded.columns.tolist()
    
    # Target column from your 'feature_engineering.py'
    target_col = 'Renewables (% equivalent primary energy)' 
    
    # Target column from your original script
    other_target_col = 'Renewable energy share in the total final energy consumption (%)'

    if target_col in data_encoded.columns:
        # We don't drop it from the *data* here, we let feature_engineering.py handle that
        # But we MUST remove it from the *list of features* for the API
        if target_col in ohe_columns:
            ohe_columns.remove(target_col)
            logger.debug(f"Removed target '{target_col}' from OHE column list.")
            
    if other_target_col in data_encoded.columns:
        # This one *was* being dropped from the data, so we continue that
        data_encoded.drop([other_target_col], axis=1, inplace=True)
        if other_target_col in ohe_columns:
            ohe_columns.remove(other_target_col)
            logger.debug(f"Dropped and removed '{other_target_col}' from OHE column list.")
            
    logger.debug("Encoding finished")
    return data_encoded, ohe_columns

# --- Main Execution ---

def main():
    try:
        with open('params.yaml', 'r') as f:
            params = yaml.safe_load(f)['data_preprocessing']
            
        data_path = os.path.dirname(params['input_path'])
        raw_file = os.path.basename(params['input_path'])
        df = pd.read_csv(os.path.join(data_path, raw_file))
        logger.debug("raw data is loaded")

        #drop the unnecessary columns and clean column names along with data types
        df = drop_cols(df)
        df = fix_types(df) # This function is now fixed
        logger.debug("dropping columns and modifying data types successful")

        #removing null values and removing outliers
        df = group_values(df) # This function will no longer crash
        df, ohe_columns = encode_data(df)
        logger.debug("successfully completed preprocessing")

        #saving the preprocessed data
        preprocessed_data_path = params['output_path']
        imputation_path = params['imputation_path']
        ohe_columns_path = params['ohe_columns_path']

        os.makedirs(os.path.dirname(preprocessed_data_path), exist_ok=True)
        df.to_csv(preprocessed_data_path, index = False)
        
        # Save artifacts
        os.makedirs(os.path.dirname(imputation_path), exist_ok=True)
        with open(imputation_path, 'w') as f:
            json.dump(imputer_values, f, indent=4)
        logger.debug(f"Imputation values saved to {imputation_path}")
            
        with open(ohe_columns_path, 'w') as f:
            json.dump(ohe_columns, f, indent=4)
        logger.debug(f"OHE columns saved to {ohe_columns_path}")

    except FileNotFoundError as e:
        logger.error('File Not Found: %s',e)
        raise
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s',e)
        raise
    except Exception as e:
        logger.error('Failed to complete preprocesing: %s',e)
        print(f"Error: %s",e)
        raise

if __name__ == "__main__":
    main()