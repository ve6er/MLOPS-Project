import pandas as pd
import os 
import logging


#Logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

#Setting up logger
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

#Setup console Logger
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

#Adding Logs to File
log_file_path = os.path.join(log_dir,'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path) 
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# Global mean imputation values
imputation_values = {}

#cleans columns names and drops irrelevant columns
def drop_cols(df: pd.DataFrame):  
    return df.drop(['Land Area(Km2)', 'Latitude', 'Longitude'],axis=1)

#converts density and year columns to numeric data type
def fix_types(df: pd.DataFrame):
    # ... (keep existing function) ...
    return df

#deals with missing values by grouping records by country or entity
def group_values(df: pd.DataFrame):
    mean_columns = ['Access to electricity (% of population)', 'Access to clean fuels for cooking', 'Primary energy consumption per capita (kWh/person)', 'Energy intensity level of primary energy (MJ/$2017 PPP GDP)', 'gdp_growth', 'gdp_per_capita', 'Density\\n(P/Km2)']
    zero_columns = ['Electricity from fossil fuels (TWh)', 'Electricity from nuclear (TWh)', 'Electricity from renewables (TWh)', 'Renewable energy share in the total final energy consumption (%)', 'Low-carbon electricity (% electricity)', 'Renewables (% equivalent primary energy)', 'Value_co2_emissions_kt_by_country', 'Financial flows to developing countries (US $)', 'Renewable-electricity-generating-capacity-per-capita'] 

    data_filled = df.copy()
    data_filled = data_filled.groupby('Entity').transform(lambda x: x.fillna(x.mean())).combine_first(df)

    data_filled['Entity'] = df['Entity']
    data_filled['Year'] = df['Year']

    # --- Start modification: Save global means for imputation ---
    global imputer_values
    for col in mean_columns:
        if col in data_filled.columns:
            mean_val = data_filled[col].mean()
            imputer_values[col] = mean_val # Store mean
            data_filled[col] = data_filled[col].fillna(mean_val) # Use stored mean

    for col in zero_columns:
        if col in data_filled.columns:
            imputer_values[col] = 0.0 # Store zero
            data_filled[col] = data_filled[col].fillna(0)
    # --- End modification ---

    return data_filled

#encode the data for the column entities
def encode_data(df: pd.DataFrame):
    data_encoded = pd.get_dummies(df, columns=['Entity'], drop_first=True)
    
    # --- Start modification: Get OHE columns BEFORE dropping target ---
    # We assume 'Renewable energy share...' is the target or near-target
    # app.py needs the full list of features the model was trained on.
    # The feature_engineering script will handle the final target drop.
    # Let's save the columns *after* encoding.
    ohe_columns = data_encoded.columns.tolist()
    # --- End modification ---
    
    data_encoded.drop(['Renewable energy share in the total final energy consumption (%)'],axis=1, inplace=True)
    return data_encoded, ohe_columns # Return columns


def main():
    try:
        # --- Start modification: Load params ---
        with open('params.yaml', 'r') as f:
            params = yaml.safe_load(f)['data_preprocessing']
            
        data_path = os.path.dirname(params['input_path'])
        raw_file = os.path.basename(params['input_path'])
        df = pd.read_csv(os.path.join(data_path, raw_file))
        # --- End modification ---
        logger.debug("raw data is loaded")

        df = drop_cols(df)
        df = fix_types(df)
        logger.debug("dropping columns and modifying data types successful")

        df = group_values(df)
        df, ohe_columns = encode_data(df) # Get columns
        logger.debug("successfully completed preprocessing")

        # --- Start modification: Save outputs based on params.yaml ---
        preprocessed_data_path = params['preprocessed_path']
        imputation_path = params['imputation_path']
        ohe_columns_path = params['ohe_columns_path']

        os.makedirs(os.path.dirname(preprocessed_data_path), exist_ok=True)
        df.to_csv(preprocessed_data_path, index = False)
        
        # Save artifacts
        os.makedirs(os.path.dirname(imputation_path), exist_ok=True)
        with open(imputation_path, 'w') as f:
            json.dump(imputer_values, f, indent=4)
            
        with open(ohe_columns_path, 'w') as f:
            # The final feature list for the model might not include the target
            # Let's assume feature_engineering.py will create the *final* list
            # For now, this list is a good start
            # A better implementation: save OHE columns in feature_engineering
            # But let's follow app.py's apparent logic
            final_cols = [col for col in ohe_columns if col != 'Renewable energy share in the total final energy consumption (%)']
            json.dump(final_cols, f, indent=4)
        # --- End modification ---

    except FileNotFoundError as e:
        logger.error('File Not Found: %s',e)
        raise
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s',e)
        raise
    except Exception as e:
        logger.error('Failed to complete preprocesing: %s',e)
        print(f"Error: %s",e)

if __name__ == "__main__":
    main()