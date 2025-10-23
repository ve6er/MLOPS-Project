import pandas as pd
import os 
import logging
import json # <-- Added for saving JSON files

#Logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

#Setting up logger
logger = logging.getLogger('updated_data_preprocessing') # <-- Changed name
logger.setLevel('DEBUG')

#Setup console Logger
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

#Adding Logs to File
log_file_path = os.path.join(log_dir,'updated_data_preprocessing.log') # <-- Changed name
file_handler = logging.FileHandler(log_file_path) 
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# --- FIX 1: Added logic to prevent duplicate log messages ---
if logger.hasHandlers():
    logger.handlers.clear()
    
logger.addHandler(console_handler)
logger.addHandler(file_handler)


#cleans columns names and drops irrelevant columns
def drop_cols(df: pd.DataFrame):  
    logger.debug("Cleaning column names...")
    # --- THIS IS THE FINAL, CORRECT FIX ---
    # We need '\\\\n' (four backslashes) to tell the regex engine
    # to find the *literal* text '\n'.
    df.columns = df.columns.str.replace('\\\\n', '', regex=True).str.strip()
    logger.debug("Column names cleaned. 'Density\\n(P/Km2)' is now 'Density(P/Km2)'.")
    
    drop_list = ['Land Area(Km2)', 'Latitude', 'Longitude']
    logger.debug(f"Dropping columns: {drop_list}")
    return df.drop(drop_list, axis=1)

#converts density and year columns to numeric data type
def fix_types(df: pd.DataFrame):
    logger.debug("Fixing data types...")
    # --- BUG FIX a ---
    # The column is now named 'Density(P/Km2)' because drop_cols() fixed it.
    # We must use the *new* name here.
    try:
        df['Density(P/Km2)'] = pd.to_numeric(df['Density(P/Km2)'], errors='coerce')
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df['Year'] = df['Year'].astype(int)
        logger.debug("Converted 'Density(P/Km2)' and 'Year'.")
    except KeyError as e:
        logger.error(f"KeyError in fix_types. This should not happen now. Error: {e}")
        raise
    return df


#deals with missing values by grouping records by country or entity
def group_values(df: pd.DataFrame):
    logger.debug("Starting imputation...")
    
    # --- BUG FIX b ---
    # Must use the *new* column name 'Density(P/Km2)' here as well.
    mean_columns = [
        'Access to electricity (% of population)', 
        'Access to clean fuels for cooking', 
        'Primary energy consumption per capita (kWh/person)', 
        'Energy intensity level of primary energy (MJ/$2017 PPP GDP)', 
        'gdp_growth', 
        'gdp_per_capita', 
        'Density(P/Km2)' # <-- This is the fixed name
    ]
    zero_columns = [
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

    imputation_values = {} # <-- NEW: Create a dict to store imputation values

    data_filled = df.copy()
    
    logger.debug("Applying groupby transform for complex imputation...")
    data_filled = data_filled.groupby('Entity').transform(lambda x: x.fillna(x.mean())).combine_first(df)

    data_filled['Entity'] = df['Entity']
    data_filled['Year'] = df['Year']

    logger.debug("Applying global imputation for remaining NaNs...")
    for col in mean_columns:
        if col in data_filled.columns:
            # --- FIX: Check for NaNs before calculating mean ---
            col_mean = 0
            if not data_filled[col].isnull().all():
                 col_mean = data_filled[col].mean() 
            data_filled[col] = data_filled[col].fillna(col_mean)
            imputation_values[col] = col_mean # <-- NEW: Store the mean

    for col in zero_columns:
        if col in data_filled.columns:
            data_filled[col] = data_filled[col].fillna(0)
            imputation_values[col] = 0 # <-- NEW: Store the zero

    # --- NEW 1: Save the imputation values to a file in the root directory ---
    try:
        # We save this in the *root* folder (../)
        imputation_path = os.path.join(os.path.dirname(__file__), '..', 'imputation_values.json')
        with open(imputation_path, 'w') as f:
            json.dump(imputation_values, f, indent=4)
        logger.info(f"Saved imputation values to {imputation_path}")
    except Exception as e:
        logger.error(f"Failed to save imputation values: {e}")
    # --- End of new code ---
    
    logger.debug("Imputation complete.")
    return data_filled


#encode the data for the column entities, creating columns of 0/1 type for each unique value in the original columns
def encode_data(df: pd.DataFrame):
    logger.debug("Applying One-Hot Encoding...")
    data_encoded = pd.get_dummies(df, columns=['Entity'], drop_first=True)
    
    logger.info(f"Shape *before* dropping column: {data_encoded.shape}")

    logger.debug("Dropping target-related column...")
    data_encoded.drop(['Renewable energy share in the total final energy consumption (%)'],axis=1, inplace=True)
    
    logger.info(f"Shape *after* dropping column: {data_encoded.shape}") 
    
    # --- NEW 2: Save the final OHE column names ---
    try:
        ohe_cols_path = os.path.join(os.path.dirname(__file__), '..', 'ohe_columns.json')
        
        # Get all 191 columns
        all_columns = data_encoded.columns.tolist() 
        
        # --- THIS IS THE FIX ---
        # We must remove the target variable from the list
        # This list should only contain the FEATURES (X)
        TARGET_VARIABLE = 'Renewable-electricity-generating-capacity-per-capita'
        
        if TARGET_VARIABLE in all_columns:
            logger.info(f"Removing target variable '{TARGET_VARIABLE}' from OHE list.")
            all_columns.remove(TARGET_VARIABLE)
        else:
            logger.warning(f"Target variable '{TARGET_VARIABLE}' not found in column list!")
        
        ohe_columns_features_only = all_columns # This list now has 190 features
        # --- END OF FIX ---
        
        logger.info(f"Final feature list length is {len(ohe_columns_features_only)}")
        
        with open(ohe_cols_path, 'w') as f:
            json.dump(ohe_columns_features_only, f, indent=4)
            
        logger.info(f"Saved {len(ohe_columns_features_only)} OHE column names to {ohe_cols_path}") # This MUST log 190
    except Exception as e:
        logger.error(f"Failed to save OHE columns: {e}")
    # --- End of new code ---

    return data_encoded


def main():
    try:
        # --- Make paths relative to the script location (more robust) ---
        SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
        ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
        
        data_path = os.path.join(ROOT_DIR, "data")
        raw_data_file = os.path.join(data_path, "raw", "global-data-on-sustainable-energy.csv")

        
        df = pd.read_csv(raw_data_file)
        logger.debug("raw data is loaded")

        #drop the unnecessary columns and clean column names along with data types
        df = drop_cols(df)
        df = fix_types(df)
        logger.debug("dropping columns and modifying data types successful")

        #removing null values and removing outliers
        df = group_values(df)
        df = encode_data(df)
        logger.debug("successfully completed preprocessing")

        #saving the preprocessed data
        preprocessed_data_path = os.path.join(data_path, "processed")
        os.makedirs(preprocessed_data_path, exist_ok=True)
        
        output_csv = os.path.join(preprocessed_data_path,"global_sustainable_energy_preprocessed.csv")
        df.to_csv(output_csv, index = False)
        logger.info(f"Saved preprocessed data to {output_csv}")
        
    except FileNotFoundError as e:
        logger.error('File Not Found: %s',e)
        raise
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s',e)
        raise
    except Exception as e:
        logger.error('Failed to complete preprocesing: %s',e)
        print(f"Error: {e}") # <-- Print the actual error
        raise 

if __name__ == "__main__":
    main()

