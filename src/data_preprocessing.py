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


#cleans columns names and drops irrelevant columns
def drop_cols(df: pd.DataFrame):  
    #df.columns = df.columns.str.replace('\\n', '\n', regex=True).str.strip()
    return df.drop(['Land Area(Km2)', 'Latitude', 'Longitude'],axis=1)

#converts density and year columns to numeric data type
def fix_types(df: pd.DataFrame):
    df['Density\\n(P/Km2)'] = pd.to_numeric(df['Density\\n(P/Km2)'], errors='coerce')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Year'] = df['Year'].astype(int)
    
    return df


#deals with missing values by grouping records by country or entity
def group_values(df: pd.DataFrame):
    mean_columns = ['Access to electricity (% of population)', 'Access to clean fuels for cooking', 'Primary energy consumption per capita (kWh/person)', 'Energy intensity level of primary energy (MJ/$2017 PPP GDP)', 'gdp_growth', 'gdp_per_capita', 'Density\\n(P/Km2)']
    zero_columns = ['Electricity from fossil fuels (TWh)', 'Electricity from nuclear (TWh)', 'Electricity from renewables (TWh)', 'Renewable energy share in the total final energy consumption (%)', 'Low-carbon electricity (% electricity)', 'Renewables (% equivalent primary energy)', 'Value_co2_emissions_kt_by_country', 'Financial flows to developing countries (US $)', 'Renewable-electricity-generating-capacity-per-capita'] 

    data_filled = df.copy()
    data_filled = data_filled.groupby('Entity').transform(lambda x: x.fillna(x.mean())).combine_first(df)

    data_filled['Entity'] = df['Entity']
    data_filled['Year'] = df['Year']

    for col in mean_columns:
        if col in data_filled.columns:
            data_filled[col] = data_filled[col].fillna(data_filled[col].mean())

    for col in zero_columns:
        if col in data_filled.columns:
            data_filled[col] = data_filled[col].fillna(0)

    return data_filled




#encode the data for the column entities, creating columns of 0/1 type for each unique value in the original columns
def encode_data(df: pd.DataFrame):
    data_encoded = pd.get_dummies(df, columns=['Entity'], drop_first=True)
    data_encoded.drop(['Renewable energy share in the total final energy consumption (%)'],axis=1, inplace=True)
    return data_encoded


def main():
    try:
        data_path = "./data"
        df = pd.read_csv(os.path.join(data_path,"raw/global-data-on-sustainable-energy.csv"))
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
        preprocessed_data_path = os.path.join(data_path, "preprocessed")
        os.makedirs(preprocessed_data_path, exist_ok=True)
        df.to_csv(os.path.join(preprocessed_data_path,"global_sustainable_energy_preprocessed2.csv"),index = False)
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

