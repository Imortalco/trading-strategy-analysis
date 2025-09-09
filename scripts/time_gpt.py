import pandas as pd
import os
from dotenv import load_dotenv
from nixtla import NixtlaClient

def get_price_forecast():
    load_dotenv()

    nixtla_client = NixtlaClient(api_key = os.getenv('NIXTLA_API_KEY'))
    data = pd.read_csv(f'data/btc_data.csv')

    fcst_df = nixtla_client.forecast(df=data, 
                            h=31, 
                            freq='D',
                            id_col='unique_id',
                            level=[80,90],
                            model = 'timegpt-1',
                            time_col='time',
                            target_col='close') 

    return fcst_df

if __name__ == "__main__":
    get_price_forecast()