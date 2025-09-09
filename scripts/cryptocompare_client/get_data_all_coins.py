# Run this script only to get new training data
import cryptocompare
import pandas as pd
import os
from constants.file_names import *
from constants.coin_symbols import *

end_date = '2024-01-01'
currency_symbol = 'USD'

if not os.path.exists(CSV_FOLDER):
    os.makedirs(CSV_FOLDER)

df = pd.DataFrame()
df_list = []


#TO DO: maybe it would be better to try with a bigger data set than 2000 days ?? 
for coin_symbol, file_path in zip(COIN_SYMBOLS, FILE_PATHS):
    coin_data = cryptocompare.get_historical_price_day(coin=coin_symbol, currency=currency_symbol, limit=2000, toTs=pd.to_datetime(end_date))
    temp_df = pd.DataFrame(coin_data)
    temp_df.dropna(axis=1, inplace=True)
    temp_df.drop(['conversionType','conversionSymbol'],axis=1,inplace=True)
    temp_df.insert(0,'unique_id', coin_symbol)
    temp_df['time'] = pd.to_datetime(temp_df['time'], unit='s')
    print(f'File: {file_path} \n {temp_df} \n')
    df = df_list.append(temp_df)


df = pd.concat(df_list, ignore_index=True)
df.reset_index(drop=True, inplace=True) 

# column_to_move = df.pop('unique_id')
# df.insert(0,'unique_id', column_to_move)
df.to_csv(f'{CSV_FOLDER}/coin_data', index=False)





