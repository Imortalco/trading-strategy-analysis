# Run this script only to get new data by hours
import cryptocompare
import pandas as pd
import os
#from constants.file_names import *
#from constants.coin_symbols import *

#The end_date should be + 3 hours
#If you want 2025-04-02 22:00, end_date should be 2025-04-03 01:00 💀🔫

end_date = '2024-12-02 02:00' 
currency_symbol = 'USD'
coin_symbol = 'BTC'
file_name = 'btc_data.csv'
print(f"END DATE: {pd.to_datetime(end_date)}")
if not os.path.exists('data/by_hour'):
    os.makedirs('data/by_hour')

coin_data = cryptocompare.get_historical_price_hour(coin=coin_symbol, currency=currency_symbol, limit=24, toTs=pd.to_datetime(end_date))
df = pd.DataFrame(coin_data)
df.dropna(axis=1, inplace=True)
df.drop(['conversionType','conversionSymbol'],axis=1,inplace=True)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.insert(0,'unique_id', coin_symbol)
print(f'File: data/by_hour/{file_name} \n {df} \n')

df.to_csv(f'D:/ML/AITradingBot/data/by_hour/{file_name}', index=False)