import cryptocompare
import pandas as pd
import os

def get_crypto_data(end_date, currency_symbol, coin_symbol):
    if not os.path.exists('data'):
        os.makedirs('data')

    coin_data = cryptocompare.get_historical_price_day(
        coin=coin_symbol,
        currency=currency_symbol,
        limit=2000,
        toTs=pd.to_datetime(end_date)
    )

    df = pd.DataFrame(coin_data)

    df.dropna(axis=1, inplace=True)
    df.drop(['conversionType', 'conversionSymbol'], axis=1, inplace=True)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.insert(0, 'unique_id', coin_symbol)

    print(f'File: {"data/" + coin_symbol.lower() + "_data.csv"} \n{df}\n')

    df.to_csv(f'data/{coin_symbol.lower()}_data.csv', index=False)
    
if __name__ == "__main__":
    end_date = '2025-01-01'
    currency_symbol = 'USD'
    coin_symbol = 'BTC'

    get_crypto_data(end_date, currency_symbol, coin_symbol)
