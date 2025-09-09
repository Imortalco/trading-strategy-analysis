import pandas as pd
import os
from datetime import datetime
from lumibot.strategies.strategy import Strategy
from lumibot.backtesting import YahooDataBacktesting

class TimeGPTStrategy(Strategy):

    def initialize(self, path:str):
        # Load TimeGPT forecasts
        self.forecasts = pd.read_csv(path, parse_dates=["time"])
        self.forecasts["time"] = self.forecasts["time"].dt.date 
        self.symbol = "BTC-USD"

    def on_trading_iteration(self):
        # Get current time and match with forecast
        current_time = self.get_datetime().date()
        # Look up forecast for today
        forecast_row = self.forecasts[
            (self.forecasts["time"] == current_time)
        ]

        if forecast_row.empty:
            print("No forecast row !")
            return  # No forecast for today

        forecast_price = forecast_row["TimeGPT"].values[0]
        current_price = self.get_last_price(self.symbol)

        # Calculate % difference
        #pct_diff = (current_price - forecast_price) / forecast_price

        # Get current position
        position = self.get_position(self.symbol)
        current_quantity = position.quantity if position else 0
        
        if current_price <= forecast_price and current_quantity == 0:
            # BUY full position
            print("BUY !")
            quantity = self.portfolio_value // current_price
            order = self.create_order(self.symbol, quantity, "buy")
            self.submit_order(order)

        elif current_price >= forecast_price and current_quantity > 0:
            # SELL full position
            print("SELL !")
            order = self.create_order(self.symbol, position.quantity, "sell")
            self.submit_order(order)

path = "data/forecasts/forecast_three_months_day_by_day"
backtesting_start = datetime(2024,10,1)
backtesting_end = datetime(2024,12,31)
results = TimeGPTStrategy.run_backtest(YahooDataBacktesting, 
                                       backtesting_start, 
                                       backtesting_end,   
                                         parameters={
                                            "path": path 
                                        })
print("Done !")