from time_gpt import get_price_forecast
from news_models.load_news_model import get_sentiment_score

#Configurable parameters
alpha = 0.7 # weight for price forecast 
beta = 0.3 # weight for sentiment score
buy_threshold = 0.02
sell_threshold = -0.02

def get_combined_signal():
    price_forecast = get_price_forecast()
    sentiment_score = get_sentiment_score()