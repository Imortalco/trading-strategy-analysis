import scrapy
import pandas as pd
from constants import file_names
from bs4 import BeautifulSoup
from scrapy_selenium import SeleniumRequest

class CryptoPanicSpider(scrapy.Spider):
    name = "CryptoPanicNews"

    def start_requests(self):
        currency_id = 91 #BTC currency id
        df = pd.read_csv("../data/crypto_news/cryptopanic_news.csv")
        news_currency_df = pd.read_csv('../data/crypto_news/news__currency.csv')

        #Filtering by date
        df['newsDatetime'] = pd.to_datetime(df['newsDatetime'])
        df = df[(df['newsDatetime']>='2024-12-01') & (df['newsDatetime']<='2024-12-31')]

        #Filtering by currency
        bitcoin_news_id = news_currency_df[news_currency_df['currencyId'] == currency_id]['newsId']
        filtered_news_df = df[df['id'].isin(bitcoin_news_id)]
        print(f"FILTERED NEWS: \n {filtered_news_df.head()}")

        for index, row in filtered_news_df.iterrows():
            print(f"Sending request for article ID {row['id']} published at {row['newsDatetime']}")
            yield SeleniumRequest(url=row['url'], callback=self._parse, wait_time=30,
                                  meta = {'id': row['id'], 'published_at': row['newsDatetime']})

    def _parse(self, response, **kwargs):
        #Extracting from meta data
        _id = response.meta['id']
        published_at = response.meta['published_at']

        soup = BeautifulSoup(response.text, "html.parser")

        details_pane = soup.find("div", id="detail_pane")

        #Find elemets
        article_title_span = details_pane.find("span", class_="text")
        description_body_div = details_pane.find("div", class_="description-body")

        article_title_text = article_title_span.text.strip() if article_title_span else "No title found !"
        description_text = description_body_div.text.strip() if description_body_div else "" # Return empty string in case of the description being a img

        concatenated_text = f"{article_title_text}{description_text}"

        yield {"id": _id,
               "news": concatenated_text,
               "published_at": published_at}

