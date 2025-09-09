import requests
import pandas as pd
import time
from dotenv import load_dotenv
import os

load_dotenv()

global_api_rate_limit = .2
cryptopanic_api_key = os.getenv("API_KEY")

def create_url(currencies=None, filter = None, kind=None, region=None, page=None, panic_score=None) -> str:
   
   """
   Creates url for CryptoPanic API reqeust
   
   Args:
      pass :)

   Returns: 
      string: The constructed endpoint with params
   """
   url = 'https://cryptopanic.com/api/v1/posts/?auth_token={}'.format(cryptopanic_api_key)


   if(currencies is not None):
      if(len(currencies.split(',')) <= 50):
         url += "&currencies={}".format(currencies)
      else:
         print("Error: Max currencies is 50 !")
         return
      
   if kind is not None and kind in ['news','media']:
      url += "&kind={}".format(kind)
   
   available_filters = ['rising','hot','bullish','bearish','important','saved','lol']
   if(filter is not None and filter in available_filters):
      url += "&filters={}".format(filter)

   available_regions = ['en', 'de', 'es', 'fr', 'it', 'pt', 'ru']  #(English), (Deutsch), (Español), (Français), (Italiano), (Português), (Русский)
   if(region is not None and region in available_regions):
      url += "&regions={}".format(region)

   if(page is not None):
      url += "&page={}".format(page)

   if(panic_score is not None):
      url += "&panic_score={}".format(panic_score)
   
   return url

def get_single_page(url):
   """
   Get single page

   Args:
      url (string): The endpoint url

   Returns: 
      Json 
   """

   time.sleep(global_api_rate_limit)

   page = requests.get(url)
   data = page.json()
   return data

def get_multiple_pages(page_count, url):
   """
   Gets multiple pages
   
   Args:
      page_count (int): The number of pages we want to get. Each page contains 200 news.
      url (string): The endpoint url

   Returns: 
      List of Pages in Json format
   """
   pages_list_json = [get_single_page(url)]

   for i in range(page_count):
      next_page_url = pages_list_json[i]["next"]
      pages_list_json.append(get_single_page(next_page_url))
   
   return pages_list_json

def get_dataframe(data):
   """
   Converts the list of pages in json into a pandas dataframe for easier processing

   Args:
      data (list of pages in json)
   Returns:
      Pandas DataFrames
   """
   ids = [] 
   titles = []
   urls = []
   dates = []

   try:
      ids = [item["id"] for item in data["results"]]
      titles = [item["title"] for item in data["results"]]
      urls = [item["url"] for item in data["results"]]
      dates = [item["published_at"] for item in data["results"]] 
   except Exception as ex:
      print(f'ERROR: {ex} ')

   df = pd.DataFrame({"id":ids, "title": titles, "url":urls, "published_at": dates})
   df["published_at"] = pd.to_datetime(df["published_at"])
   return df

def concat_pages(pages_list):
   frames = []
   for page in pages_list:
      frames.append(get_dataframe(page))
   
   return pd.concat(frames, ignore_index=True)
   
url = create_url(currencies = 'BTC')
data = get_multiple_pages(9, url) #9 appears to be max
df = concat_pages(data)
df.to_csv('data/crypto_news/crypto_news_urls.csv', index=False)
print("Succesfully got news !")