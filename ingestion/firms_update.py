from dotenv import load_dotenv
import os
import pandas as pd
import requests

load_dotenv()
MAP_KEY = os.getenv("FIRMS_API_KEY")

# now let's check how many transactions we have

url = f'https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY={MAP_KEY}'
try:
    response = requests.get(url)
    data = response.json()
    df = pd.Series(data)
    print(df)
except:
  # possible error, wrong MAP_KEY value, check for extra quotes, missing letters
    print("There is an issue with the query. \nTry in your browser: %s" % url)