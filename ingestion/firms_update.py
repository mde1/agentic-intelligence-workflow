from dotenv import load_dotenv
import os
import pandas as pd
import requests

load_dotenv()
MAP_KEY = os.getenv("FIRMS_API_KEY")

# This will have to wait - the data is being censored (delayed) by the war

# We can also focus on smaller area ex. South Asia and get last 3 days of records
countries_url = 'https://firms.modaps.eosdis.nasa.gov/api/countries'
df_countries = pd.read_csv(countries_url, sep=';')
df_countries
df_countries.to_csv("data/raw/countries.csv")