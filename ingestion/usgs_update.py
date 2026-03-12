import requests
import pandas as pd

url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/1.0_day.geojson"

data = requests.get(url).json()

rows = []

for feature in data["features"]:
    
    props = feature["properties"]
    coords = feature["geometry"]["coordinates"]
    
    row = {
        "longitude": coords[0],
        "latitude": coords[1],
        "magnitude": props["mag"],
        "title": props["title"],
        "location": props["place"],
        "type": props["type"],
        "time": pd.to_datetime(props["time"], unit="ms")
    }
    
    rows.append(row)

df = pd.DataFrame(rows)

print(df.head())