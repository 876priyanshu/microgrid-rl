import requests, pandas as pd

LAT, LON = 26.471405162085382, 73.11622959032113  # put your campus coordinates here
URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
frames = []
for year in range(2018, 2024):
    r = requests.get(URL, params={
        "start": f"{year}0101", "end": f"{year}1231",
        "latitude": LAT, "longitude": LON, "community": "re",
        "parameters": "ALLSKY_SFC_SW_DWN,T2M",
        "time-standard": "LST", "format": "json"}, timeout=120)
    r.raise_for_status()
    frames.append(pd.DataFrame(r.json()["properties"]["parameter"]))
df = pd.concat(frames).replace(-999, float("nan")).interpolate()
df.to_csv("data/solar_hourly.csv")
print(df.shape)