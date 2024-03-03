import pandas as pd
import folium

m = folium.Map(location = [56.47484364114399, -2.990229730364671], zoom_start=5, tiles="cartodb positron")
fg = folium.FeatureGroup(name="My Map")
df = pd.read_csv("./Dataset/charging_station_5.csv")
store_location = df[["Latitude", "Longitude"]].to_numpy()
name = list(df["Location"].values)
facilityType = list(df["FacilityType"].values)
print(facilityType)
colors = ['orange', 'purple', 'cadetblue', 'green']
icons = ["coffee", "location-arrow", "cutlery", "gamepad"]

for idx, cs in enumerate(store_location):
    fg.add_child(folium.Marker(location=[cs[0], cs[1]], popup=name[idx], icon=folium.Icon(icon=icons[facilityType[idx]-1], prefix="fa", color=colors[facilityType[idx]-1])))
m.add_child(fg)

m.save("map.html")