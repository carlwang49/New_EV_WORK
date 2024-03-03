
from datetime import datetime, timedelta
from dateutil import parser, zoneinfo, tz
from collections import defaultdict
from copy import deepcopy
from operator import itemgetter
import pandas as pd
import numpy as np
from queue import Queue
import math
import statistics
import random
import pytz
import matplotlib.pyplot as plt
import seaborn as sns
from haversine import haversine, Unit

sns.set()
sns.set_style('whitegrid')

# Ignore harmless warnings 
import warnings 

warnings.filterwarnings("ignore") 
charging_speed = 10
charging_fee = 0.3  
parking_slots_num = 10
ALPHA = 0.5

building_start_time = parser.parse("2018-07-01")
building_end_time = parser.parse("2018-07-08")
charging_start_time = parser.parse("2018-07-01")
charging_end_time = parser.parse("2018-07-08")

location = pd.read_csv("./Dataset/location_5.csv", index_col="locationId")
location.index = location.index.astype(str) # 將索引轉換為字串型態
location = location.sort_values(by="buildingID") # 以 "buildingID" 這個欄位為鍵進行排序。
location["buildingID"] = location["buildingID"].astype(str) # 將 "buildingID" 這個欄位轉換為字串型態。

location.loc["50911", "contractCapacity"] = 222 #13
location.loc["50266", "contractCapacity"] = 273 #10


building_data = pd.read_csv("./Dataset/generation_with_consumption_3.csv", index_col=0)
building_LSTM_data = pd.read_csv("./Result/predict_building_data.csv", index_col=0)
charging_data = pd.read_csv("./Dataset/charging_data_2_move.csv")

building_data["datetime"] = pd.to_datetime(building_data["datetime"], format="%Y-%m-%d %H:%M:%S")
building_data["buildingID"] = building_data["buildingID"].astype(str)

building_LSTM_data["datetime"] = pd.to_datetime(building_LSTM_data["datetime"], format="%Y-%m-%d %H:%M:%S")
building_LSTM_data["buildingID"] = building_LSTM_data["buildingID"].astype(str)

charging_data["createdNew"] = pd.to_datetime(charging_data["createdNew"], format="%Y-%m-%d %H:%M:%S")
charging_data["locationId"] = charging_data["locationId"].astype(str)
charging_data["userId"] = charging_data["userId"].astype(str)

building_data.loc[building_data["buildingID"] == "13", "contract_capacity"] = 222
building_data.loc[building_data["buildingID"] == "10", "contract_capacity"] = 273

columns = [
    "datetime", 
    "date", 
    "userID", 
    "locationID", 
    "buildingID", 
    "chargingLen", 
    "originLocationID", 
    "originChargingHour", 
]

def get_user_list(date):
    temp = charging_data[charging_data["createdNew"] < date]
    user_list = temp.groupby("userId").groups.keys()
    return user_list

def get_charging_request(charging_data, date):
    
    request_df = charging_data[(charging_data["createdNew"] >= date) & ((charging_data["createdNew"] < (date + timedelta(days=1))))]
    request_df["chargingHour"] = request_df["kwhNew"].apply(lambda x: x / charging_speed)
    charging_request = list()
 
    for item in request_df.iterrows():
        request = list(item[1][["_id", "userId", "chargingHour", "createdHour", "locationId"]])
        charging_request.append(request)

    return charging_request

schedule_df = pd.DataFrame([], columns=columns)
testing_start_time = charging_start_time
null_value = 0

for day in range(7):

    charging_request = get_charging_request(charging_data, testing_start_time)
    recommend = pd.read_csv(f"./Result/Baseline/UserHistoryPreference/baseline_pre/alpha_0.5/{testing_start_time.strftime('%m%d')}.csv")
    recommend["datetime"] = pd.to_datetime(recommend["datetime"], format="%Y-%m-%d %H:%M:%S")
    
    for request in charging_request:
        try:
            schedule = recommend[recommend["requestID"] == request[0]]
            schedule_df.loc[len(schedule_df)] = [
                schedule["datetime"].values[0],
                testing_start_time.strftime("%m%d"),
                schedule["userID"].values[0],
                str(schedule["locationID"].values[0]),
                location.loc[str(schedule["locationID"].values[0]), "buildingID"],
                schedule["chargingLen"].values[0],
                schedule["originLocationID"].values[0],
                schedule["originHour"].values[0],
            ]
        except Exception as e:
            print(request, e)
            null_value += 1

    testing_start_time += timedelta(days=1)

user_list = get_user_list(parser.parse("2018-07-01"))
history_charging_df = charging_data[charging_data["createdNew"] < parser.parse("2018-07-01")].copy()
history_charging_df["facilityType"] = history_charging_df["locationId"].apply(lambda x: location.loc[str(x), "FacilityType"])
history_charging_df["createdHour"] = history_charging_df["createdHour"].astype(int)

history_user_group = history_charging_df.groupby(["userId"])
user_preference = defaultdict(dict)

for user in user_list:
    ## 充電站選多個
    user_preference[user]["locationId"] = list()
    user_preference[user]["facilityType"] = list()
    user_preference[user]["createdHour"] = list()

    most_prefer_num = math.floor(len(history_user_group.get_group(user)) / 2)
    cs_charging_num = history_user_group.get_group(user).groupby("locationId").size()
    for cs in cs_charging_num.keys():
        if cs_charging_num[cs] >= most_prefer_num:
            user_preference[user]["locationId"].append(cs)
            user_preference[user]["facilityType"].append(location.loc[cs, "FacilityType"])
            user_preference[user]["createdHour"] += list(history_user_group.get_group(user).groupby("locationId").get_group(cs).groupby("createdHour").size().keys())
            user_preference[user]["createdHour"] = sorted(list(set(user_preference[user]["createdHour"])))
    ### 避免有人都沒有超過 50% 的
    if len(user_preference[user]["locationId"]) == 0:
        user_preference[user]["locationId"].append(cs_charging_num.sort_values(ascending=False).keys()[0])
        user_preference[user]["facilityType"].append(location.loc[user_preference[user]["locationId"][0], "FacilityType"])
        user_preference[user]["createdHour"] += list(history_user_group.get_group(user).groupby("locationId").get_group(user_preference[user]["locationId"][0]).groupby("createdHour").size().keys())
        user_preference[user]["createdHour"] = sorted(list(set(user_preference[user]["createdHour"])))

 
    ## 時間以選中的充電站為主
    # user_preference[user]["locationId"] = history_user_group.get_group(user).groupby("locationId").size().sort_values(ascending=False).keys()[0]
    # user_preference[user]["facilityType"] = int(location.loc[user_preference[user]["locationId"], "FacilityType"])
    # user_preference[user]["createdHour"] = history_user_group.get_group(user).groupby("locationId").get_group(user_preference[user]["locationId"]).groupby("createdHour").size().sort_values(ascending=False).keys()[0]

    ## 時間以最多次的為主
    # user_preference[user]["facilityType"] = history_user_group.get_group(user).groupby("facilityType").size().sort_values(ascending=False).keys()[0]
    # user_preference[user]["createdHour"] = history_user_group.get_group(user).groupby("createdHour").size().sort_values(ascending=False).keys()[0]
    # user_preference[user]["locationId"] = history_user_group.get_group(user).groupby("locationId").size().sort_values(ascending=False).keys()[0]

user_preference

unfavored_type = 0
hit_type = 0
distance = 0
time = 0
for i, j in user_preference.items():
    print(i, j)
print(user_preference['603475'])
print(user_list)
# print(schedule_df)
# for idx, raw in schedule_df.iterrows():
    
#     print(raw['userID'])
#     print(location.loc[raw["locationID"], "FacilityType"])
#     if location.loc[raw["locationID"], "FacilityType"] in user_preference[raw["userID"]]["facilityType"]:
#         hit_type += 1
#     if not location.loc[raw["locationID"], "FacilityType"] in user_preference[raw["userID"]]["facilityType"]:
#         unfavored_type += 1
    
#     min_distance = 1000
#     for cs in user_preference[raw["userID"]]["locationId"]:
#         value = haversine(location.loc[raw["locationID"], ["Latitude", "Longitude"]].values, location.loc[cs, ["Latitude", "Longitude"]].values, unit = Unit.KILOMETERS)
#         min_distance = min(min_distance, value)
#     distance += min_distance

#     min_time = 1000
#     for hour in user_preference[raw["userID"]]["createdHour"]:
#         value = abs(raw["datetime"].hour - hour)
#         min_time = min(min_time, value)
#     time += min_time
    
# print(f"average hit type: {hit_type / len(schedule_df)}")
# print(f"average unfavored type: {unfavored_type / len(schedule_df)}")
# print(f"average distance: {distance / len(schedule_df)}")
# print(f"average time: {time / len(schedule_df)}")


