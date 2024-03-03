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


class Evaluation():

    charging_speed = 10 # 充電速率，單位 (kw/單位充電樁)
    charging_fee = 0.3  #  US$0.3/kwh
    parking_slots_num = 10 # 充電樁數量
    ALPHA = 0.5
    
    # 建物測試的時間
    building_start_time = parser.parse("2018-07-01")
    building_end_time = parser.parse("2018-07-08")

    # 充電測試的時間
    charging_start_time = parser.parse("2018-07-01")
    charging_end_time = parser.parse("2018-07-08")

    # 充電樁位置資訊
    location = pd.read_csv("./Dataset/location_5.csv", index_col="locationId")
    location.index = location.index.astype(str) # 將索引轉換為字串型態
    location = location.sort_values(by="buildingID") # 以 "buildingID" 這個欄位為鍵進行排序。
    location["buildingID"] = location["buildingID"].astype(str) # 將 "buildingID" 這個欄位轉換為字串型態。

    # 真實產用電資料
    building_data = pd.read_csv("./Dataset/generation_with_consumption_3.csv", index_col=0)
    building_data["datetime"] = pd.to_datetime(building_data["datetime"], format="%Y-%m-%d %H:%M:%S")
    building_data["buildingID"] = building_data["buildingID"].astype(str)

    # LSTM 預測產用電資料
    building_LSTM_data = pd.read_csv("./Result/predict_building_data.csv", index_col=0)
    building_LSTM_data["datetime"] = pd.to_datetime(building_LSTM_data["datetime"], format="%Y-%m-%d %H:%M:%S")
    building_LSTM_data["buildingID"] = building_LSTM_data["buildingID"].astype(str)
    
    # EV 歷史充電資料
    charging_data = pd.read_csv("./Dataset/charging_data_2_move.csv")
    charging_data["createdNew"] = pd.to_datetime(charging_data["createdNew"], format="%Y-%m-%d %H:%M:%S")
    charging_data["locationId"] = charging_data["locationId"].astype(str)
    charging_data["userId"] = charging_data["userId"].astype(str)

    def __init__(self):
        return


    def calculate_electricity_price(self, location, locationID, info):

        testing_start_time = self.charging_start_time
        
        # 每戶每月電價: each_building_price
        # 台灣每戶每月的基本電費, 30.73 為美元台幣匯率
        each_building_price = 262.5 / 30.73  
        
        capacity_price = 15.51 # 15.51 USD/kw/month
        contract_capacity = location.loc[locationID, "contractCapacity"] # kw
        
        electricity_price = defaultdict()
        for weekday in range(1, 8):
            if weekday < 6:
                electricity_price[weekday] = [0.056] * 8 + [0.092] * 4 + [0.267] * 6 + [0.092] * 5 + [0.056] * 1
            else:
                electricity_price[weekday] = [0.056] * 24
        
        ### 基本電費 ### 
        basic_tariff = each_building_price + (contract_capacity * capacity_price)  # 每月基本電價: 契約容量(contract_capacity) * 基本電價(capacity_price)

        ### 流動電費 ###
        current_tariff = 0
        for _ in range(1, 8):
            weekday = testing_start_time.isoweekday()
            for hour in range(24):
                current_tariff += electricity_price[weekday][hour] * (info[info["datetime"] == testing_start_time]["total"].values[0])
                testing_start_time += timedelta(hours=1)
        
        ### 超約罰金 ###
        overload_penalty = 0
        overload = info["total"].max()
        
        if max(overload, contract_capacity) != contract_capacity:
            overload -= contract_capacity # 算超過契約容量多少
            overload_penalty += min(overload, contract_capacity * 0.1) * capacity_price * 2 # 超出契約容量 10% 以下的部分
            overload -= min(overload, contract_capacity * 0.1) # 若超過的部分比契約容量的 10% 多，則算超出契約容量 10% 的部分
            overload_penalty += overload * capacity_price * 3 # 超出契約容量 10% 以上的部分

        return basic_tariff, current_tariff, overload_penalty
    


    def get_charging_request(self, date):
        '''
        charging_data: 歷史充電資料
        date: 充電開始時間
        '''
    
        start_date = self.charging_data["createdNew"] >= date # 一天的開始
        end_date = self.charging_data["createdNew"] < (date + timedelta(days=1)) # 一天的最後
        
        request_df = self.charging_data[start_date & end_date].copy()
        request_df["chargingHour"] = request_df["kwhNew"].apply(lambda x: x / self.charging_speed)
        
        charging_request = list()
    
        for item in request_df.iterrows():
            request = list(item[1][["_id", "userId", "chargingHour", "createdHour", "locationId"]])
            charging_request.append(request)

        return charging_request
    

    def get_user_list(self, date):

        temp = self.charging_data[self.charging_data["createdNew"] < date]
        user_list = temp.groupby("userId").groups.keys()
        
        return user_list 


    def get_parking_slots(self, building_data, location, date):
        '''
        LSTM
        '''
        electricity = building_data.loc[(building_data["datetime"] >= date) & (building_data["datetime"] < date + timedelta(days=1))].copy()

        for idx, row in electricity.iterrows():
            
            contract_capacity = location[location["buildingID"] == row["buildingID"]]["contractCapacity"].values[0]
            max_station_num = location[location["buildingID"] == row["buildingID"]]["stationNewNum"].values[0]
            parking_slots = math.floor((contract_capacity - row["consumption"] + row["generation"]) / self.charging_speed)
            parking_slots = parking_slots if parking_slots < max_station_num else max_station_num
            electricity.loc[idx, "parkingSlots"] = parking_slots
        
        electricity["parkingSlots"] = electricity["parkingSlots"].apply(lambda x: math.floor(x) if x > 0 else 0)

        return electricity
    

    def get_residual_slots(self, location, building_info, cs, hour, charging_len, schedule_type="OMMKP"):

        df = building_info.loc[(building_info["buildingID"] == location.loc[cs, "buildingID"]) & 
                                (building_info["datetime"].dt.hour >= hour) &
                                (building_info["datetime"].dt.hour < (hour + charging_len))].copy()

        if schedule_type == "popular":
            return df["parkingSlots"].values[0] if df["parkingSlots"].all() else 0

        ### 保留一個充電位 ###
        df["parkingSlots"] = df["parkingSlots"].apply(lambda x: 0 if (x-1) < 0 else (x-1))
        
        return df["parkingSlots"].values[0] if df["parkingSlots"].all() else 0
        # return df["parkingSlots"].values[0] if (df["parkingSlots"].all() and df["parkingSlots"].values[0] > 1) else 0

    def update_user_selection(self, location, slots_df, date, schedule, charging_len):
        '''
        schedule[0]: 充電位ID
        schedule[1]: 充電的時間, hour
        '''

        slots_df.loc[(slots_df["buildingID"] == location.loc[schedule[0], "buildingID"]) &
                    (slots_df["datetime"] >= (date + timedelta(hours = schedule[1]))) &
                    (slots_df["datetime"] < (date + timedelta(hours = schedule[1] + charging_len))), "parkingSlots"] -= 1

        return slots_df