import pandas as pd
import math
from datetime import timedelta
from dateutil import parser
from collections import defaultdict

CHARGING_SPEED = 10

class base():

    def __init__(self):

        self.location = pd.read_csv("../Dataset/location_5.csv", index_col="locationId") 
        self.building_truth_data = pd.read_csv("../Dataset/generation_with_consumption_3.csv", index_col=0)  
        self.building_predict_data = pd.read_csv("../Result/predict_building_data.csv", index_col=0)  
        self.charging_data = pd.read_csv("../Dataset/charging_data_2_move.csv") 
        self.average_request_df = pd.read_csv("../Dataset/average_request_num.csv", index_col=0) # 7/1~7/7 的充電請求數
        self.user_history_preference = defaultdict(dict)
        self.change_type()
        self.get_user_preference()


    def change_type(self):
        '''
        資料處理，轉換資料型態
        '''

        # 充電站的詳細資訊
        self.location.index = self.location.index.astype(str) # locationId
        self.location["buildingID"] = self.location["buildingID"].astype(str)
        self.location["budget"] = self.location["budget"].astype(int)
        self.location["FacilityType"] = self.location["FacilityType"].astype(int)

        # 建物實際耗電量和發電量
        self.building_truth_data["datetime"] = pd.to_datetime(self.building_truth_data["datetime"], format="%Y-%m-%d %H:%M:%S")
        self.building_truth_data["buildingID"] = self.building_truth_data["buildingID"].astype(str)

        # 建物預測的耗電量和發電量
        self.building_predict_data["datetime"] = pd.to_datetime(self.building_predict_data["datetime"], format="%Y-%m-%d %H:%M:%S")
        self.building_predict_data["buildingID"] = self.building_predict_data["buildingID"].astype(str)
        
        # 充電時間/充電需求
        self.charging_data["createdNew"] = pd.to_datetime(self.charging_data["createdNew"], format="%Y-%m-%d %H:%M:%S")
        self.charging_data["locationId"] = self.charging_data["locationId"].astype(str)
        self.charging_data["userId"] = self.charging_data["userId"].astype(str)

        # 每個充電站 7/1~7/7 的平均請求數量
        self.average_request_df["datetime"] = pd.to_datetime(self.average_request_df["datetime"], format="%Y-%m-%d")
        self.average_request_df["locationId"] = self.average_request_df["locationId"].astype(str)


    def get_user_list(self, date):
        '''
        篩選資料的時間，得到 user_list
        '''
        temp = self.charging_data[self.charging_data["createdNew"] < date] # "2018-07-01" 以前的充電資料，用於做訓練
        user_list = temp.groupby("userId").groups.keys() # 返回 2018-07-01 以前的充電資料的"所有使用者"

        return user_list


    def get_charging_request(self, date):
        '''
        取得該日內所有 EV 的充電請求紀錄
        '''
        request_df = self.charging_data[(self.charging_data["createdNew"] >= date) & ((self.charging_data["createdNew"] < (date + timedelta(days=1))))].copy()
        request_df["chargingHour"] = request_df["kwhNew"].apply(lambda x: x / CHARGING_SPEED) # 充電時數
        
        charging_request = list()
        for _ , row in request_df.iterrows():
            if row["userId"] == "603475":
                continue
            request = list(row[["_id", "userId", "chargingHour", "createdHour", "locationId"]])
            charging_request.append(request)

        return charging_request
    

    def get_parking_slots(self, building_data, date):
        '''
        取得當日內所有時間點的充電樁的數量
        '''
    
        electricity = building_data.loc[(building_data["datetime"] >= date) & (building_data["datetime"] < (date + timedelta(days=1)))].copy()

        for idx, row in electricity.iterrows():
            
            max_station_num = self.location.loc[self.location["buildingID"] == row["buildingID"], "stationNewNum"].values[0] # 取第一個符合的
            parking_slots = math.floor((self.location.loc[self.location["buildingID"] == row["buildingID"], "contractCapacity"] - row["consumption"] + row["generation"]) // CHARGING_SPEED)
            parking_slots = max(min(max_station_num, parking_slots), 0)
            electricity.loc[idx, "parkingSlots"] = parking_slots

        return electricity


    def get_residual_slots(self, slots_df, cs, hour, charging_len, schedule_type="OGAP"):
        '''
        判斷要充電的時段中是否還有剩餘充電位，只要有一個時段沒空位就回傳 0
        slots_df: 充電位資料
        cs: 要檢查的充電站
        hour: 開始充電的時間
        charging_len: 要充多久
        schedule_type: 排程種類，預設是 OGAP
        '''
        current_slots_df = slots_df.loc[(slots_df["buildingID"] == self.location.loc[cs, "buildingID"]) &
                          (slots_df["datetime"].dt.hour >= hour) &
                          (slots_df["datetime"].dt.hour < (hour + charging_len))].copy()
        
        if schedule_type == "OGAP" and len(current_slots_df) != charging_len: 
            return 0

        return current_slots_df["parkingSlots"].values[0] if current_slots_df["parkingSlots"].all() else 0
    

    def get_user_preference(self):
        '''
        取得使用者的歷史偏好
        使用者去某個充電站的次數，佔所有他去的所有的充電站的總數的一半，代表他很喜歡去
        '''
        user_list = self.get_user_list(parser.parse("2018-07-01"))
        history_charging_df = self.charging_data[self.charging_data["createdNew"] < parser.parse("2018-07-01")].copy() 
        history_charging_df["facilityType"] = history_charging_df["locationId"].apply(lambda x: self.location.loc[str(x), "FacilityType"])
        history_charging_df["createdHour"] = history_charging_df["createdHour"].astype(int)
        history_user_group = history_charging_df.groupby(["userId"])

        for user in user_list:
            
            # 充電站選多個
            self.user_history_preference[user]["locationId"] = list()
            self.user_history_preference[user]["facilityType"] = list()
            self.user_history_preference[user]["createdHour"] = list()

            most_prefer_num = math.floor(len(history_user_group.get_group(user)) / 2) # 使用者總充電次數的一半
            cs_charging_num = history_user_group.get_group(user).groupby("locationId").size() # 使用者在不同充電站的充電次數

            for cs in cs_charging_num.keys():
                if cs_charging_num[cs] >= most_prefer_num: 
                    # 代表有一半的次數都是在這個充電站充電
                    self.user_history_preference[user]["locationId"].append(cs)
                    self.user_history_preference[user]["facilityType"].append(self.location.loc[cs, "FacilityType"])
                    self.user_history_preference[user]["createdHour"] += list(history_user_group.get_group(user).groupby("locationId").get_group(cs).groupby("createdHour").size().keys())
                    self.user_history_preference[user]["createdHour"] = sorted(list(set(self.user_history_preference[user]["createdHour"])))
 
            # 避免有人都沒有超過 50% 的
            if len(self.user_history_preference[user]["locationId"]) == 0:
                self.user_history_preference[user]["locationId"].append(cs_charging_num.sort_values(ascending=False).keys()[0])
                self.user_history_preference[user]["facilityType"].append(self.location.loc[self.user_history_preference[user]["locationId"][0], "FacilityType"])
                self.user_history_preference[user]["createdHour"] += list(history_user_group.get_group(user).groupby("locationId").get_group(self.user_history_preference[user]["locationId"][0]).groupby("createdHour").size().keys())
                self.user_history_preference[user]["createdHour"] = sorted(list(set(self.user_history_preference[user]["createdHour"])))
        
   
    
            
