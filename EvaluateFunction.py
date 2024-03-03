import pandas as pd
import numpy as np
import math
from datetime import timedelta
from dateutil import parser
from collections import defaultdict
from haversine import haversine, Unit
import json


CHARGING_SPEED = 10
CHARGING_FEE = 0.3  
PARKING_SLOTS_NUMS = 10


class EvaluateFunction():

    def __init__(self):

        self.charging_start_time = parser.parse("2018-07-01")
        self.charging_end_time = parser.parse("2018-07-08")
        self.building_start_time = parser.parse("2018-07-01")
        self.building_end_time = parser.parse("2018-07-08") 

        self.location = pd.read_csv("./Dataset/location_5.csv", index_col="locationId")
        self.building_data = pd.read_csv("./Dataset/generation_with_consumption_3.csv", index_col=0) # 建物資料
        self.building_LSTM_data = pd.read_csv("./Result/predict_building_data.csv", index_col=0) # 建物預測資料
        self.charging_data = pd.read_csv("./Dataset/charging_data_2_move.csv") # 充電歷史資料
        self.change_type()
        self.cal_building_parking_slots()
        self.truth_data = self.cal_building_parking_slots()
        self.user_preference = self.get_user_preference()
    

    def change_type(self):

        self.location.index = self.location.index.astype(str) # 將索引(locaiotnID)轉換為字串型態
        self.location = self.location.sort_values(by="buildingID") # 以 "buildingID" 這個欄位為鍵進行排序。
        self.location["buildingID"] = self.location["buildingID"].astype(str) # 將 "buildingID" 這個欄位轉換為字串型態。

        self.location.loc["50911", "contractCapacity"] = 222 # 設定 50911 的 contractCapacity
        self.location.loc["50266", "contractCapacity"] = 273 # 設定 50266 的 contractCapacity

        self.building_data["datetime"] = pd.to_datetime(self.building_data["datetime"], format="%Y-%m-%d %H:%M:%S")
        self.building_data["buildingID"] = self.building_data["buildingID"].astype(str)

        self.building_LSTM_data["datetime"] = pd.to_datetime(self.building_LSTM_data["datetime"], format="%Y-%m-%d %H:%M:%S")
        self.building_LSTM_data["buildingID"] = self.building_LSTM_data["buildingID"].astype(str)

        self.charging_data["createdNew"] = pd.to_datetime(self.charging_data["createdNew"], format="%Y-%m-%d %H:%M:%S")
        self.charging_data["locationId"] = self.charging_data["locationId"].astype(str)
        self.charging_data["userId"] = self.charging_data["userId"].astype(str)

        self.building_data.loc[self.building_data["buildingID"] == "13", "contract_capacity"] = 222       
        self.building_data.loc[self.building_data["buildingID"] == "10", "contract_capacity"] = 273

    
    def cal_building_parking_slots(self):
        '''
        計算建物的充電位數
        '''
        truth_data = self.building_data.copy()
        truth_data = truth_data.loc[(truth_data["datetime"] >= self.charging_start_time) & (truth_data["datetime"] < self.charging_end_time)]

        for idx, row in truth_data.iterrows():

            max_station_num = self.location[self.location["buildingID"] == row["buildingID"]]["stationNewNum"].values[0]
            parking_slots = math.floor((row["contract_capacity"] - row["consumption"] + row["generation"]) / CHARGING_SPEED)
            parking_slots = max(min(max_station_num, parking_slots), 0)
            truth_data.loc[idx, "parkingSlots"] = parking_slots
        
        return truth_data

    def calculate_electricity_price(self, location, locationID, info):
        '''
        電價計算
        '''
        testing_start_time = self.charging_start_time
        
        each_building_price = 262.5 / 30.73
        capacity_price = 15.51   # 236.2 NT$ 236.2 --> US$ 15.51 Kw/month
        contract_capacity = location.loc[locationID, "contractCapacity"]
        electricity_price = defaultdict()

        for weekday in range(1, 8):
            if weekday < 6:
                electricity_price[weekday] = [0.056] * 8 + [0.092] * 4 + [0.267] * 6 + [0.092] * 5 + [0.056] * 1
            else:
                electricity_price[weekday] = [0.056] * 24
        
        ### 基本電費 ### 
        # 每戶每月電價: each_building_price
        # 每月基本電價: 契約容量(contract_capacity) * 基本電價(capacity_price)
        basic_tariff = each_building_price + (contract_capacity * capacity_price)

        ### 流動電費 ###
        current_tariff = 0
        ### 星期一到星期日
        for _ in range(1, 8):
            weekday = testing_start_time.isoweekday()
            for hour in range(24):
                current_tariff += electricity_price[weekday][hour] * (info[info["datetime"] == testing_start_time]["total"].values[0])
                testing_start_time += timedelta(hours=1)
        
        ### 超約罰金 ###
        overload_penalty = 0
        overload = info["total"].max()
        
        if max(overload, contract_capacity) != contract_capacity:
            overload -= contract_capacity
            overload_penalty += min(overload, contract_capacity * 0.1) * capacity_price * 2 # 超出契約容量 10% 以下的部分
            overload -= min(overload, contract_capacity * 0.1)
            overload_penalty += overload * capacity_price * 3 # 超出契約容量 10% 以上的部分

        return basic_tariff, current_tariff, overload_penalty
    

    def get_charging_request(self, charging_data: pd.DataFrame, date):
        '''
        充電請求
        '''
        request_df = charging_data[(charging_data["createdNew"] >= date) & ((charging_data["createdNew"] < (date + timedelta(days=1))))].copy()
        request_df["chargingHour"] = request_df["kwhNew"].apply(lambda x: x / CHARGING_SPEED)
        charging_request = list()
    
        for item in request_df.iterrows():
            if item[1]["userId"] == "603475":
                continue
            request = list(item[1][["_id", "userId", "chargingHour", "createdHour", "locationId"]])
            charging_request.append(request)

        return charging_request



    def get_user_list(self, date):
        '''
        取得使用者名單
        '''
        
        temp = self.charging_data[self.charging_data["createdNew"] < date]
        user_list = temp.groupby("userId").groups.keys()
        
        return user_list


    def get_parking_slots(self, building_data, location, date):
        '''
        取得充電位數量
        '''
        electricity = building_data[ (building_data["datetime"] >= date) & (building_data["datetime"] < date + timedelta(days=1))]

        for idx, row in electricity.iterrows():
            contract_capacity = location[location["buildingID"] == row["buildingID"]]["contractCapacity"].values[0]
            max_station_num = location[location["buildingID"] == row["buildingID"]]["stationNewNum"].values[0]
            parking_slots = math.floor((contract_capacity - row["consumption"] + row["generation"]) / CHARGING_SPEED)
            parking_slots = parking_slots if parking_slots < max_station_num else max_station_num
            electricity.loc[idx, "parkingSlots"] = parking_slots
        
        electricity["parkingSlots"] = electricity["parkingSlots"].apply(lambda x: math.floor(x) if x > 0 else 0)

        return electricity
    

    def get_residual_slots(self, location, building_info, cs, hour, charging_len, schedule_type="popular"):
        '''
        取得剩餘充電位
        待確認
        '''
        df = building_info.loc[(building_info["buildingID"] == location.loc[cs, "buildingID"]) & 
                            (building_info["datetime"].dt.hour >= hour) &
                            (building_info["datetime"].dt.hour < (hour + charging_len))]

        
        if schedule_type == "popular":
            return df["parkingSlots"].values[0] if df["parkingSlots"].all() else 0

        ### 保留一個充電位 ###
        df["parkingSlots"] = df["parkingSlots"].apply(lambda x: 0 if (x-1) < 0 else (x-1))
        
        return df["parkingSlots"].values[0] if df["parkingSlots"].all() else 0
        # return df["parkingSlots"].values[0] if (df["parkingSlots"].all() and df["parkingSlots"].values[0] > 1) else 0


    def update_user_selection(self, location, slots_df, date, schedule, charging_len):
        '''
        更新目前充電位
        '''

        slots_df.loc[(slots_df["buildingID"] == location.loc[schedule[0], "buildingID"]) &
                    (slots_df["datetime"] >= (date + timedelta(hours = schedule[1]))) &
                    (slots_df["datetime"] < (date + timedelta(hours = schedule[1] + charging_len))), "parkingSlots"] -= 1

        return slots_df
    


    def get_user_preference(self):
        '''
        計算 user preference
        '''

        user_list = self.get_user_list(parser.parse("2018-07-01"))
        history_charging_df = self.charging_data[self.charging_data["createdNew"] < parser.parse("2018-07-01")].copy()
        history_charging_df["facilityType"] = history_charging_df["locationId"].apply(lambda x: self.location.loc[str(x), "FacilityType"])
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
                    user_preference[user]["facilityType"].append(self.location.loc[cs, "FacilityType"])
                    user_preference[user]["createdHour"] += list(history_user_group.get_group(user).groupby("locationId").get_group(cs).groupby("createdHour").size().keys())
                    user_preference[user]["createdHour"] = sorted(list(set(user_preference[user]["createdHour"])))
            
            ### 避免有人都沒有超過 50% 的
            if len(user_preference[user]["locationId"]) == 0:
                user_preference[user]["locationId"].append(cs_charging_num.sort_values(ascending=False).keys()[0])
                user_preference[user]["facilityType"].append(self.location.loc[user_preference[user]["locationId"][0], "FacilityType"])
                user_preference[user]["createdHour"] += list(history_user_group.get_group(user).groupby("locationId").get_group(user_preference[user]["locationId"][0]).groupby("createdHour").size().keys())
                user_preference[user]["createdHour"] = sorted(list(set(user_preference[user]["createdHour"])))
        
        return user_preference



    def cal_price(self, schedule_df: pd.DataFrame):

        overage = 0
        electricity_price = pd.DataFrame([], columns=["basic_tariff",
                                                    "current_tariff",
                                                    "overload_penalty",
                                                    "total"])

        schedule_statistic_count = pd.DataFrame([], 
                                    columns=[self.charging_start_time + timedelta(hours=hour) for hour in range(0, 7*24)])

        overload_percentage = list()
        
        schedule_revenue = defaultdict()
        schedule_ev_charging_volume = list()
        schedule_electricity_cost = defaultdict()
        schedule_ev_revenue = defaultdict()

        for cs in self.location.index:

            info = self.building_data[(self.building_data["buildingID"] == self.location.loc[cs, "buildingID"])
                                & (self.building_data["datetime"] >= self.building_start_time) & (self.building_data["datetime"] < self.building_end_time)].copy()

            ### EV charging data ###
            ev_info = [0 for i in range(24*7)]
            current = self.charging_start_time

            for day in range(7):
                for hour in range(24):
                    charging_value = len(schedule_df[(schedule_df["locationID"] == cs) &
                                                    (schedule_df["datetime"] >= current) &
                                                    (schedule_df["datetime"] < (current + timedelta(days=1))) &
                                                    (schedule_df["datetime"].dt.hour <= hour) &
                                                    ((schedule_df['datetime'].dt.hour + schedule_df["chargingLen"]) > hour)]) * CHARGING_SPEED

                    ev_info[(day * 24) + hour] = charging_value
                current += timedelta(days=1)

            schedule_ev_charging_volume.append(ev_info)
            schedule_ev_revenue[cs] = sum(ev_info) * CHARGING_FEE
            info["charging"] = ev_info
            info["total"] = info["consumption"] - info["generation"] + info["charging"]

            ### check number of exceed ###
            info["exceed"] = info["total"] - self.location.loc[cs, "contractCapacity"]
            info["exceed"] = info["exceed"].apply(lambda x: 0 if x < 0 else x)

            overload_slots = 0
            for raw in info["exceed"]:
                if raw != 0:
                    overload_slots += 1
                # overload_slots += math.ceil(raw / charging_speed)

            overload_percentage.append(overload_slots / (7 * 24))
            # print(int(location.loc[cs, 'buildingID']), ":")
            # print(f"overload = {'{:.2f}'.format(info['exceed'].sum())}")
            # print(f"overload_percentage = {'{:.4f}'.format(overload_percentage[-1])}")

            overage += info["exceed"].sum()
            basic_tariff, current_tariff, overload_penalty = self.calculate_electricity_price(self.location, cs, info)
            total_price = basic_tariff + current_tariff + overload_penalty
            
            electricity_price.loc[len(electricity_price)] = [basic_tariff, current_tariff, overload_penalty, total_price]
            schedule_revenue[cs] = (-1) * total_price + (CHARGING_FEE * info["charging"].sum())
            schedule_electricity_cost[cs] = total_price
            # print(f"revenue = {'{:.2f}'.format(schedule_revenue[cs])}")

            info["chargingCount"] = info["charging"].apply(lambda x: x/CHARGING_SPEED)
            info.set_index("datetime", inplace=True)
            schedule_statistic_count.loc[len(schedule_statistic_count)] = info["chargingCount"].T

        print(f"==============================================")
        print(f"average revenue = {float(sum(schedule_revenue.values()) / 20)}")
        print(electricity_price.mean(axis=0))
        average_revenue = float(sum(schedule_revenue.values()) / 20)
        basic_tariff_mean = electricity_price['basic_tariff'].mean()
        current_tariff_mean = electricity_price['current_tariff'].mean()
        overload_penalty_mean = electricity_price['overload_penalty'].mean()

        return basic_tariff_mean, current_tariff_mean, overload_penalty_mean, average_revenue, schedule_ev_charging_volume


    def cal_variation(self, schedule_ev_charging_volume):
        '''
        Variaiton 1, Variation 2
        '''
       
        testing_start_time = self.charging_start_time
        schedule_utilization = defaultdict(list)
        schedule_var = list()
        
        while testing_start_time < self.charging_end_time:
            for cs in range(20):
                truth_slots = self.truth_data.loc[(self.truth_data["datetime"] == testing_start_time) & 
                                                  (self.truth_data["buildingID"] == str(cs)), "parkingSlots"].values[0]
                
                days = (testing_start_time - self.charging_start_time).days
                seconds = (testing_start_time - self.charging_start_time).seconds
                schedule_slots = schedule_ev_charging_volume[cs][int(days * 24 + (seconds / 3600))] / CHARGING_SPEED
                schedule_utilization[testing_start_time].append(schedule_slots / truth_slots if truth_slots != 0 else 1)
                

            schedule_var.append(np.var(schedule_utilization[testing_start_time]))
            testing_start_time += timedelta(hours=1)

        schedule_variation = sum(schedule_var) / len(schedule_var)
        
       
        data = schedule_utilization
        total_usage = defaultdict(list)

        # 遍历所有的时间点
        for dt, usage in data.items():
            for i in range(20):
                total_usage[i].append(usage[i])

        value_sum = 0
        for value in total_usage.values():
            value_sum += np.var(value)
        
        print(f"==============================================")
        print("Variaiton 1: ", round(schedule_variation, 4))
        print("Variation 2: ", round(value_sum/20, 4))
        var_1 = round(schedule_variation, 4)
        var_2 = round(value_sum/20, 4)

        return var_1, var_2



    def cal_user_preference_1(self, schedule_df):
        # '''
        # 使用哲紜的 user preference
        # '''
        unfavored_type = 0
        hit_type = 0
        distance = 0
        time = 0
        
        max_time = 0
        for idx, raw in schedule_df.iterrows():
            
            if self.location.loc[raw["locationID"], "FacilityType"] in self.user_preference[raw["userID"]]["facilityType"]:
                hit_type += 1
            if not self.location.loc[raw["locationID"], "FacilityType"] in self.user_preference[raw["userID"]]["facilityType"]:
                unfavored_type += 1
            
            min_distance = 1000
            for cs in self.user_preference[raw["userID"]]["locationId"]:
                value = haversine(self.location.loc[raw["locationID"], ["Latitude", "Longitude"]].values, 
                                  self.location.loc[cs, ["Latitude", "Longitude"]].values, unit = Unit.KILOMETERS)
                min_distance = min(min_distance, value)
            distance += min_distance

            min_time = 1000
            for hour in self.user_preference[raw["userID"]]["createdHour"]:
                value = abs(raw["datetime"].hour - hour)
                min_time = min(min_time, value)
                max_time = max(max_time, value)
            time += min_time
        
        print("====================== user preference origin =======================")
        print(f"average hit type: {round(hit_type / len(schedule_df), 4)}")
        print(f"average unfavored type: {round(unfavored_type / len(schedule_df), 4)}")
        print(f"average distance: {round(distance / len(schedule_df), 4)}")
        print(f"average time: {round(time / len(schedule_df), 4)}")
        print(f"max time: {round(max_time, 4)}")

        average_unfavored_type = round(unfavored_type / len(schedule_df), 4)
        average_distance = round(distance / len(schedule_df), 4)
        average_time = round(time / len(schedule_df), 4)

        return average_unfavored_type, average_distance, average_time

    def cal_user_preference_04(self, schedule_df):
        '''
        使用奐揚的 user preference, unfavored 0.4
        '''
        with open('user_facility_perc_dic.json', 'r') as f:
            # Load JSON data from file
            user_facility_perc_dic = json.load(f)

        hit_type = 0
        unfavored_type = 0

        for _ , raw in schedule_df.iterrows():
            
            facility_type_list = [int(i) for i in user_facility_perc_dic[raw["userID"]].keys()]
            facility_type = self.location.loc[raw["locationID"], "FacilityType"]
            if facility_type in facility_type_list and user_facility_perc_dic[raw["userID"]][str(facility_type)] >= 0.4:
                hit_type += 1
            if facility_type in facility_type_list and user_facility_perc_dic[raw["userID"]][str(facility_type)] < 0.4:
                unfavored_type += 1
            if facility_type not in facility_type_list:
                unfavored_type += 1
               
        print("=================== user preference 0.4 =======================")
        print(f"hit type: {hit_type}")
        print(f"average hit type: {round(hit_type / len(schedule_df), 4)}")
        print(f"unfavored type: {unfavored_type}")
        print(f"average unfavored type: {round(unfavored_type / len(schedule_df), 4)}")

        average_unfavored_type_04 = round(unfavored_type / len(schedule_df), 4)
        
        return average_unfavored_type_04


    def cal_user_preference_03(self, schedule_df):
        '''
        使用奐揚的 user preference, unfavored 0.4
        '''
        with open('user_facility_perc_dic.json', 'r') as f:
            # Load JSON data from file
            user_facility_perc_dic = json.load(f)

        hit_type = 0
        unfavored_type = 0

        for _ , raw in schedule_df.iterrows():
            
            facility_type_list = [int(i) for i in user_facility_perc_dic[raw["userID"]].keys()]
            facility_type = self.location.loc[raw["locationID"], "FacilityType"]
            if facility_type in facility_type_list and user_facility_perc_dic[raw["userID"]][str(facility_type)] >= 0.3:
                hit_type += 1
            if facility_type in facility_type_list and user_facility_perc_dic[raw["userID"]][str(facility_type)] < 0.3:
                unfavored_type += 1
            if facility_type not in facility_type_list:
                unfavored_type += 1
               
        print("=================== user preference 0.3 =======================")
        print(f"hit type: {hit_type}")
        print(f"average hit type: {round(hit_type / len(schedule_df), 4)}")
        print(f"unfavored type: {unfavored_type}")
        print(f"average unfavored type: {round(unfavored_type / len(schedule_df), 4)}")

        average_unfavored_type_03 = round(unfavored_type / len(schedule_df), 4)
        
        return average_unfavored_type_03


    def cal_favor_ratio(self, schedule_df):

        '''
        使用奐揚的 user preference, unfavored 0.4
        '''
        with open('user_facility_perc_dic.json', 'r') as f:
            # Load JSON data from file
            user_facility_perc_dic = json.load(f)


        favor_count = []

        for _ , raw in schedule_df.iterrows():
            
            facility_type_list = [int(i) for i in user_facility_perc_dic[raw["userID"]].keys()]
            facility_type = self.location.loc[raw["locationID"], "FacilityType"]
            
            if facility_type in facility_type_list:
                favor_count.append(user_facility_perc_dic[raw["userID"]][str(facility_type)])
            else:
                favor_count.append(0)

        print("=================== favor ratio =======================")
        print(f"favor ratio: {round(sum(favor_count)/len(favor_count), 4)}")
        favor_ratio = round(sum(favor_count)/len(favor_count), 4)
        
        return favor_ratio

