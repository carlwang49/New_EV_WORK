import pandas as pd
import numpy as np
import json
import warnings
import math
import matplotlib.pyplot as plt
import seaborn as sns
from haversine import haversine, Unit
from datetime import datetime, timedelta
from dateutil import parser
from collections import defaultdict

TEST_NAME = input("Enter the test name: ")
FLAG = True # True if Baseline
BASELINE_PATH = "./NewResult/Baseline/SCOMMIT_v2_test/2024-02-29/SIGMOID_INCENTIVE_UNIT_0.2/"
PATH = "./NewResult/Baseline/unscheduled_userBehavior_without_prediction_maxSlots_10/2024-02-29/"

sns.set()
sns.set_style('whitegrid')
warnings.filterwarnings("ignore") 


CHARGING_SPEED = 10
CHARGING_FEE = 0.3  
PARKING_SLOTS_NUMS = 10

building_start_time = parser.parse("2018-07-01")
building_end_time = parser.parse("2018-07-08")
charging_start_time = parser.parse("2018-07-01")
charging_end_time = parser.parse("2018-07-08")

location = pd.read_csv("./Dataset/location_5.csv", index_col="locationId")
location.index = location.index.astype(str) # 將索引(locaiotnID)轉換為字串型態
location = location.sort_values(by="buildingID") # 以 "buildingID" 這個欄位為鍵進行排序。
location["buildingID"] = location["buildingID"].astype(str) # 將 "buildingID" 這個欄位轉換為字串型態。

location.loc["50911", "contractCapacity"] = 222 # 設定 50911 的 contractCapacity
location.loc["50266", "contractCapacity"] = 273 # 設定 50266 的 contractCapacity

building_data = pd.read_csv("./Dataset/generation_with_consumption_3.csv", index_col=0) # 建物資料
building_LSTM_data = pd.read_csv("./Result/predict_building_data.csv", index_col=0) # 建物預測資料
charging_data = pd.read_csv("./Dataset/charging_data_2_move.csv") # 充電歷史資料

building_data["datetime"] = pd.to_datetime(building_data["datetime"], format="%Y-%m-%d %H:%M:%S")
building_data["buildingID"] = building_data["buildingID"].astype(str)

building_LSTM_data["datetime"] = pd.to_datetime(building_LSTM_data["datetime"], format="%Y-%m-%d %H:%M:%S")
building_LSTM_data["buildingID"] = building_LSTM_data["buildingID"].astype(str)

charging_data["createdNew"] = pd.to_datetime(charging_data["createdNew"], format="%Y-%m-%d %H:%M:%S")
charging_data["locationId"] = charging_data["locationId"].astype(str)
charging_data["userId"] = charging_data["userId"].astype(str)

building_data.loc[building_data["buildingID"] == "13", "contract_capacity"] = 222
building_data.loc[building_data["buildingID"] == "10", "contract_capacity"] = 273



def calculate_electricity_price(location, locationID, info):

    '''
    電價計算
    '''

    testing_start_time = charging_start_time
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




def get_charging_request(charging_data, date):
    '''
    充電請求
    '''
    request_df = charging_data[(charging_data["createdNew"] >= date) & ((charging_data["createdNew"] < (date + timedelta(days=1))))]
    request_df["chargingHour"] = request_df["kwhNew"].apply(lambda x: x / CHARGING_SPEED)
    charging_request = list()
 
    for item in request_df.iterrows():
        if item[1]["userId"] == "603475":
            continue
        request = list(item[1][["_id", "userId", "chargingHour", "createdHour", "locationId"]])
        charging_request.append(request)

    return charging_request


# '''
# 計算建物的充電位數
# '''
truth_data = building_data.copy()
truth_data = truth_data.loc[(truth_data["datetime"] >= charging_start_time) & (truth_data["datetime"] < charging_end_time)]

for idx, row in truth_data.iterrows():

    max_station_num = location[location["buildingID"] == row["buildingID"]]["stationNewNum"].values[0]
    parking_slots = math.floor((row["contract_capacity"] - row["consumption"] + row["generation"]) / CHARGING_SPEED)

    parking_slots = max(min(max_station_num, parking_slots), 0)
    truth_data.loc[idx, "parkingSlots"] = parking_slots


def get_user_list(date):
    '''
    取得使用者名單
    '''
    
    temp = charging_data[charging_data["createdNew"] < date]
    user_list = temp.groupby("userId").groups.keys()
    return user_list



def get_parking_slots(building_data, location, date):
    ### LSTM ###
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


def get_residual_slots(location, building_info, cs, hour, charging_len, schedule_type="popular"):

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


def update_user_selection(location, slots_df, date, schedule, charging_len):
    '''
    更新目前充電位
    '''

    slots_df.loc[(slots_df["buildingID"] == location.loc[schedule[0], "buildingID"]) &
                 (slots_df["datetime"] >= (date + timedelta(hours = schedule[1]))) &
                 (slots_df["datetime"] < (date + timedelta(hours = schedule[1] + charging_len))), "parkingSlots"] -= 1

    return slots_df

# '''
# 計算 user preference
# '''

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


if __name__ == '__main__':

    if FLAG:    
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

        schedule_df = pd.DataFrame([], columns=columns)
        testing_start_time = charging_start_time
        null_value = 0
        import time
        count = 0
        for day in range(7):

            charging_request = get_charging_request(charging_data, testing_start_time)
            recommend = pd.read_csv(BASELINE_PATH + f"{testing_start_time.strftime('%m%d')}.csv")
            recommend["datetime"] = pd.to_datetime(recommend["datetime"], format="%Y-%m-%d %H:%M:%S")
            
            for request in charging_request:
                
                try:
                    schedule = recommend[recommend["requestID"] == request[0]]
                    user_accept_list = schedule['user_accept'].values.tolist()
                    num_true = user_accept_list.count(True)
                    count += num_true
                    # 創建一個新的數據框，儲存所有符合條件的行
                    user_accept_true_df = schedule[schedule['user_accept'] == True]

                    new_rows = pd.DataFrame({
                        'datetime': schedule['datetime'].values,
                        'testing_start_time': testing_start_time.strftime("%m%d"),
                        'userID': schedule['userID'].values,
                        'locationID': schedule['locationID'].values.astype(str),
                        'buildingID': location.loc[schedule['locationID'].values.astype(str), 'buildingID'],
                        'chargingLen': schedule['chargingLen'].values,
                        'originLocationID': schedule['originLocationID'].values,
                        'originHour': schedule['originHour'].values
                    })
                    # new_rows = pd.DataFrame({
                    #     'datetime': user_accept_true_df['datetime'].values,
                    #     'testing_start_time': testing_start_time.strftime("%m%d"),
                    #     'userID': user_accept_true_df['userID'].values,
                    #     'locationID': user_accept_true_df['locationID'].values.astype(str),
                    #     'buildingID': location.loc[user_accept_true_df['locationID'].values.astype(str), 'buildingID'],
                    #     'chargingLen': user_accept_true_df['chargingLen'].values,
                    #     'originLocationID': user_accept_true_df['originLocationID'].values,
                    #     'originHour': user_accept_true_df['originHour'].values
                    # })
                    
                    # 新增這些行到schedule_df
                    schedule_df = pd.concat([schedule_df, new_rows])
            
                except Exception as e:
                    print(request, e)
                    null_value += 1

            testing_start_time += timedelta(days=1)

        print(f"=============={TEST_NAME}=======================")
        print("miss_value:", null_value)
        print("accept: ", count)
        print("accept rate:", round(count/schedule_df.shape[0], 4))
        print("request number: ", schedule_df.shape[0])

        overage = 0
        electricity_price = pd.DataFrame([], columns=["basic_tariff", 
                                                    "current_tariff", 
                                                    "overload_penalty", 
                                                    "total"])
        
        schedule_statistic_count = \
            pd.DataFrame([], 
            columns=[charging_start_time + timedelta(hours=hour) for hour in range(0, 7*24)])

        overload_percentage = list()

        schedule_revenue = defaultdict()
        schedule_ev_charging_volume = list()
        schedule_electricity_cost = defaultdict() 
        schedule_ev_revenue = defaultdict()
        schedule_df["datetime"] = pd.to_datetime(schedule_df["datetime"])
        
        for cs in location.index:
            
            info = building_data[ (building_data["buildingID"] == location.loc[cs, "buildingID"]) 
                                & (building_data["datetime"] >= building_start_time) & (building_data["datetime"] < building_end_time)]

            ### EV charging data ###
            ev_info = [0 for i in range(24*7)]
            current = charging_start_time
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
            info["exceed"] = info["total"] - location.loc[cs, "contractCapacity"]
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
            basic_tariff, current_tariff, overload_penalty = calculate_electricity_price(location, cs, info)
            total_price = basic_tariff + current_tariff + overload_penalty
            electricity_price.loc[len(electricity_price)] = [basic_tariff, current_tariff, overload_penalty, total_price]
            schedule_revenue[cs] = (-1) * total_price + (CHARGING_FEE * info["charging"].sum())
            schedule_electricity_cost[cs] = total_price
            # print(f"revenue = {'{:.2f}'.format(schedule_revenue[cs])}")

            info["chargingCount"] = info["charging"].apply(lambda x: x/CHARGING_SPEED)
            info.set_index("datetime", inplace=True)
            schedule_statistic_count.loc[len(schedule_statistic_count)] = info["chargingCount"].T

        print(f"=============={TEST_NAME}=======================")
        print(f"average revenue = {float(sum(schedule_revenue.values()) / 20)}")
        print(electricity_price.mean(axis=0))

        # '''
        # Variaiton 1
        # '''
        print(f"=============={TEST_NAME}=======================")
        testing_start_time = charging_start_time
        schedule_utilization = defaultdict(list)
        schedule_var = list()
        while testing_start_time < charging_end_time:
            for cs in range(20):
                truth_slots = truth_data.loc[(truth_data["datetime"] == testing_start_time) & (truth_data["buildingID"] == str(cs)), "parkingSlots"].values[0]
                
                days = (testing_start_time - charging_start_time).days
                seconds = (testing_start_time - charging_start_time).seconds
                schedule_slots = schedule_ev_charging_volume[cs][int(days * 24 + (seconds / 3600))] / CHARGING_SPEED
                schedule_utilization[testing_start_time].append(schedule_slots / truth_slots if truth_slots != 0 else 1)
                

            schedule_var.append(np.var(schedule_utilization[testing_start_time]))
            testing_start_time += timedelta(hours=1)

        schedule_variation = sum(schedule_var) / len(schedule_var)
        print("Variaiton 1: ", round(schedule_variation, 4))

        # '''
        # Variation 2
        # '''
        data = schedule_utilization

        total_usage = defaultdict(list)

        # 遍历所有的时间点
        for dt, usage in data.items():
            for i in range(20):
                total_usage[i].append(usage[i])

        value_sum = 0
        for value in total_usage.values():
            value_sum += np.var(value)

        print("Variation 2: ", round(value_sum/20, 4))


        # '''
        # 使用哲紜的 user preference
        # '''
        unfavored_type = 0
        hit_type = 0
        distance = 0
        time = 0
        
        max_time = 0
        for idx, raw in schedule_df.iterrows():
            
            if location.loc[raw["locationID"], "FacilityType"] in user_preference[raw["userID"]]["facilityType"]:
                hit_type += 1
            if not location.loc[raw["locationID"], "FacilityType"] in user_preference[raw["userID"]]["facilityType"]:
                unfavored_type += 1
            
            min_distance = 1000
            for cs in user_preference[raw["userID"]]["locationId"]:
                value = haversine(location.loc[raw["locationID"], ["Latitude", "Longitude"]].values, location.loc[cs, ["Latitude", "Longitude"]].values, unit = Unit.KILOMETERS)
                min_distance = min(min_distance, value)
            distance += min_distance

            min_time = 1000
            
            for hour in user_preference[raw["userID"]]["createdHour"]:
                value = abs(raw["datetime"].hour - hour)
                min_time = min(min_time, value)
                max_time = max(max_time, value)
            time += min_time
        
        print("===================哲紜的 user preference=======================")
        print(f"average hit type: {round(hit_type / len(schedule_df), 4)}")
        print(f"average unfavored type: {round(unfavored_type / len(schedule_df), 4)}")
        print(f"average distance: {round(distance / len(schedule_df), 4)}")
        print(f"average time: {round(time / len(schedule_df), 4)}")
        print(f"max time: {round(max_time, 4)}")

        # '''
        # 使用奐揚的 user preference, unfavored 0.4
        # '''
        with open('user_facility_perc_dic.json', 'r') as f:
            # Load JSON data from file
            user_facility_perc_dic = json.load(f)

        user_list = get_user_list(parser.parse("2018-07-01"))
        history_charging_df = charging_data[charging_data["createdNew"] < parser.parse("2018-07-01")].copy()
        history_charging_df["facilityType"] = history_charging_df["locationId"].apply(lambda x: location.loc[str(x), "FacilityType"])
        history_charging_df["createdHour"] = history_charging_df["createdHour"].astype(int)

        history_user_group = history_charging_df.groupby(["userId"])


        hit_type = 0
        unfavored_type = 0
        distance = 0
        time = 0

        for idx, raw in schedule_df.iterrows():
            
            facility_type_list = [int(i) for i in user_facility_perc_dic[raw["userID"]].keys()]
            facility_type = location.loc[raw["locationID"], "FacilityType"]
            if facility_type in facility_type_list and user_facility_perc_dic[raw["userID"]][str(facility_type)] >= 0.4:
                hit_type += 1
            if facility_type in facility_type_list and user_facility_perc_dic[raw["userID"]][str(facility_type)] < 0.4:
                unfavored_type += 1
            if facility_type not in facility_type_list:
                unfavored_type += 1
            
            min_distance = 1000
            count = 0
            try:
                for cs in user_preference[raw["userID"]]["locationId"]:
                    value = haversine(location.loc[raw["locationID"], ["Latitude", "Longitude"]].values, location.loc[cs, ["Latitude", "Longitude"]].values, unit = Unit.KILOMETERS)
                    min_distance = min(min_distance, value)
                distance += min_distance

                min_time = 1000
                for hour in user_preference[raw["userID"]]["createdHour"]:
                    value = abs(raw["datetime"].hour - hour)
                    min_time = min(min_time, value)
                time += min_time
            except Exception as e:
                count += 1
                print(count)
                print(cs)
                print(e)
        
        print("=================== user preference 0.4 =======================")
        print(f"hit type: {hit_type}")
        print(f"average hit type: {round(hit_type / len(schedule_df), 4)}")
        print(f"unfavored type: {unfavored_type}")
        print(f"average unfavored type: {round(unfavored_type / len(schedule_df), 4)}")
        print(f"average distance: {round(distance / len(schedule_df), 4)}")
        print(f"average time: {round(time / len(schedule_df), 4)}")


        # '''
        # 使用奐揚的 user preference, unfavored 0.3
        # '''
        with open('user_facility_perc_dic.json', 'r') as f:
            # Load JSON data from file
            user_facility_perc_dic = json.load(f)

        user_list = get_user_list(parser.parse("2018-07-01"))
        history_charging_df = charging_data[charging_data["createdNew"] < parser.parse("2018-07-01")].copy()
        history_charging_df["facilityType"] = history_charging_df["locationId"].apply(lambda x: location.loc[str(x), "FacilityType"])
        history_charging_df["createdHour"] = history_charging_df["createdHour"].astype(int)

        history_user_group = history_charging_df.groupby(["userId"])


        hit_type = 0
        unfavored_type = 0
        distance = 0
        time = 0

        for idx, raw in schedule_df.iterrows():
            
            facility_type_list = [int(i) for i in user_facility_perc_dic[raw["userID"]].keys()]
            facility_type = location.loc[raw["locationID"], "FacilityType"]
            if facility_type in facility_type_list and user_facility_perc_dic[raw["userID"]][str(facility_type)] >= 0.3:
                hit_type += 1
            if facility_type in facility_type_list and user_facility_perc_dic[raw["userID"]][str(facility_type)] < 0.3:
                unfavored_type += 1
            if facility_type not in facility_type_list:
                unfavored_type += 1
            
            min_distance = 1000
            count = 0
            try:
                for cs in user_preference[raw["userID"]]["locationId"]:
                    value = haversine(location.loc[raw["locationID"], ["Latitude", "Longitude"]].values, location.loc[cs, ["Latitude", "Longitude"]].values, unit = Unit.KILOMETERS)
                    min_distance = min(min_distance, value)
                distance += min_distance

                min_time = 1000
                for hour in user_preference[raw["userID"]]["createdHour"]:
                    value = abs(raw["datetime"].hour - hour)
                    min_time = min(min_time, value)
                time += min_time
            except Exception as e:
                count += 1
                print(count)
                print(cs)
                print(e)

        print("=================== user preference 0.3 =======================")
        print(f"hit type: {hit_type}")
        print(f"average hit type: {round(hit_type / len(schedule_df), 4)}")
        print(f"unfavored type: {unfavored_type}")
        print(f"average unfavored type: {round(unfavored_type / len(schedule_df), 4)}")
        print(f"average distance: {round(distance / len(schedule_df), 4)}")
        print(f"average time: {round(time / len(schedule_df), 4)}")


        # '''
        # 使用奐揚的 user preference, unfavored 0.4
        # '''
        with open('user_facility_perc_dic.json', 'r') as f:
            # Load JSON data from file
            user_facility_perc_dic = json.load(f)


        favor_count = []

        for idx, raw in schedule_df.iterrows():
            
            facility_type_list = [int(i) for i in user_facility_perc_dic[raw["userID"]].keys()]
            facility_type = location.loc[raw["locationID"], "FacilityType"]
            if facility_type in facility_type_list:
                favor_count.append(user_facility_perc_dic[raw["userID"]][str(facility_type)])
            else:
                favor_count.append(0)

        print("=================== favor ratio =======================")
        print(f"favor ratio: {round(sum(favor_count)/len(favor_count), 4)}")


    exit(0)
    ################################################################################################################################################
    ################################################################################################################################################
    ################################################################################################################################################
    ################################################################################################################################################
    ################################################################################################################################################
    

    for random_counter in range(1, 11):
        
        print(f"第 {random_counter} 次 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        columns=[
            "datetime", 
            "date", 
            "userID", 
            "locationID", 
            "buildingID", 
            "chargingLen", 
            "originLocationID", 
            "originChargingHour", 
            "incentive", 
            "score",
        ]

        schedule_df = pd.DataFrame([], columns=columns)
        testing_start_time = charging_start_time
        null_value = 0

        all_request_counts = 0
        count = 0

        for day in range(7):

            charging_request = get_charging_request(charging_data, testing_start_time)
            all_request_counts += len(charging_request)
            # recommend = pd.read_csv(PATH + f"{random_counter}/{testing_start_time.strftime('%m%d')}.csv")
            recommend = pd.read_csv(PATH + f"{testing_start_time.strftime('%m%d')}.csv")
            recommend["datetime"] = pd.to_datetime(recommend["datetime"], format="%Y-%m-%d %H:%M:%S")
            
            for request in charging_request:
                try:
                    schedule = recommend[recommend["requestID"] == request[0]]
        
                    if schedule["user_accept"].values[0]:
                        count += 1
                        schedule_df.loc[len(schedule_df)] = [
                            schedule["datetime"].values[0],
                            testing_start_time.strftime("%m%d"),
                            schedule["userID"].values[0],
                            str(schedule["locationID"].values[0]),
                            location.loc[str(schedule["locationID"].values[0]), "buildingID"],
                            schedule["chargingLen"].values[0],
                            schedule["originLocationID"].values[0],
                            schedule["originHour"].values[0],
                            schedule["incentive"].values[0],
                            schedule["score"].values[0]
                            ]
                except Exception as e:
                    print(request, e)
                    null_value += 1

            testing_start_time += timedelta(days=1)

        print(f"=============={TEST_NAME}=======================")
        print("accept count: ", count) 
        print("all requests: ", all_request_counts) 
        print("accepte rate: ", round(count/all_request_counts, 4))
        print("miss request: ", null_value)
        print("average incentive per charging station: ", schedule_df['incentive'].sum()/20)

        overage = 0
        electricity_price = pd.DataFrame([], columns=["basic_tariff", "current_tariff", "overload_penalty", "total"])
        schedule_statistic_count = pd.DataFrame([], columns=[charging_start_time + timedelta(hours=hour) for hour in range(0, 7*24)])

        overload_percentage = list()

        schedule_revenue = defaultdict()
        schedule_ev_charging_volume = list()
        schedule_electricity_cost = defaultdict() 
        schedule_ev_revenue = defaultdict()

        for cs in location.index:
            
            info = building_data[ (building_data["buildingID"] == location.loc[cs, "buildingID"]) &  (building_data["datetime"] >= building_start_time) & (building_data["datetime"] < building_end_time) ]

            ### EV charging data ###
            ev_info = [0 for i in range(24*7)]
            current = charging_start_time
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
            info["exceed"] = info["total"] - location.loc[cs, "contractCapacity"]
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
            basic_tariff, current_tariff, overload_penalty = calculate_electricity_price(location, cs, info)
            total_price = basic_tariff + current_tariff + overload_penalty
            electricity_price.loc[len(electricity_price)] = [basic_tariff, current_tariff, overload_penalty, total_price]
            schedule_revenue[cs] = (-1) * total_price + (CHARGING_FEE * info["charging"].sum())
            schedule_electricity_cost[cs] = total_price
            # print(f"revenue = {'{:.2f}'.format(schedule_revenue[cs])}")

            info["chargingCount"] = info["charging"].apply(lambda x: x/CHARGING_SPEED)
            info.set_index("datetime", inplace=True)
            schedule_statistic_count.loc[len(schedule_statistic_count)] = info["chargingCount"].T

        print(f"=============={TEST_NAME}=======================")
        print(f"average revenue = {float(sum(schedule_revenue.values()) / 20)}")
        print(electricity_price.mean(axis=0))

        
        import numpy as np

        testing_start_time = charging_start_time
        schedule_utilization = defaultdict(list)
        schedule_var = list()
        while testing_start_time < charging_end_time:
            for cs in range(20):
                truth_slots = truth_data.loc[(truth_data["datetime"] == testing_start_time) & (truth_data["buildingID"] == str(cs)), "parkingSlots"].values[0]
                
                days = (testing_start_time - charging_start_time).days
                seconds = (testing_start_time - charging_start_time).seconds
                schedule_slots = schedule_ev_charging_volume[cs][int(days * 24 + (seconds / 3600))] / CHARGING_SPEED
                schedule_utilization[testing_start_time].append(schedule_slots / truth_slots if truth_slots != 0 else 1)
                

            schedule_var.append(np.var(schedule_utilization[testing_start_time]))
            testing_start_time += timedelta(hours=1)

        schedule_variation = sum(schedule_var) / len(schedule_var)

        print("variation 1: ", round(schedule_variation, 4))

        # 遍歷每個時間點
        data = schedule_utilization

        # 初始化一個字典來存儲每個小時的利用率總和，以及一個數量計數器
        total_usage = defaultdict(list)

        # 遍历所有的时间点
        for dt, usage in data.items():
            # 检查这个时间点是否在我们所需的七天范围内
            for i in range(20):
                total_usage[i].append(usage[i])

        # 计算每个小时的平均利用率

        value_sum = 0
        for value in total_usage.values():
            value_sum += np.var(value)

        print("variation 2: ", round(value_sum/20, 4))


        from datetime import timedelta

        # 假設你的資料叫做 data
        data = schedule_utilization

        # 初始化一個列表來儲存每個充電站的總利用率
        total_usage = [0.0]*20

        # 初始化一個計數器來計算時間點的數量
        count = 0

        # 遍歷每個時間點
        for dt, usage in data.items():
            # 將該時間點的每個充電站的利用率加到總利用率中
            for i in range(20):
                total_usage[i] += usage[i]
            # 增加時間點的計數器
            count += 1

        # 計算每個充電站的平均利用率
        average_usage = [usage / count for usage in total_usage]

        print("Spqtioal variation: ", round(np.var(average_usage), 4))

        from collections import defaultdict
        from datetime import datetime

        # 假設你的資料叫做 data
        data = schedule_utilization

        # 初始化一個字典來存儲每個小時的利用率總和，以及一個數量計數器
        total_usage = list()

        # 遍历所有的时间点
        for dt, usage in data.items():

            total_usage.append(sum(usage)/20)

        print("Temporal variation 1: ", round(np.var(total_usage), 4))


        from collections import defaultdict
        from datetime import datetime

        # 假設你的資料叫做 data
        data = schedule_utilization

        # 初始化一個字典來存儲每個小時的利用率總和，以及一個數量計數器
        total_usage_by_hour = defaultdict(float)
        counts_by_hour = defaultdict(int)

        # 遍历所有的时间点
        for dt, usage in data.items():
            # 检查这个时间点是否在我们所需的七天范围内
            hour_key = dt.hour  # 创建一个关键字来表示小时
            total_usage_by_hour[hour_key] += (sum(usage)/20)
            counts_by_hour[hour_key] += 1

        # 计算每个小时的平均利用率
        average_usage_by_hour =[total_usage / counts_by_hour[hour] for hour, total_usage in total_usage_by_hour.items()]
        average_usage_by_hour

        print("Temporal Variation 2: ",round(np.var(average_usage_by_hour), 4))


        unfavored_type = 0
        hit_type = 0
        distance = 0
        time = 0

        for idx, raw in schedule_df.iterrows():
            
            if raw["userID"] == '603475':
                continue
            if location.loc[raw["locationID"], "FacilityType"] in user_preference[raw["userID"]]["facilityType"]:
                hit_type += 1
            if not location.loc[raw["locationID"], "FacilityType"] in user_preference[raw["userID"]]["facilityType"]:
                unfavored_type += 1
            
            min_distance = 1000
            for cs in user_preference[raw["userID"]]["locationId"]:
                value = haversine(location.loc[raw["locationID"], ["Latitude", "Longitude"]].values, location.loc[cs, ["Latitude", "Longitude"]].values, unit = Unit.KILOMETERS)
                min_distance = min(min_distance, value)
            distance += min_distance

            min_time = 1000
            for hour in user_preference[raw["userID"]]["createdHour"]:
                value = abs(raw["datetime"].hour - hour)
                min_time = min(min_time, value)
            time += min_time
        
        print("===================哲紜的 user preference=======================")
        print(f"average hit type: {round(hit_type / len(schedule_df), 4)}")
        print(f"average unfavored type: {round(unfavored_type / len(schedule_df), 4)}")
        print(f"average distance: {round(distance / len(schedule_df), 4)}")
        print(f"average time: {round(time / len(schedule_df), 4)}")


        import json
        with open('user_facility_perc_dic.json', 'r') as f:
            # Load JSON data from file
            user_facility_perc_dic = json.load(f)


        user_list = get_user_list(parser.parse("2018-07-01"))
        history_charging_df = charging_data[charging_data["createdNew"] < parser.parse("2018-07-01")].copy()
        history_charging_df["facilityType"] = history_charging_df["locationId"].apply(lambda x: location.loc[str(x), "FacilityType"])
        history_charging_df["createdHour"] = history_charging_df["createdHour"].astype(int)

        history_user_group = history_charging_df.groupby(["userId"])


        hit_type = 0
        unfavored_type = 0
        distance = 0
        time = 0

        for idx, raw in schedule_df.iterrows():
            
            facility_type_list = [int(i) for i in user_facility_perc_dic[raw["userID"]].keys()]
            facility_type = location.loc[raw["locationID"], "FacilityType"]
            if facility_type in facility_type_list and user_facility_perc_dic[raw["userID"]][str(facility_type)] >= 0.4:
                hit_type += 1
            if facility_type in facility_type_list and user_facility_perc_dic[raw["userID"]][str(facility_type)] < 0.4:
                unfavored_type += 1
            if facility_type not in facility_type_list:
                unfavored_type += 1
            
            min_distance = 1000
            count = 0
            try:
                for cs in user_preference[raw["userID"]]["locationId"]:
                    value = haversine(location.loc[raw["locationID"], ["Latitude", "Longitude"]].values, location.loc[cs, ["Latitude", "Longitude"]].values, unit = Unit.KILOMETERS)
                    min_distance = min(min_distance, value)
                distance += min_distance

                min_time = 1000
                for hour in user_preference[raw["userID"]]["createdHour"]:
                    value = abs(raw["datetime"].hour - hour)
                    min_time = min(min_time, value)
                time += min_time
            except Exception as e:
                count += 1
                print(count)
                print(cs)
                print(e)

        print("======================preference 0.4==============================")
        print(f"hit type: {hit_type}")
        print(f"average hit type: {round(hit_type / len(schedule_df), 4)}")
        print(f"unfavored type: {unfavored_type}")
        print(f"average unfavored type: {round(unfavored_type / len(schedule_df), 4)}")
        print(f"average distance: {round(distance / len(schedule_df), 4)}")
        print(f"average time: {round(time / len(schedule_df), 4)}")



        import json
        with open('user_facility_perc_dic.json', 'r') as f:
            # Load JSON data from file
            user_facility_perc_dic = json.load(f)

        user_list = get_user_list(parser.parse("2018-07-01"))
        history_charging_df = charging_data[charging_data["createdNew"] < parser.parse("2018-07-01")].copy()
        history_charging_df["facilityType"] = history_charging_df["locationId"].apply(lambda x: location.loc[str(x), "FacilityType"])
        history_charging_df["createdHour"] = history_charging_df["createdHour"].astype(int)

        history_user_group = history_charging_df.groupby(["userId"])


        hit_type = 0
        unfavored_type = 0
        distance = 0
        time = 0

        for idx, raw in schedule_df.iterrows():
            
            facility_type_list = [int(i) for i in user_facility_perc_dic[raw["userID"]].keys()]
            facility_type = location.loc[raw["locationID"], "FacilityType"]
            if facility_type in facility_type_list and user_facility_perc_dic[raw["userID"]][str(facility_type)] >= 0.3:
                hit_type += 1
            if facility_type in facility_type_list and user_facility_perc_dic[raw["userID"]][str(facility_type)] < 0.3:
                unfavored_type += 1
            if facility_type not in facility_type_list:
                unfavored_type += 1
            
            min_distance = 1000
            count = 0
            try:
                for cs in user_preference[raw["userID"]]["locationId"]:
                    value = haversine(location.loc[raw["locationID"], ["Latitude", "Longitude"]].values, location.loc[cs, ["Latitude", "Longitude"]].values, unit = Unit.KILOMETERS)
                    min_distance = min(min_distance, value)
                distance += min_distance

                min_time = 1000
                for hour in user_preference[raw["userID"]]["createdHour"]:
                    value = abs(raw["datetime"].hour - hour)
                    min_time = min(min_time, value)
                time += min_time
            except Exception as e:
                count += 1
                print(count)
                print(cs)
                print(e)

        print("======================preference 0.3==============================")
        print(f"hit type: {hit_type}")
        print(f"average hit type: {round(hit_type / len(schedule_df), 4)}")
        print(f"unfavored type: {unfavored_type}")
        print(f"average unfavored type: {round(unfavored_type / len(schedule_df), 4)}")
        print(f"average distance: {round(distance / len(schedule_df), 4)}")
        print(f"average time: {round(time / len(schedule_df), 4)}")


        # '''
        # 使用奐揚的 user preference, unfavored 0.4
        # '''
        with open('user_facility_perc_dic.json', 'r') as f:
            # Load JSON data from file
            user_facility_perc_dic = json.load(f)


        favor_count = []

        for idx, raw in schedule_df.iterrows():
            
            facility_type_list = [int(i) for i in user_facility_perc_dic[raw["userID"]].keys()]
            facility_type = location.loc[raw["locationID"], "FacilityType"]
            if facility_type in facility_type_list:
                favor_count.append(user_facility_perc_dic[raw["userID"]][str(facility_type)])
            else:
                favor_count.append(0)

        print("======================favor ratio==============================")
        print(f"favor ratio: {round(sum(favor_count)/len(favor_count), 4)}")
        print("\n")
        print("\n")
        exit(0)

    

    