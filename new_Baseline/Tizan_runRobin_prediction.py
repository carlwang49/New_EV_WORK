import os
import pandas as pd
import time
import numpy as np
from datetime import timedelta
from dateutil import parser
from collections import defaultdict
from base import base
from UserBehavior import UserBehavior

TESTING_NAME = "runRobin_userBehavior_prediction"
TEST_DAY = "2023-07-05"
SIGMOID_INCENTIVE_UNIT = 0.2

TESTING_START_DATE = parser.parse("2018-07-01")
TESTING_END_DATE = parser.parse("2018-07-08")

INCENTIVE_NUMS = 0


class RunRobin(base):
    
    def __init__(self):

        super().__init__()
        self.current_date = None
        self.round_index = 0
        self.schedule_order = list()

    def init_schdeule_order(self):

        for hour in range(24):
            for cs in range(20):
                self.schedule_order.append([cs, hour])


    def cal_parking_slots_nums(self, slots_num, slots_df):

        for buildingID in range(20):
            for hour in range(24):
                mask = ((slots_df["buildingID"] == str(buildingID)) \
                    & (slots_df["datetime"] == (self.current_date + timedelta(hours=hour))))
                slots_num[buildingID][hour] = slots_df.loc[mask, "parkingSlots"].values[0]
        
        return slots_num

    def update_user_selection(self, slots_df, date, csID, hour, charging_len):

        slots_df.loc[(slots_df["buildingID"] == self.location.loc[csID, "buildingID"]) &
                     (slots_df["datetime"] >= (date + timedelta(hours = hour))) &
                     (slots_df["datetime"] < (date + timedelta(hours = hour + charging_len))), "parkingSlots"] -= 1
        
        return slots_df
    

    def get_runRobin_result(self, charging_len, slots_num):

        check_slots = False
        while not check_slots:
            # 檢查時間夠不夠充
            if (self.schedule_order[self.round_index][1] + charging_len) > 24: # schedule_order[round_index][1]: 充電起始小時
                self.round_index = (self.round_index + 1) % len(self.schedule_order)
                continue
            
            # 檢查有沒有充電位置
            # print(self.schedule_order)
            
            if not slots_num[self.schedule_order[self.round_index][0]][self.schedule_order[self.round_index][1]:(self.schedule_order[self.round_index][1] + charging_len)].all():
                self.round_index = (self.round_index + 1) % len(self.schedule_order)
            else: 
                slots_num[self.schedule_order[self.round_index][0]][self.schedule_order[self.round_index][1]: (self.schedule_order[self.round_index][1]+charging_len)] -= 1
                check_slots = True
        
        return slots_num


    def get_origin_request(self, csID, hour, slots_df, charging_len):

        for _ in range(24):
            hour %= 24
            parking_slots = self.get_residual_slots(slots_df, csID, hour, charging_len)
            if parking_slots <= 0:
                hour += 1
            else:
                return csID, hour
        

if __name__ == "__main__":
    
    start = time.time()
    model = RunRobin()
    user_behavior = UserBehavior()
    model.current_date = TESTING_START_DATE
    user_choose_station = defaultdict(lambda:0)
    model.init_schdeule_order()
    
    for day in range(7):

        slots_num = np.zeros((20, 24)) # 每個充電站的充電位數量
        slots_df = model.get_parking_slots(model.building_predict_data, model.current_date)
        slots_num = model.cal_parking_slots_nums(slots_num, slots_df)
        charging_request = model.get_charging_request(model.current_date)
        
        columns = [
            "requestID", 
            "userID", 
            "datetime", 
            "locationID", 
            "chargingLen", 
            "originLocationID", 
            "originHour",
        ]

        schedule_df = pd.DataFrame([], columns=columns)
        for requestID, userID, charging_len, origin_hour, origin_cs in charging_request:
            
            # try:
            slots_num = model.get_runRobin_result(int(charging_len), slots_num)
            runRobin_csID = model.location.loc[model.location["buildingID"] ==\
                                               str(model.schedule_order[model.round_index][0])].index[0]
            runRobin_hour = int(model.schedule_order[model.round_index][1])
        
            recommend_csID, recommend_hour = runRobin_csID, runRobin_hour
            factor_time = user_behavior.factor_time(recommend_hour, userID, model.charging_data)
            factor_cate = user_behavior.factor_cate(model.location, recommend_csID, userID)
            factor_dist = user_behavior.factor_dist(model.location, model.charging_data, recommend_csID, userID, TESTING_START_DATE)
            print("factor_time, factor_cate,  factor_dist: ", factor_time, factor_cate, factor_dist)
            dissimilarity = user_behavior.get_dissimilarity(factor_time, factor_cate, factor_dist)
            prob = user_behavior.estimate_willingeness(dissimilarity, INCENTIVE_NUMS, SIGMOID_INCENTIVE_UNIT)
            user_accept = user_behavior.get_user_decision(prob)
            print("probability: ", prob)
            print("user_accept: ", user_accept)
            if not user_accept:
                recommend_csID, recommend_hour =  model.get_origin_request(origin_cs, origin_hour, slots_df, charging_len)
            

            user_choose_station[recommend_csID] += 1
            print("user_choose =", (recommend_csID, recommend_hour, charging_len))
            print(runRobin_csID, runRobin_hour)
            schedule_df.loc[len(schedule_df)] = [
                requestID,
                userID,
                model.current_date + timedelta(hours=recommend_hour),
                recommend_csID,
                charging_len,
                origin_cs,
                origin_hour,
            ]
            slots_df = model.update_user_selection(slots_df, model.current_date, recommend_csID, recommend_hour, charging_len)
            model.round_index = (model.round_index + 1) % len(model.schedule_order)
            # except Exception as e:
            #     print(f"{userID}, {charging_len}, {origin_hour}, {origin_cs} ERROR: {e}")

        for item in user_choose_station.keys():
            print(item, "-", model.location.loc[item, "buildingID"], ":", user_choose_station[item])

        path = f"../Result/Carl/new_Baseline/{TESTING_NAME}/{TEST_DAY}/SIGMOID_INCENTIVE_UNIT_{SIGMOID_INCENTIVE_UNIT}/"
        
        if not os.path.isdir(path):
            os.makedirs(path)
        
        schedule_df.to_csv(path + f"{model.current_date.strftime('%m%d')}.csv", index=None)

        print(f"========== {day+1} done ==========")
        model.current_date+= timedelta(days=1)

    end = time.time()
    print(f"Time: {(end-start)/60} min")

