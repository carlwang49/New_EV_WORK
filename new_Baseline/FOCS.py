'''
每個 request
# create_time: 為每個request在testing data的原充電時間
# arrival_time: 預設為create_time往後一小時
# format: {"create_time":, "arrival_time":, "uid":, "remaining_charging_hours":}
## "remaining_charging_hours" default 值為原本user需要充電的時數
'''

'''
Evaluation 方式
FOCS 會將一個request分好幾次充完, 因此一個 request會對應多個推薦, 每個推薦都需要記錄下來做evaluation
'''
import itertools
import random
import os
import pandas as pd
import time
from datetime import timedelta, datetime
from dateutil import parser
from collections import defaultdict, deque
from UserBehavior import UserBehavior
from base import base

TESTING_NAME = "FOCS_test"
TEST_DAY = "2023-07-09"
SIGMOID_INCENTIVE_UNIT = 0.2
INCENTIVE_NUMS = 0

TESTING_START_DATE = parser.parse("2018-07-01")
TESTING_END_DATE = parser.parse("2018-07-08")

CHARGING_SPEED = 10  # 10 kW
CHARGING_FEE = 0.3  # $ 0.3/kWh


class FOCS(base):


    def __init__(self):

        super().__init__()
        self.current_date = None
        self.electricity_price_dic = defaultdict(float)
        self.user_behavior = UserBehavior()
        self.columns=["requestID", "userID", "datetime", "locationID", "chargingLen", "originLocationID", "originHour", "user_accept", "remain_hour"]
        self.user_choose_station = defaultdict(lambda:0)
        self.schedule_df = None
        self.get_electricity_price_dic()


    def get_electricity_price_dic(self):

        weekend_price = [0.056] * 24
        weekdays_price = [0.056] * 8 + [0.092] * 4 + [0.267] * 6 + [0.092] * 5 + [0.056] * 1

        for weekday in range(1, 8):
            self.electricity_price_dic[weekday] = weekdays_price if weekday < 6 else weekend_price
    

    def record_schedule_df(self, requestID, userID, arrival_time, cs, charging_len, origin_cs, origin_hour, current_date, slots_df, user_accept, remain_hour):

        self.schedule_df.loc[len(self.schedule_df)] = [
                requestID,
                userID,
                current_date + timedelta(hours=arrival_time),
                cs,
                charging_len,
                origin_cs,
                origin_hour,
                user_accept,
                remain_hour
            ]
        
        self.user_choose_station[cs] += 1
        slots_df = self.update_user_selection(slots_df, model.current_date, cs, arrival_time, charging_len)

        return slots_df


    def cs_available(self, request, slots_df):
        
        arrival_time = request["arrival_time"]
        cs_available_list = []
        for cs in self.location.index:
            if self.get_residual_slots(slots_df, cs, arrival_time, 1) > 0:
                cs_available_list.append(cs)
        
        return cs_available_list
    

    def user_accepts(self, request:dict, recommend_csID) -> bool: 
        '''
        计算用户是否接受推荐的充电站和充电时间。
        '''
        user_behavior = self.user_behavior
        factor_time = user_behavior.factor_time(request["arrival_time"], request["userID"], self.charging_data)
        factor_cate = user_behavior.factor_cate(self.location, recommend_csID, request["userID"])
        factor_dist = user_behavior.factor_dist(self.location, self.charging_data, recommend_csID, request["userID"], TESTING_START_DATE)
        dissimilarity = user_behavior.get_dissimilarity(factor_time, factor_cate, factor_dist)
        prob = user_behavior.estimate_willingeness(dissimilarity, INCENTIVE_NUMS, SIGMOID_INCENTIVE_UNIT)
        user_accept = user_behavior.get_user_decision(prob)

        return user_accept


    def choose_cs(self, request: dict, cs_available_list: list):
        '''
        若該 request EV 的 arrival 仍有cs有空位 (此處空位指的是實體充電樁數量, 而非 prefer parking slot)
        則選擇一個仍有空位且先前尚未推薦的cs (若全都推薦過則random選有空位的cs
        '''
        # 從 request 中取出已推薦過的 CS
        recommended_CS = request["recommended_CS"]

        # 從有空位的 CS 中篩選出尚未推薦過的 CS
        not_recommended_CS = [cs for cs in cs_available_list if cs not in recommended_CS]

        # 若還有尚未推薦過的 CS，則從中隨機選取一個
        if not_recommended_CS:
            chosen_CS = random.choice(not_recommended_CS)
        
        # 若所有有空位的 CS 都已推薦過，則從所有有空位的 CS 中隨機選取一個
        else:
            chosen_CS = random.choice(cs_available_list)

        # 將選取的 CS 加入到已推薦過的 CS 列表中
        request["recommended_CS"].append(chosen_CS)

        return chosen_CS


    def update_user_selection(self, slots_df, date, csID, hour, charging_len):

        slots_df.loc[(slots_df["buildingID"] == self.location.loc[csID, "buildingID"]) &
                     (slots_df["datetime"] >= (date + timedelta(hours = hour))) &
                     (slots_df["datetime"] < (date + timedelta(hours = hour + charging_len))), "parkingSlots"] -= 1
        
        return slots_df
    

    def update_unit_value(self, request: dict, current_date: datetime):
        '''
        unit_value: (完成此 request 得到的充電收益 - arrival_time的時間電價 * remaining_charging_hours) / remaining_charging_hours
        Arrival time: origin_hour + 1 hour
        '''
        day = current_date.isoweekday()
        arrival_time = request["arrival_time"] # 需確定一下是否是固定的
        arrival_time_electricity_price = self.electricity_price_dic[day][arrival_time]
        charging_profit = request["charging_len"] * CHARGING_SPEED * CHARGING_FEE
        remaining_charging_hours = request["remaining_charging_hours"]
        unit_value = (charging_profit - arrival_time_electricity_price * remaining_charging_hours) / remaining_charging_hours
        
        return unit_value

    def compute_initial_unit_value(self, request):
        '''
        request: dict
        default unit_value: (完成此request得到的充電收益) / 該request的remaining_charging_hours
        charging_profit: 充電小時數 * 充電速率 * 充電費用
        '''
        charging_profit = request["charging_len"] * CHARGING_SPEED * CHARGING_FEE
        unit_value = charging_profit / request["remaining_charging_hours"]

        return unit_value
    

    def get_origin_request(self, csID, hour, slots_df, charging_len):

        for _ in range(24):
            hour %= 24
            parking_slots = self.get_residual_slots(slots_df, csID, hour, charging_len)
            if parking_slots <= 0:
                hour += 1
            else:
                return csID, hour


    def FOCS(self, all_requests, slots_df, current_date):
        '''
        request_list: [(requestID, userID, charging_len, origin_hour, origin_cs), ...]
        all_requests: 一天所有的 request
        '''
        print(f"======================={current_date}=======================")
        self.schedule_df = pd.DataFrame([], columns=self.columns)
        previous_incomplete_list = []
        unit_value_dic = defaultdict(float)
        keys = ["requestID", "userID", "charging_len", "origin_hour", "origin_cs"]
        all_requests_dicts = [{**{key: value for key, value in zip(keys, request)}, 
                                "remaining_charging_hours": request[2], 
                                "arrival_time": (request[3] + 1) % 24,
                                "recommended_CS": []} for request in all_requests]

        all_timestep_cycle = itertools.cycle(range(24))
        
        user_list = []
        count = 0
        while True:
            
            t = next(all_timestep_cycle) # 從 0 開始到 23
            current_received_request = [request for request in all_requests_dicts if request["origin_hour"] == t] if t < 24 else []

            # 然後從 all_requests_dicts 中移除 current_received_request 中的字典
            all_requests_dicts = [request for request in all_requests_dicts if request not in current_received_request]

            incomplete_request_list = current_received_request + previous_incomplete_list
            if not incomplete_request_list and count > 23:
                break
            
            previous_incomplete_list = [] # 清空 previous_incomplete_list

            # 計算 incomplete_request_list 所有的 unit_value
            for r in incomplete_request_list:
                
                # 若某 request 還未計算 unit_value
                requestID = r["requestID"]
                if requestID not in unit_value_dic:
                    unit_value_dic[requestID] = self.compute_initial_unit_value(r)
                
                # 若某 request 在之前時段已計算過 unit_value
                else:
                    unit_value_dic[requestID] = self.update_unit_value(r, current_date)

            sorted_unit_value_list = sorted(incomplete_request_list, 
                                            key=lambda x: unit_value_dic[x["requestID"]], reverse=True)

            for r in sorted_unit_value_list:
                
                cs_available_list = self.cs_available(r, slots_df)
         
                if cs_available_list: 
                    
                    cs = self.choose_cs(r, cs_available_list) 
                    
                    ## 若使用者接受充電方案
                    user_accept = self.user_accepts(r, cs)
                    if user_accept: 
                        
                        if r['remaining_charging_hours'] > 1:
                            
                            slots_df = self.record_schedule_df(r["requestID"], r["userID"], r["arrival_time"], 
                                                    cs, 1, r["origin_cs"], r["origin_hour"], 
                                                    current_date, slots_df, user_accept, r['remaining_charging_hours'])
                            
                            print("user_choose=", (r["requestID"], r["userID"], cs, r['arrival_time'], 1))
                            
                            r['remaining_charging_hours'] -= 1
                            r['arrival_time'] = (r['arrival_time'] + 1) % 24
                            previous_incomplete_list.append(r)
                            
                            # 紀錄      
        
                        else:
                            ## 此 request 已充完電
                            # 紀錄
                            print("user_choose =", (r["requestID"], r["userID"] ,cs, r['arrival_time'], 1))
                            slots_df = self.record_schedule_df(r["requestID"], r["userID"], r['arrival_time'], 
                                            cs, 1, r["origin_cs"], r["origin_hour"], 
                                            current_date, slots_df, user_accept, r['remaining_charging_hours'])
                    
                    ## 若使用者不接受充電方案
                    else:
                        # 不更新 request 的資訊，直接當作此 request 已不接受任何排程
                        # 也就是把他選擇他原本要充電的 CS, 然後把他衝完
                        # 若原本位置滿了就shift一小時
                        cs, arrival_time = self.get_origin_request(cs, r["arrival_time"], slots_df, r["remaining_charging_hours"])
                        # 紀錄
                        if r["requestID"] in user_list:
                            print(sorted_unit_value_list)
                            print(r)
                            exit(0)
                        print("user_choose=", (r["requestID"], r["userID"], cs, arrival_time, r["remaining_charging_hours"]))
                        slots_df = self.record_schedule_df(r["requestID"], r["userID"], arrival_time, 
                                    cs, r["remaining_charging_hours"], r["origin_cs"], r["origin_hour"], 
                                    current_date, slots_df, user_accept, r['remaining_charging_hours'])
                        user_list.append(r["requestID"])  
            
                else:
                    r['arrival_time'] = (r['arrival_time'] + 1) % 24 
                    previous_incomplete_list.append(r)
            
            count += 1
            
if __name__ == "__main__":

    start = time.time()
    
    model = FOCS()
    model.current_date = TESTING_START_DATE

    for day in range(7):

        user_list = model.get_user_list(model.current_date)
        slots_df = model.get_parking_slots_without_predition(model.building_truth_data, model.current_date)
        charging_request = model.get_charging_request(model.current_date)
        charging_request = sorted(charging_request, key=lambda x: x[3])
        model.FOCS(charging_request, slots_df, model.current_date)

        for item in model.user_choose_station.keys():
            print(item, "-", model.location.loc[item, "buildingID"], ":", model.user_choose_station[item])

        path = f"../Result/Carl/new_Baseline/{TESTING_NAME}/{TEST_DAY}/SIGMOID_INCENTIVE_UNIT_{SIGMOID_INCENTIVE_UNIT}/"
        os.makedirs(path, exist_ok=True)
        model.schedule_df.to_csv(path + f"{model.current_date.strftime('%m%d')}.csv", index=None)

        print(f"========== {day+1} done ==========")
        model.current_date+= timedelta(days=1)

    end = time.time()
    print(f"Time: {(end-start)/60} min")
