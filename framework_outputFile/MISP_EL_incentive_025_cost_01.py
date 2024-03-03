'''
MISP + EL + output
'''
import os
import pandas as pd
import numpy as np
import time
from datetime import timedelta
from dateutil import parser
from collections import defaultdict
from MISP import MISP

### global variable ###
ALPHA = 0.5
TESTING_NAME = "MISP_EL_incentive_025_cost_01"
TEST_DAY = "2023-06-27"

INCENTIVE_BUDGET = 400
TESTING_START_DATE = parser.parse("2018-07-01")
TESTING_END_DATE = parser.parse("2018-07-08")

MU = 0
SIGMA_SQUARE = 1


class EL_MISP(MISP):

    def __init__(self):
        super().__init__()
        self.charging_csID_avg_request_nums = None
        self.set_charging_csID_avg_request_nums()
       

    def set_charging_csID_avg_request_nums(self):
        
        charging_one_month_ago_data = self.charging_data[(self.charging_data["createdNew"] < parser.parse("2018-07-01")) 
                                                            & (self.charging_data["createdNew"] >= parser.parse("2018-06-01"))].copy()
        self.charging_csID_avg_request_nums = charging_one_month_ago_data.groupby("locationId").size().reset_index(name='request_count')
        self.charging_csID_avg_request_nums['request_count'] = (self.charging_csID_avg_request_nums['request_count'] / 24).astype(int)
        self.charging_csID_avg_request_nums['request_count'] = self.charging_csID_avg_request_nums['request_count'].apply(lambda x: max(x, 1))
    


    def get_all_combinations(self, user_list, slots_df, userID, charging_len):
        '''
        取得所有可能的充電組合
        user_list: 所有使用者 ID
        slots_df: 充電位資訊
        userID: 要排程的那個使用者 ID
        charging_len: 使用者充電時間長度
        '''

        combinations = list()
        for csID in self.location.index:
            for hour in range(24):
                
                ### check residual parking slots ###
                parking_slots = self.get_residual_slots(slots_df, csID, hour, charging_len)
                if parking_slots <= 0:
                    continue

                # N_i = self.charging_csID_avg_request_nums.loc[self.charging_csID_avg_request_nums['locationId'] == csID, 'request_count'].values[0]
                csID_curr_budget = self.budget[csID]
                # coupon = min(max(csID_curr_budget // N_i, 1), 10)
                coupon = 3
                if coupon > csID_curr_budget:
                    continue
                
                score, willingness, personal = self.expected_score(user_list, userID, csID, hour, coupon)
                schedule_cost = self.incentive_cost[coupon]
                combinations.append((csID, hour, score, schedule_cost, personal, willingness))
        
        return combinations 


if __name__ == "__main__":
        
    start = time.time()
    model = EL_MISP()
    model.current_date = TESTING_START_DATE
    user_choose_station = defaultdict(lambda: 0)

    for day in range(7):

        ### User interaction matrix ###
        user_list = model.get_user_list(model.current_date)
        model.get_user_interaction_value(user_list)

        ### Spendable parking slots prediction ###
        slots_df = model.get_parking_slots(model.building_predict_data, model.current_date)
        slots_df = model.reserve_parking_slots(slots_df, model.current_date)

        ### Utilization ###
        model.calculate_utilization(slots_df=slots_df, first=True)
        model.average_utilization()

        ### Get building budget and threshold ###
        model.set_budgets()

        ### EV charging request ###
        charging_request = model.get_charging_request(model.current_date)

        ### schedule ###
        progress = 1
        columns = [
            "requestID", 
            "userID", 
            "datetime", 
            "locationID", 
            "chargingLen", 
            "score", 
            "incentive", 
            "originLocationID", 
            "originHour",
            "threshold",
            "personal",
            "willingness",
        ]

        schedule_df = pd.DataFrame([], columns=columns)
        
        for requestID, userID, charging_len, origin_hour, origin_cs in charging_request:
            
            try:
                recommend = model.OGAP(user_list, slots_df, userID, charging_len) # (csID, hour, score, schedule_cost)
                user_choose = model.choose_recommend(recommend)
                
                user_choose_station[user_choose[0]] += 1
                print("user_choose =", user_choose)

                schedule_df.loc[len(schedule_df)] = [
                    requestID,
                    userID,
                    model.current_date +
                    timedelta(hours=user_choose[1]), # 使用者選擇的時間
                    user_choose[0], # 使用者選擇的充電站
                    charging_len,
                    user_choose[2], # 分數
                    model.incentive_cost.index(user_choose[3]), # 使用者使用的 incentive 
                    origin_cs,
                    origin_hour,
                    user_choose[6],
                    user_choose[4],
                    user_choose[5]
                ]
                
                slots_df = model.update_user_selection(slots_df, model.current_date, user_choose, charging_len)
                model.calculate_utilization(schedule=user_choose, charging_len=charging_len)
                model.average_utilization()
                print(f"progress: {progress}/{len(charging_request)}")

            except Exception as e:
                print(f"{userID}, {charging_len}, {origin_hour}, {origin_cs} ERROR: {e}")
            
            progress += 1

        for item in user_choose_station.keys():
            print(item, "-", model.location.loc[item, "buildingID"], ":", user_choose_station[item])

        path = f"../Result/Carl/MISP/{TESTING_NAME}/{TEST_DAY}/alpha_{ALPHA}/"

        if not os.path.isdir(path):
            os.makedirs(path)

        schedule_df.to_csv(path + f"{model.current_date.strftime('%m%d')}.csv", index=None)

        print(f"{day+1} default count: {model.default_count}")
        print(f"========== {day+1} done ==========")
        model.current_date += timedelta(days=1)

        # update average request number
        model.update_average_request_num(model.current_date, user_choose_station)

    end = time.time()
    print(f"Time: {(end-start)/60} min")
