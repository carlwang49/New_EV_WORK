from UserBehavior import UserBehavior
import os
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from dateutil import parser
from collections import defaultdict, deque
from base import base

TESTING_NAME = "FCFS_RR_V2"
TESTING_START_DATE = parser.parse("2018-07-01")
TESTING_END_DATE = parser.parse("2018-07-08")

MAX_INCENTIVE_NUMS = 6
INCENTIVE_NUMS = 0
MU = 0
SIGMA_SQUARE = 1

PATH = f"../NewResult/Baseline/{TESTING_NAME}/"


class FCFS_RR_V2(base):
    
    def __init__(self):

        super().__init__()
        self.current_date = None
        self.incentive_cost = [0.1, 0.12, 0.16, 0.25, 0.4]
        self.user_most_prefer_csID = pd.read_csv("../Dataset/charging_data_2_move.csv")
        self.run_user_most_prefer_csID()
        self.location_queue = deque(self.location.index.to_list())


    def update_user_selection(self, slots_df, date, csID, hour, charging_len):

        slots_df.loc[(slots_df["buildingID"] == self.location.loc[csID, "buildingID"]) &
                     (slots_df["datetime"] >= (date + timedelta(hours = hour))) &
                     (slots_df["datetime"] < (date + timedelta(hours = hour + charging_len))), "parkingSlots"] -= 1
        
        return slots_df


    def run_user_most_prefer_csID(self):
        
        self.user_most_prefer_csID["createdNew"] = pd.to_datetime(self.user_most_prefer_csID["createdNew"])
        self.user_most_prefer_csID = self.user_most_prefer_csID[self.user_most_prefer_csID["createdNew"] < TESTING_START_DATE]


    def FCFS_RR(self, slots_df, userID, charging_len):

        most_charging_time_per_user = self.charging_data.groupby('userId')['createdHour'].agg(lambda x: x.value_counts().index[0])
        user_most_common_hour = int(most_charging_time_per_user[userID])
        hour = user_most_common_hour

        for _ in range(20):
            csID = self.location_queue[0]
            for _ in range(0, 24):
                hour %= 24
                parking_slots = self.get_residual_slots(slots_df, csID, hour, charging_len)
                if parking_slots > 0:
                    self.location_queue.rotate(-1)
                    return csID, hour
                hour += 1
            self.location_queue.rotate(-1)


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
    model = FCFS_RR_V2()
    user_behavior = UserBehavior()
    model.current_date = TESTING_START_DATE
    user_choose_station = defaultdict(lambda:0)

    for day in range(7):

        user_list = model.get_user_list(model.current_date)
        slots_df = model.get_parking_slots(model.building_predict_data, model.current_date)
        charging_request = model.get_charging_request(model.current_date)

        columns = [
            "requestID", 
            "userID", 
            "datetime", 
            "locationID", 
            "chargingLen", 
            "originLocationID", 
            "originHour",
            "x_value"
        ]

        schedule_df = pd.DataFrame([], columns=columns)
        charging_request = sorted(charging_request, key=lambda x: x[3])

        for requestID, userID, charging_len, origin_hour, origin_cs in charging_request:
        
            try:
                current_round = model.FCFS_RR(slots_df, userID, charging_len)
                round_csID, round_hour = current_round if current_round else model.get_origin_request(origin_cs, origin_hour, slots_df, charging_len)

                incentive_norm = INCENTIVE_NUMS / MAX_INCENTIVE_NUMS
                factor_time = user_behavior.factor_time(round_hour, userID)
                factor_cate = user_behavior.factor_cate(model.location, round_csID, userID)
                factor_dist = user_behavior.factor_dist(model.location, model.user_most_prefer_csID, round_csID, userID)
                dissimilarity = user_behavior.get_dissimilarity(factor_time, factor_cate, factor_dist)
                x = user_behavior.get_distribution_x(dissimilarity, incentive_norm)
                y = user_behavior.get_norm_y(x, MU, SIGMA_SQUARE)
                user_accept = user_behavior.get_user_decision(y)
                round_csID, round_hour = (round_csID, round_hour) if user_accept else model.get_origin_request(origin_cs, origin_hour, slots_df, charging_len)
                
                user_choose_station[round_csID] += 1
                print("user_choose =", (round_csID, round_hour, charging_len))
                schedule_df.loc[len(schedule_df)] = [
                    requestID,
                    userID,
                    model.current_date + timedelta(hours=round_hour),
                    round_csID,
                    charging_len,
                    origin_cs,
                    origin_hour,
                    x
                ]
                slots_df = model.update_user_selection(slots_df, model.current_date, round_csID, round_hour, charging_len)

            except Exception as e:
                    print(f"{userID}, {charging_len}, {origin_hour}, {origin_cs} ERROR: {e}")

        for item in user_choose_station.keys():
            print(item, "-", model.location.loc[item, "buildingID"], ":", user_choose_station[item])

        path = PATH
        if not os.path.isdir(path):
            os.makedirs(path)

        schedule_df.to_csv(path + f"{model.current_date.strftime('%m%d')}.csv", index=None)

        print(f"========== {day+1} done ==========")
        model.current_date+= timedelta(days=1)

    end = time.time()
    print(f"Time: {(end-start)/60} min")

