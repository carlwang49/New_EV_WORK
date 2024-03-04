import os
import pandas as pd
import time
import numpy as np
from datetime import timedelta
from dateutil import parser
from collections import defaultdict
from base import base
from UserBehavior import UserBehavior

TESTING_NAME = "unscheduled_userBehavior_without_prediction_maxSlots_10"
TEST_DAY = "2024-03-04"

TESTING_START_DATE = parser.parse("2018-07-01")
TESTING_END_DATE = parser.parse("2018-07-08")

INCENTIVE_NUMS = 0
PATH = f"../NewResult/Baseline/{TESTING_NAME}/{TEST_DAY}/"


class Unschedule(base):
    
    def __init__(self):

        super().__init__()
        self.current_date = None


    def update_user_selection(self, slots_df, date, csID, hour, charging_len):

        slots_df.loc[(slots_df["buildingID"] == self.location.loc[csID, "buildingID"]) &
                     (slots_df["datetime"] >= (date + timedelta(hours = hour))) &
                     (slots_df["datetime"] < (date + timedelta(hours = hour + charging_len))), "parkingSlots"] -= 1
        
        return slots_df
    

    def get_origin_request(self, csID, hour, slots_df, charging_len):

        parking_slots = self.get_residual_slots(slots_df, csID, 23, charging_len)

        for _ in range(24):
            hour %= 24
            parking_slots = self.get_residual_slots(slots_df, csID, hour, charging_len)
            if parking_slots <= 0:
                hour += 1
            else:
                return csID, hour
        

if __name__ == "__main__":
    
    start = time.time()
    model = Unschedule()
    model.current_date = TESTING_START_DATE
    user_choose_station = defaultdict(lambda:0)
    
    for day in range(7):

        slots_df = model.get_parking_slots_without_predition(model.building_truth_data, model.current_date)
        charging_request = model.get_charging_request(model.current_date)
        charging_request = sorted(charging_request, key=lambda x: x[3])
        
        keys = ["requestID", "userID", "charging_len", "origin_hour", "origin_cs"]
        keys = ["requestID", "userID", "charging_len", "origin_hour", "origin_cs"]
        all_requests_dicts = [{**{key: value for key, value in zip(keys, request)}, 
                           "remaining_charging_hours": request[2]} for request in charging_request]
        
        columns = [
            "requestID", 
            "userID", 
            "datetime", 
            "locationID", 
            "chargingLen", 
            "originLocationID", 
            "originHour",
            "user_accept"
        ]

        schedule_df = pd.DataFrame([], columns=columns)

        for requestID, userID, charging_len, origin_hour, origin_cs in charging_request:
            
            # try:
            recommend_csID, recommend_hour = model.get_origin_request(origin_cs, origin_hour, slots_df, charging_len)

            user_choose_station[recommend_csID] += 1
            print("user_choose =", (recommend_csID, recommend_hour, charging_len))
            schedule_df.loc[len(schedule_df)] = [
                requestID,
                userID,
                model.current_date + timedelta(hours=recommend_hour),
                recommend_csID,
                charging_len,
                origin_cs,
                origin_hour,
                None
            ]
            slots_df = model.update_user_selection(slots_df, model.current_date, recommend_csID, recommend_hour, charging_len)
            # except Exception as e:
            #     print(f"{userID}, {charging_len}, {origin_hour}, {origin_cs} ERROR: {e}")

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

