'''
PRE without userBehavior
'''
import os
import pandas as pd
import time
from datetime import timedelta
from dateutil import parser
from collections import defaultdict, deque
from base import base

TESTING_NAME = "FCFS_RR_V1_without_userBehavior"
TEST_DAY = "2023-06-28"

TESTING_START_DATE = parser.parse("2018-07-01")
TESTING_END_DATE = parser.parse("2018-07-08")


class FCFS_RR_V1(base):
    
    def __init__(self):

        super().__init__()
        self.current_date = None
        self.user_most_prefer_csID = pd.read_csv("../Dataset/charging_data_2_move.csv")
        self.run_user_most_prefer_csID()
        self.location_queue = deque(self.location.index.to_list())


    def update_user_selection(self, slots_df, date, csID, hour, charging_len):

        slots_df.loc[(slots_df["buildingID"] == self.location.loc[csID, "buildingID"]) &
                     (slots_df["datetime"] >= (date + timedelta(hours = hour))) &
                     (slots_df["datetime"] < (date + timedelta(hours = hour + charging_len))), "parkingSlots"] -= 1
        
        return slots_df


    def FCFS_RR(self, slots_df, charging_len):

        for _ in range(20):
            csID = self.location_queue[0]
            for hour in range(0, 24):
                parking_slots = self.get_residual_slots(slots_df, csID, hour, charging_len)
                if parking_slots > 0:
                    self.location_queue.rotate(-1)
                    return csID, hour

            self.location_queue.rotate(-1)


    def get_origin_request(self, csID, hour, slots_df, charging_len):

        for _ in range(24):
            hour %= 24
            parking_slots = self.get_residual_slots(slots_df, csID, hour, charging_len)
            if parking_slots <= 0:
                hour += 1
            else:
                return csID, hour
            

    def run_user_most_prefer_csID(self):
        
        self.user_most_prefer_csID["createdNew"] = pd.to_datetime(self.user_most_prefer_csID["createdNew"])
        self.user_most_prefer_csID = self.user_most_prefer_csID[self.user_most_prefer_csID["createdNew"] < TESTING_START_DATE]


if __name__ == "__main__":

    start = time.time()
    model = FCFS_RR_V1()
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
        ]
    
        schedule_df = pd.DataFrame([], columns=columns)
        charging_request = sorted(charging_request, key=lambda x: x[3])


        for requestID, userID, charging_len, origin_hour, origin_cs in charging_request:

            try:
                if model.FCFS_RR(slots_df, charging_len):
                    round_csID, round_hour = model.FCFS_RR(slots_df, charging_len)
                else:
                    round_csID, round_hour = model.get_origin_request(origin_cs, origin_hour, slots_df, charging_len)
                
                user_choose_station[round_csID] += 1
                print("user_choose =", (round_csID, round_hour, charging_len))
                schedule_df.loc[len(schedule_df)] = [
                    requestID,
                    userID,
                    model.current_date + timedelta(hours=round_hour),
                    round_csID,
                    charging_len,
                    origin_cs,
                    origin_hour
                ]

                slots_df = model.update_user_selection(slots_df, model.current_date, round_csID, round_hour, charging_len)
            except Exception as e:
                    print(f"{userID}, {charging_len}, {origin_hour}, {origin_cs} ERROR: {e}")

        for item in user_choose_station.keys():
            print(item, "-", model.location.loc[item, "buildingID"], ":", user_choose_station[item])

        path = f"../Result/Carl/new_Baseline/{TESTING_NAME}/{TEST_DAY}/"

        if not os.path.isdir(path):
            os.makedirs(path)
        
        schedule_df.to_csv(path + f"{model.current_date.strftime('%m%d')}.csv", index=None)

        print(f"========== {day+1} done ==========")
        model.current_date+= timedelta(days=1)

    end = time.time()
    print(f"Time: {(end-start)/60} min")

