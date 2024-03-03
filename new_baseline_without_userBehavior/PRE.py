'''
PRE without userBehavior
'''
import os
import pandas as pd
import time
from datetime import timedelta
from dateutil import parser
from collections import defaultdict
from base import base


TESTING_NAME = "PRE_without_userBehavior"
TEST_DAY = "2023-06-28"

TESTING_START_DATE = parser.parse("2018-07-01")
TESTING_END_DATE = parser.parse("2018-07-08")



class UserHistoryPreference(base):
    
    
    def __init__(self):

        super().__init__()
        self.current_date = None
        self.incentive_cost = [0.1, 0.12, 0.16, 0.25, 0.4, 0.55] # 用來計算 userBehavior
        self.user_most_prefer_csID = pd.read_csv("../Dataset/charging_data_2_move.csv")
        self.run_user_most_prefer_csID()


    def run_user_most_prefer_csID(self):
        
        self.user_most_prefer_csID["createdNew"] = pd.to_datetime(self.user_most_prefer_csID["createdNew"])
        self.user_most_prefer_csID = self.user_most_prefer_csID[self.user_most_prefer_csID["createdNew"] < TESTING_START_DATE]


    def update_user_selection(self, slots_df, date, csID, hour, charging_len):

        slots_df.loc[(slots_df["buildingID"] == self.location.loc[csID, "buildingID"]) &
                     (slots_df["datetime"] >= (date + timedelta(hours = hour))) &
                     (slots_df["datetime"] < (date + timedelta(hours = hour + charging_len))), "parkingSlots"] -= 1
        
        return slots_df


    def get_user_preference_list(self, userID):

        user_history_charging_data = self.charging_data[self.charging_data['userId'] == userID]
        location_hour_visits = user_history_charging_data.groupby(['locationId', 'createdHour']).size().reset_index(name='visits').sort_values('visits', ascending=False)
        user_preference_list = location_hour_visits[['locationId', 'createdHour']].apply(tuple, axis=1).tolist()  # [(csID, hour), ...]
        
        return user_preference_list
    

    def get_user_most_prefer(self, user_preference_list, slots_df, charging_len):

        for csID, hour in user_preference_list:
            parking_slots = self.get_residual_slots(slots_df, csID, hour, charging_len)
            if parking_slots > 0:
                return csID, hour
        
            
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
    model = UserHistoryPreference()
    model.current_date = TESTING_START_DATE
    user_choose_station = defaultdict(lambda:0)

    for day in range(7):

        user_list = model.get_user_list(model.current_date)
        slots_df = model.get_parking_slots(model.building_truth_data, model.current_date)
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
                user_preference_list = model.get_user_preference_list(userID)
                
                if model.get_user_most_prefer(user_preference_list, slots_df, charging_len):
                    prefer_csID, prefer_hour = model.get_user_most_prefer(user_preference_list, slots_df, charging_len)
                    
                else:
                    prefer_csID, prefer_hour = model.get_origin_request(origin_cs, origin_hour, slots_df, charging_len)
            
                user_choose_station[prefer_csID] += 1
                print("user_choose =", (prefer_csID, prefer_hour, charging_len))
                
                schedule_df.loc[len(schedule_df)] = [
                    requestID,
                    userID,
                    model.current_date + timedelta(hours=prefer_hour),
                    prefer_csID,
                    charging_len,
                    origin_cs,
                    origin_hour,
                ]

                slots_df = model.update_user_selection(slots_df, model.current_date, prefer_csID, prefer_hour, charging_len)
            
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

