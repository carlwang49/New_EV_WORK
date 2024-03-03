import os
import time
import pandas as pd
from datetime import timedelta
from dateutil import parser
from collections import defaultdict
from UserBehavior import UserBehavior
from base import base


TESTING_NAME = "PRE_userBehavior_sigmoid_02"
TEST_DAY = "2023-07-03"
SIGMOID_INCENTIVE_UNIT = 0.2

TESTING_START_DATE = parser.parse("2018-07-01")
TESTING_END_DATE = parser.parse("2018-07-08")

INCENTIVE_NUMS = 0


class PRE(base):
    
    def __init__(self):

        super().__init__()
        self.current_date = None


    def update_user_selection(self, slots_df, date, csID, hour, charging_len):

        slots_df.loc[(slots_df["buildingID"] == self.location.loc[csID, "buildingID"]) &
                     (slots_df["datetime"] >= (date + timedelta(hours = hour))) &
                     (slots_df["datetime"] < (date + timedelta(hours = hour + charging_len))), "parkingSlots"] -= 1
        
        return slots_df


    def get_user_preference_list(self, userID):
        '''
        取得 UserID 歷史最常去的時間和地點排序
        '''

        time_filter = (self.charging_data['createdNew'] >= '2018-01-01') & (self.charging_data['createdNew'] <= '2018-06-30')
        history_charging_data = self.charging_data.loc[time_filter].copy()
        user_history_charging_data = history_charging_data[history_charging_data['userId'] == userID]
        location_hour_visits = user_history_charging_data.groupby(['locationId', 'createdHour']).size().reset_index(name='visits').sort_values('visits', ascending=False)
        print(userID)
        print(location_hour_visits)
        time.sleep(2)
        user_preference_list = location_hour_visits[['locationId', 'createdHour']].apply(tuple, axis=1).tolist()
        
        return user_preference_list
    

    def get_user_most_prefer(self, user_preference_list, slots_df, charging_len):
        
        for csID, hour in user_preference_list:
            parking_slots = self.get_residual_slots(slots_df, csID, hour, charging_len)
            if parking_slots > 0:
                return csID, hour
        
        return None
        
        
            
    def get_origin_request(self, csID, hour, slots_df, charging_len):

        for _ in range(24):
            hour = hour % 24
            parking_slots = self.get_residual_slots(slots_df, csID, hour, charging_len)
            if parking_slots <= 0:
                hour += 1
            else:
                return csID, hour
            

if __name__ == "__main__":
    
    start = time.time()
    model = PRE()
    user_behavior = UserBehavior()
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
   
            # try:
            user_preference_list = model.get_user_preference_list(userID)
            user_preference = model.get_user_most_prefer(user_preference_list, slots_df, charging_len) 
            
            if user_preference:
                recommend_csID, recommend_hour = user_preference 
                factor_time = user_behavior.factor_time(recommend_hour, userID)
                factor_cate = user_behavior.factor_cate(model.location, recommend_csID, userID)
                factor_dist = user_behavior.factor_dist(model.location, model.charging_data, recommend_csID, userID)
                dissimilarity = user_behavior.get_dissimilarity(factor_time, factor_cate, factor_dist)
                prob = user_behavior.estimate_willingeness(dissimilarity, INCENTIVE_NUMS, SIGMOID_INCENTIVE_UNIT)
                user_accept = user_behavior.get_user_decision(prob)
                print("probability: ", prob)
                print("user_accept: ", user_accept) 
                recommend_csID, recommend_hour = user_preference if user_accept else model.get_origin_request(origin_cs, origin_hour, slots_df, charging_len)
            else:
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
            ]
            slots_df = model.update_user_selection(slots_df, model.current_date, recommend_csID, recommend_hour, charging_len)
            
            # except Exception as e:
            #     print(f"{userID}, {charging_len}, {origin_hour}, {origin_cs} ERROR: {e}")

        for item in user_choose_station.keys():
            print(item, "-", model.location.loc[item, "buildingID"], ":", user_choose_station[item])
        
        path = f"../Result/Carl/new_Baseline/{TESTING_NAME}/{TEST_DAY}/"
        
        if not os.path.isdir(path):
            os.makedirs(path)
        
        schedule_df.to_csv(path + f"{model.current_date.strftime('%m%d')}.csv", index=None)

        print(f"========== {day+1} done ==========")
        model.current_date += timedelta(days=1)

    end = time.time()
    print(f"Time: {(end-start)/60} min")

