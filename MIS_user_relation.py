from email.policy import default
import pandas as pd
import numpy as np
from collections import defaultdict
import os

METHOD = "MIS"
path = f"./Result/Baseline/MIS/0721_new_relation/Relation/"      
if not os.path.isdir(path):
    os.makedirs(path)


user_df = pd.read_csv("./Dataset/user_list_2.csv", index_col=0)
location_df = pd.read_csv("./Dataset/location_5.csv", index_col="locationId")["buildingID"]
location_df.index = location_df.index.astype(str)
charging_data_df = pd.read_csv("./Dataset/charging_data_2_move.csv")

# 將 'createdNew' 轉換為日期格式
charging_data_df['createdNew'] = pd.to_datetime(charging_data_df['createdNew'])
charging_data_df["userId"] = charging_data_df["userId"].astype(str)
charging_data_df["locationId"] = charging_data_df["locationId"].astype(str)
charging_history_data_df = charging_data_df[charging_data_df['createdNew'] < pd.Timestamp(2023,6,30)]

for id, row in user_df.iterrows():
    df = pd.read_csv(f"./Dataset/Relation_training_data/{row['name']}.csv", index_col="createdHour")
    for hour in range(24):
        for locationID in df.columns:
            
            user_data = charging_history_data_df.loc[charging_history_data_df['userId'] == row['name']]
            user_filter = ((user_data['locationId'] == locationID) & (user_data['createdHour'] == hour))
            user_cs_hr_data = user_data.loc[user_filter]


            # location_counts = user_data['locationId'].value_counts()
            # location_prob = 0 if locationID not in list(location_counts.keys()) \
            #     else location_counts[locationID]
            
            # created_hour_counts = user_data['createdHour'].value_counts()
            # created_hour_prob = 0 if hour not in list(created_hour_counts.keys()) \
            #     else (created_hour_counts[hour] / created_hour_counts.sum())
              
            buildingID = location_df[locationID]
            # df.loc[hour, locationID] = (created_hour_prob + location_prob) / 2
            df.loc[hour, locationID] = len(user_cs_hr_data)
    
    df.to_csv(f"./Result/Baseline/MIS/0721_new_relation/Relation_2/{row['name']}.csv")
    print(f"{id} done")
