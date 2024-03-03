import pandas as pd
from collections import defaultdict
import json


class DataClassificaiton():

    def __init__(self):
        
        self.charging_record = pd.read_csv("./Dataset/charging_data_2_move.csv")
        self.location_info = pd.read_csv("./Dataset/location_5.csv")

    def userID_with_facilityType_count(self):

        userID_with_facilityType_count = defaultdict(dict)
        
        # add FacilityType column to charging_record dataframe
        for idx, row in self.charging_record.iterrows():
            condition = self.location_info["locationId"] == row["locationId"]
            self.charging_record.loc[idx , "FacilityType"] = self.location_info.loc[condition, "FacilityType"].values[0]

        facilityType_count = self.charging_record.groupby(['userId', 'FacilityType']).size()

        for (userID, facilityType), count in facilityType_count.items():
            userID_with_facilityType_count[userID][facilityType] = count

        with open('./Dataset/userID_with_facilityType_count.json', 'w') as f:
            json.dump(userID_with_facilityType_count, f)
        

    def userID_with_locationID_count(self):
        
        userID_with_locatoinID_count = defaultdict(dict)
        locationID_count = self.charging_record.groupby(['userId', 'locationId']).size()

        for (userID, locationId), count in locationID_count.items():
            userID_with_locatoinID_count[userID][locationId] = count

        with open('./Dataset/userID_with_locatoinID_count.json', 'w') as f:
            json.dump(userID_with_locatoinID_count, f)

        return
    
    def userID_with_everyHour_count(self):

        userID_with_everyHour_count = defaultdict(dict)
        everyHour_count = self.charging_record.groupby(['userId', 'createdHour']).size()

        for (userID, createdHour), count in everyHour_count.items():
            userID_with_everyHour_count[userID][createdHour] = count

        with open('./Dataset/userID_with_everyHour_count.json', 'w') as f:
            json.dump(userID_with_everyHour_count, f)


        return
    
data_classificaiton = DataClassificaiton()
data_classificaiton.userID_with_facilityType_count()
data_classificaiton.userID_with_locationID_count()
data_classificaiton.userID_with_everyHour_count()
