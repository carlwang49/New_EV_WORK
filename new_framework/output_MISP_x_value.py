import pandas as pd
from base import base
from dateutil import parser 
from datetime import timedelta
from datetime import datetime, timedelta
import json

TESTING_START_DATE = parser.parse("2018-07-01")
base = base()
null_value = 0
testing_start_time = TESTING_START_DATE
columns = [
        "origin_hour", 
        "chargingLen", 
        "userID", 
        "Type",
        "recommend_csID", 
        "recommend_datetime",
        "incentive",  
    ]
schedule_df = pd.DataFrame([], columns=columns)

for day in range(7):

    print(day)

    PRE_result = None
    REMAIN_result = None
    EL_MISP_result = None
    EL_MISP_QLearn_result = None
    FCFS_RR_v1_result = None
    FCFS_RR_v2_result = None
    MISP_result = None
    MISP_QLearn_result = None
    LLF_RR_result = None
    

    
    charging_request = base.get_charging_request(testing_start_time)

        
    PRE_result = pd.read_csv(f"../Result/MISP/0719_expected_score_6/alpha_0.5/{testing_start_time.strftime('%m%d')}.csv")

    PRE_result["datetime"] = pd.to_datetime(PRE_result["datetime"], format="%Y-%m-%d %H:%M:%S")
   
    charging_request = sorted(charging_request, key=lambda x: x[3])

    for rID, userID, charging_len, origin_hour, origin_cs in charging_request:

        try:
            schedule_A = PRE_result[PRE_result["requestID"] == rID]
            
            schedule_df.loc[len(schedule_df)] = [
                schedule_A["originHour"].values[0],
                schedule_A["chargingLen"].values[0],
                schedule_A["userID"].values[0],
                "MISP",
                str(schedule_A["locationID"].values[0]),
                schedule_A["datetime"].values[0],
                schedule_A["incentive"].values[0]
            ]

        except Exception as e:
            print(e)
            null_value += 1

    testing_start_time += timedelta(days=1)


def nested_dict(df):
    
    result = {}

    for _, row in df.iterrows():
        
        key = ((datetime.combine(row['recommend_datetime'].date(), datetime.min.time()) + timedelta(hours=row['origin_hour'])).strftime('%Y-%m-%d %H:%M:%S'), 
               row['chargingLen'], 
               row['userID'])
        
        if str(key) not in result:
            result[str(key)] = {}

        if row['Type'] not in result[str(key)]:
            result[str(key)][row['Type']] = []

        result[str(key)][row['Type']].append({
            'recommend': (row['recommend_csID'], row['recommend_datetime'].hour, row['incentive'])
        })
        
    return result


json_dict = nested_dict(schedule_df)


print(json_dict)
# with open('user_request_recommend_dic.json', 'w') as f:
#     json.dump(json_dict, f)


