import pandas as pd
from base import base
from dateutil import parser 
from datetime import timedelta
from datetime import datetime, timedelta
import json

TESTING_START_DATE = parser.parse("2018-07-01")
base = base()
null_value = 0

for day in range(7):

    testing_start_time = TESTING_START_DATE

    PRE_result = None
    REMAIN_result = None
    EL_MISP_result = None
    EL_MISP_QLearn_result = None
    FCFS_RR_v1_result = None
    FCFS_RR_v2_result = None
    MISP_result = None
    MISP_QLearn_result = None
    LLF_RR_result = None
    
    columns = [
        "origin_hour", 
        "chargingLen", 
        "userID", 
        "Type",
        "recommend_csID", 
        "recommend_datetime",
        "incentive",  
        "corresponding_x",
    ]

    schedule_df = pd.DataFrame([], columns=columns)
    charging_request = base.get_charging_request(testing_start_time)

        
    PRE_result = pd.read_csv(f"../Result/Baseline/PRE/{testing_start_time.strftime('%m%d')}.csv")
    REMAIN_result = pd.read_csv(f"../Result/Baseline/Remaining/baseline_remaining/{testing_start_time.strftime('%m%d')}.csv")
    EL_MISP_result = pd.read_csv(f"../Result/MISP/EL_MISP/alpha_0.5/{testing_start_time.strftime('%m%d')}.csv")
    EL_MISP_QLearn_result = pd.read_csv(f"../Result/MISP_Q_Learning/EL_MISP_QLearn/alpha_0.5/{testing_start_time.strftime('%m%d')}.csv")
    FCFS_RR_v1_result = pd.read_csv(f"../Result/Baseline/FCFS_RR_v1/{testing_start_time.strftime('%m%d')}.csv")
    FCFS_RR_v2_result = pd.read_csv(f"../Result/Baseline/FCFS_RR_V2/{testing_start_time.strftime('%m%d')}.csv")
    MISP_result = pd.read_csv(f"../Result/MISP/MISP_with_userBehavior/alpha_0.5/{testing_start_time.strftime('%m%d')}.csv")
    MISP_QLearn_result = pd.read_csv(f"../Result/MISP_Q_Learning/MISP_Q_Learning/alpha_0.5/{testing_start_time.strftime('%m%d')}.csv")
    LLF_RR_result = pd.read_csv(f"../Result/Baseline/LLF_RR_V2/{testing_start_time.strftime('%m%d')}.csv")

    PRE_result["datetime"] = pd.to_datetime(PRE_result["datetime"], format="%Y-%m-%d %H:%M:%S")
    REMAIN_result["datetime"] = pd.to_datetime(REMAIN_result["datetime"], format="%Y-%m-%d %H:%M:%S")
    EL_MISP_result["datetime"] = pd.to_datetime(EL_MISP_result["datetime"], format="%Y-%m-%d %H:%M:%S")
    EL_MISP_QLearn_result["datetime"] = pd.to_datetime(EL_MISP_QLearn_result["datetime"], format="%Y-%m-%d %H:%M:%S")
    FCFS_RR_v1_result["datetime"] = pd.to_datetime(FCFS_RR_v1_result["datetime"], format="%Y-%m-%d %H:%M:%S")
    FCFS_RR_v2_result["datetime"] = pd.to_datetime(FCFS_RR_v2_result["datetime"], format="%Y-%m-%d %H:%M:%S")
    MISP_result["datetime"] = pd.to_datetime(MISP_result["datetime"], format="%Y-%m-%d %H:%M:%S")
    MISP_QLearn_result["datetime"] = pd.to_datetime(MISP_QLearn_result["datetime"], format="%Y-%m-%d %H:%M:%S")
    LLF_RR_result["datetime"] = pd.to_datetime(LLF_RR_result["datetime"], format="%Y-%m-%d %H:%M:%S")

    charging_request = sorted(charging_request, key=lambda x: x[3])

    for rID, userID, charging_len, origin_hour, origin_cs in charging_request:

        # try:
        schedule_A = PRE_result[PRE_result["requestID"] == rID]
        schedule_B = REMAIN_result[REMAIN_result["requestID"] == rID]
        schedule_C = EL_MISP_result[EL_MISP_result["requestID"] == rID]
        schedule_D = EL_MISP_QLearn_result[EL_MISP_QLearn_result["requestID"] == rID]
        schedule_E = FCFS_RR_v1_result[FCFS_RR_v1_result["requestID"] == rID]
        schedule_F = FCFS_RR_v2_result[FCFS_RR_v2_result["requestID"] == rID]
        schedule_G = MISP_result[MISP_result["requestID"] == rID]
        schedule_H = MISP_QLearn_result[MISP_QLearn_result["requestID"] == rID]
        schedule_I = LLF_RR_result[LLF_RR_result["requestID"] == rID]
        
        schedule_df.loc[len(schedule_df)] = [
            schedule_A["originHour"].values[0],
            schedule_A["chargingLen"].values[0],
            schedule_A["userID"].values[0],
            "PRE",
            str(schedule_A["locationID"].values[0]),
            schedule_A["datetime"].values[0],
            0,
            schedule_A["x_value"].values[0],
        ]


        schedule_df.loc[len(schedule_df)] = [
            schedule_B["originHour"].values[0],
            schedule_B["chargingLen"].values[0],
            schedule_B["userID"].values[0],
            "REMAIN",
            str(schedule_A["locationID"].values[0]),
            schedule_B["datetime"].values[0],
            0,
            schedule_B["x_value"].values[0],
        ]

        schedule_df.loc[len(schedule_df)] = [
            schedule_C["originHour"].values[0],
            schedule_C["chargingLen"].values[0],
            schedule_C["userID"].values[0],
            "EL_MISP",
            str(schedule_C["locationID"].values[0]),
            schedule_C["datetime"].values[0],
            schedule_C["incentive"].values[0],
            schedule_C["x_value"].values[0],
        ]

        schedule_df.loc[len(schedule_df)] = [
            schedule_D["originHour"].values[0],
            schedule_D["chargingLen"].values[0],
            schedule_D["userID"].values[0],
            "EL_MISP_QLearn",
            str(schedule_D["locationID"].values[0]),
            schedule_D["datetime"].values[0],
            schedule_D["incentive"].values[0],
            schedule_D["x_value"].values[0],
        ]

        schedule_df.loc[len(schedule_df)] = [
            schedule_E["originHour"].values[0],
            schedule_E["chargingLen"].values[0],
            schedule_E["userID"].values[0],
            "FCFS_RR_v1",
            str(schedule_E["locationID"].values[0]),
            schedule_E["datetime"].values[0],
            0,
            schedule_E["x_value"].values[0],
        ]

        schedule_df.loc[len(schedule_df)] = [
            schedule_F["originHour"].values[0],
            schedule_F["chargingLen"].values[0],
            schedule_F["userID"].values[0],
            "FCFS_RR_v2",
            str(schedule_F["locationID"].values[0]),
            schedule_F["datetime"].values[0],
            0,
            schedule_F["x_value"].values[0],
        ]

        schedule_df.loc[len(schedule_df)] = [
            schedule_G["originHour"].values[0],
            schedule_G["chargingLen"].values[0],
            schedule_G["userID"].values[0],
            "MISP",
            str(schedule_G["locationID"].values[0]),
            schedule_G["datetime"].values[0],
            schedule_G["incentive"].values[0],
            schedule_G["x_value"].values[0],
        ]

        schedule_df.loc[len(schedule_df)] = [
            schedule_H["originHour"].values[0],
            schedule_H["chargingLen"].values[0],
            schedule_H["userID"].values[0],
            "MISP_QLearn",
            str(schedule_H["locationID"].values[0]),
            schedule_H["datetime"].values[0],
            schedule_H["incentive"].values[0],
            schedule_H["x_value"].values[0],
        ]

        schedule_df.loc[len(schedule_df)] = [
            schedule_I["originHour"].values[0],
            schedule_I["chargingLen"].values[0],
            schedule_I["userID"].values[0],
            "LLF_RR",
            str(schedule_I["locationID"].values[0]),
            schedule_I["datetime"].values[0],
            0,
            schedule_I["x_value"].values[0],
        ]


        # except Exception as e:
        #     print(e)
        #     null_value += 1

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
            'recommend': (row['recommend_csID'], row['recommend_datetime'].hour, row['incentive']), 
            'corresponding_x': row['corresponding_x']
        })
        
    return result


json_dict = nested_dict(schedule_df)
print(json_dict)
with open('user_request_x_value_dic.json', 'w') as f:
    json.dump(json_dict, f)


