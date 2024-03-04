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

    MISP_result = None
    MISP_Q_Learning_result = None
    EL_MISP_QLearning_result = None
    EL_MISP_result = None

    
    columns = [
        "origin_hour", 
        "chargingLen", 
        "userID", 
        "Type",
        "recommend_datetime",
        "incentive", 
        "score",
        "threshold",  
        "personal",
        "willingness"
    ]

    schedule_df = pd.DataFrame([], columns=columns)
    charging_request = base.get_charging_request(testing_start_time)

        
    MISP_result = pd.read_csv(f"../Result/MISP/without_user_behavior/MISP_output/alpha_0.5/{testing_start_time.strftime('%m%d')}.csv")
    MISP_Q_Learning_result = pd.read_csv(f"../Result/MISP/without_user_behavior/MISP_Q_Learning_output/alpha_0.5/{testing_start_time.strftime('%m%d')}.csv")
    EL_MISP_QLearning_result = pd.read_csv(f"../Result/MISP/    /EL_MISP_QLearning/alpha_0.5/{testing_start_time.strftime('%m%d')}.csv")
    EL_MISP_result = pd.read_csv(f"../Result/MISP/without_user_behavior/EL_MISP_output/alpha_0.5/{testing_start_time.strftime('%m%d')}.csv")
    

    MISP_result["datetime"] = pd.to_datetime(MISP_result["datetime"], format="%Y-%m-%d %H:%M:%S")
    MISP_Q_Learning_result["datetime"] = pd.to_datetime(MISP_Q_Learning_result["datetime"], format="%Y-%m-%d %H:%M:%S")
    EL_MISP_QLearning_result["datetime"] = pd.to_datetime(EL_MISP_QLearning_result["datetime"], format="%Y-%m-%d %H:%M:%S")
    EL_MISP_result["datetime"] = pd.to_datetime(EL_MISP_result["datetime"], format="%Y-%m-%d %H:%M:%S")

    charging_request = sorted(charging_request, key=lambda x: x[3])

    for rID, userID, charging_len, origin_hour, origin_cs in charging_request:

        # try:
        schedule_A = MISP_result[MISP_result["requestID"] == rID]
        schedule_B = MISP_Q_Learning_result[MISP_Q_Learning_result["requestID"] == rID]
        schedule_C = EL_MISP_QLearning_result[EL_MISP_QLearning_result["requestID"] == rID]
        schedule_D = EL_MISP_result[EL_MISP_result["requestID"] == rID]

        schedule_df.loc[len(schedule_df)] = [
            schedule_A["originHour"].values[0],
            schedule_A["chargingLen"].values[0],
            schedule_A["userID"].values[0],
            "MISP",
            schedule_A["datetime"].values[0],
            schedule_A["incentive"].values[0],
            schedule_A["score"].values[0],
            schedule_A["threshold"].values[0],
            schedule_A["personal"].values[0],
            schedule_A["willingness"].values[0]
        ]

        schedule_df.loc[len(schedule_df)] = [
            schedule_B["originHour"].values[0],
            schedule_B["chargingLen"].values[0],
            schedule_B["userID"].values[0],
            "MISP_Q_Learning",
            schedule_B["datetime"].values[0],
            schedule_B["incentive"].values[0],
            schedule_B["score"].values[0],
            schedule_B["threshold"].values[0],
            schedule_B["personal"].values[0],
            schedule_B["willingness"].values[0]
        ]

        schedule_df.loc[len(schedule_df)] = [
            schedule_C["originHour"].values[0],
            schedule_C["chargingLen"].values[0],
            schedule_C["userID"].values[0],
            "L_MISP_QLearning",
            schedule_C["datetime"].values[0],
            schedule_C["incentive"].values[0],
            schedule_C["score"].values[0],
            schedule_C["threshold"].values[0],
            schedule_C["personal"].values[0],
            schedule_C["willingness"].values[0]
        ]

        schedule_df.loc[len(schedule_df)] = [
            schedule_D["originHour"].values[0],
            schedule_D["chargingLen"].values[0],
            schedule_D["userID"].values[0],
            "EL_MISP",
            schedule_D["datetime"].values[0],
            schedule_D["incentive"].values[0],
            schedule_D["score"].values[0],
            schedule_D["threshold"].values[0],
            schedule_D["personal"].values[0],
            schedule_D["willingness"].values[0]
        ]

        # except Exception as e:
        #     print(e)
        #     null_value += 1

        testing_start_time += timedelta(days=1)

print(schedule_df)

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
            'incentive': row['incentive'], 
            'score': row['score'],
            'threshold': row['threshold'],
            'personal': row['personal'],
            'willingness': row['willingness'],
        })
        
    return result


json_dict = nested_dict(schedule_df)
print(json_dict)
with open('user_request_personal_willingness_dic.json', 'w') as f:
    json.dump(json_dict, f)


