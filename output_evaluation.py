from EvaluateFunction import EvaluateFunction
from datetime import timedelta
from collections import defaultdict
import pandas as pd
import json

# sigmoid_list = ["15", "2", "25"]
# epsilon_list = [1, 3, 7, 9]
# Result/Carl/MISP/MISP_Q_Learning_i025_c01_s0{15}_ep0{1}/SIGMOID_INCENTIVE_UNIT_COST_0.{15}/EPSILON_RATE_0.{1}/2023-07-11/alpha_0.2/

if __name__ == '__main__':
    
    # data_dict = {}
    
    # sigmoid_list = ["15", "2", "25"]
    # dim_emb_list = [60, 110, 160, 210]

    # for sigmoid in sigmoid_list:
    #     for dim in dim_emb_list:
    #         PATH = f"Result/Carl/MISP/MISP_QLearn_incentive_025_cost_01_s0{sigmoid}_emb_{dim}/SIGMOID_INCENTIVE_UNIT_COST_0.{sigmoid}/DIM_{dim}/2023-07-13/alpha_0.2"
    # sigmoid_list = ["15", "2", "25"]
    # epsilon_list = [1, 3, 7, 9]

    # for sigmoid in sigmoid_list:
    #     for epsilon in epsilon_list:
    #         PATH = f"Result/Carl/MISP/MISP_Q_Learning_i025_c01_s0{sigmoid}_ep0{epsilon}/SIGMOID_INCENTIVE_UNIT_COST_0.{sigmoid}/EPSILON_RATE_0.{epsilon}/2023-07-14/alpha_0.2"

    evaluate = EvaluateFunction()

    columns = [
        "datetime",
        "date",
        "userID",
        "locationID",
        "buildingID",
        "chargingLen",
        "originLocationID",
        "originChargingHour",
        "incentive", 
    ]

    schedule_df = pd.DataFrame([], columns=columns)
    user_accept_schedule_df = pd.DataFrame([], columns=columns)

    testing_start_time = evaluate.charging_start_time

    null_value = 0
    count = 0
    PATH = "Result/Carl/MF/MISMF_QLearn_incentive_025_cost_01_s02_e02/SIGMOID_INCENTIVE_UNIT_COST_0.2/DIM_10/2023-07-15/alpha_0.2"
    for day in range(7):

        charging_request = evaluate.get_charging_request(evaluate.charging_data, testing_start_time)
        recommend = pd.read_csv("./" + PATH + "/" + f"{testing_start_time.strftime('%m%d')}.csv")
        recommend["datetime"] = pd.to_datetime(recommend["datetime"], format="%Y-%m-%d %H:%M:%S")

        for request in charging_request:

            try:
                schedule = recommend[recommend["requestID"] == request[0]]
                # 創建一個新的數據框，儲存所有符合條件的行
                new_rows = pd.DataFrame({
                    'datetime': schedule['datetime'].values,
                    'testing_start_time': testing_start_time.strftime("%m%d"),
                    'userID': schedule['userID'].values,
                    'locationID': schedule['locationID'].values.astype(str),
                    'buildingID': evaluate.location.loc[schedule['locationID'].values.astype(str), 'buildingID'],
                    'chargingLen': schedule['chargingLen'].values,
                    'originLocationID': schedule['originLocationID'].values,
                    'originHour': schedule['originHour'].values,
                    'incentive': schedule['incentive'].values,
                })
                schedule_df = pd.concat([schedule_df, new_rows])  # 新增這些行到schedule_df

                user_accept_true_df = schedule[schedule['user_accept'] == True]
                user_accept_rows = pd.DataFrame({
                    'datetime': user_accept_true_df['datetime'].values,
                    'testing_start_time': testing_start_time.strftime("%m%d"),
                    'userID': user_accept_true_df['userID'].values,
                    'locationID': user_accept_true_df['locationID'].values.astype(str),
                    'buildingID': evaluate.location.loc[user_accept_true_df['locationID'].values.astype(str), 'buildingID'],
                    'chargingLen': user_accept_true_df['chargingLen'].values,
                    'originLocationID': user_accept_true_df['originLocationID'].values,
                    'originHour': user_accept_true_df['originHour'].values,
                    'incentive': user_accept_true_df['incentive'].values,
                })
                user_accept_schedule_df = pd.concat([user_accept_schedule_df, user_accept_rows]) 
                
                user_accept_list = schedule['user_accept'].values.tolist()
                num_true = user_accept_list.count(True)
                count += num_true

            except Exception as e:
                print(request, e)
                null_value += 1

        testing_start_time += timedelta(days=1)

    total_incentive = schedule_df['incentive'].sum()
    incentive_cost = round(total_incentive/20, 4)
    print("miss_value:", null_value)
    print("accept: ", count)
    print("accept rate:", round(count/schedule_df.shape[0], 4))
    print("request number: ", schedule_df.shape[0])
    print("incentive_cost: ", incentive_cost)
    print("===============================================")
    print("miss_value:", null_value)
    print("accept: ", count)
    print("accept rate:", round(count/user_accept_schedule_df.shape[0], 4))
    print("request number: ", user_accept_schedule_df.shape[0])
    
    request_number = user_accept_schedule_df.shape[0]
    basic_tariff_mean, current_tariff_mean, overload_penalty_mean, average_profit, _ = evaluate.cal_price(schedule_df=schedule_df)
    _, _, _, _, schedule_ev_charging_volume = evaluate.cal_price(schedule_df=user_accept_schedule_df)
    
    var_1, var_2 = evaluate.cal_variation(schedule_ev_charging_volume)
    average_unfavored_type, average_distance, average_time = evaluate.cal_user_preference_1(schedule_df=user_accept_schedule_df)
    average_unfavored_type_04 = evaluate.cal_user_preference_04(user_accept_schedule_df)
    average_unfavored_type_03 = evaluate.cal_user_preference_03(user_accept_schedule_df)
    favor_ratio = evaluate.cal_favor_ratio(user_accept_schedule_df)
    min_distance = 0.010980711346939084 
    max_distance = 9.949851106124495
    satisfaction_score = (favor_ratio + ((average_time - 0)/(23-0)) + ((average_distance - min_distance) / (max_distance - min_distance))) / 3
    satisfaction_score = round(satisfaction_score, 4)
    satisfaction_score_2 = (favor_ratio + (1 - ((average_time - 0)/(10-0))) + (1 - ((average_distance - min_distance) / (max_distance - min_distance)))) / 3
    print("satisfaction_score:", satisfaction_score)
    print("satisfaction_score_2:", satisfaction_score_2)

    # key = f"sigmoid_{sigmoid}_epsilon_{epsilon}"
    # key = f"sigmoid_{sigmoid}_DIM_{dim}"

    
    # # 添加新的键值对到字典
    # data_dict[key] = {
    #     "request_number": request_number,
    #     "basic_tariff_mean": basic_tariff_mean,
    #     "current_tariff_mean": current_tariff_mean,
    #     "overload_penalty_mean": overload_penalty_mean,
    #     "incentive_cost": incentive_cost,
    #     "average_profit": average_profit,
    #     "var_1": var_1, 
    #     "var_2": var_2, 
    #     "average_time": average_time, 
    #     "average_distance": average_distance, 
    #     "average_unfavored_type": average_unfavored_type,
    #     "average_unfavored_type_04": average_unfavored_type_04,
    #     "average_unfavored_type_03": average_unfavored_type_03,
    #     "favor_ratio": favor_ratio,
    #     "satisfaction_score": satisfaction_score,
    #     "satisfaction_score_2": satisfaction_score_2
    # }

    # 将字典转换为 JSON 并写入文件
    # with open('QLearn_Sigmoid_Epsilon_output_0714.json', 'w') as json_file:
        # json.dump(data_dict, json_file, indent=4)
    # with open('MISP_QLearn_dimension_output_0714.json', 'w') as json_file:
    #     json.dump(data_dict, json_file, indent=4)
