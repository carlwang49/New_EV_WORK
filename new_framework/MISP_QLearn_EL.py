import os
import pandas as pd
import time
from datetime import timedelta
from dateutil import parser
from collections import defaultdict
from MISP_QLearning import QLearn
from EpsilonGreedy import EpsilonGreedy
import json


### global variable ###
ALPHA = 0.5
TESTING_NAME = "MISP_QLearn_EL_output"
TEST_DAY = "2023-06-27"

INCENTIVE_BUDGET = 400

TESTING_START_DATE = parser.parse("2018-07-01")
TESTING_END_DATE = parser.parse("2018-07-08")

EPSILON_RATE = 0.5
MU = 0
SIGMA_SQUARE = 1


class EL_MISP_QLearn(QLearn):

    def __init__(self):
        super().__init__()
        self.charging_csID_avg_request_nums = None
        self.set_charging_csID_avg_request_nums()
       

    def set_charging_csID_avg_request_nums(self):
        
        charging_one_month_ago_data = self.charging_data[(self.charging_data["createdNew"] < parser.parse("2018-07-01")) 
                                                            & (self.charging_data["createdNew"] >= parser.parse("2018-06-01"))].copy()
        self.charging_csID_avg_request_nums = charging_one_month_ago_data.groupby("locationId").size().reset_index(name='request_count')
        self.charging_csID_avg_request_nums['request_count'] = (self.charging_csID_avg_request_nums['request_count'] / 24).astype(int)
        self.charging_csID_avg_request_nums['request_count'] = self.charging_csID_avg_request_nums['request_count'].apply(lambda x: max(x, 1))


    def get_all_combinations(self, user_list, slots_df, userID, charging_len):
        '''
        取得所有可能的充電組合
        user_list: 所有使用者 ID
        slots_df: 充電位資訊
        userID: 要排程的那個使用者 ID
        charging_len: 使用者充電時間長度
        '''

        combinations = list()
        for csID in self.location.index:
            for hour in range(24):
                
                ### check residual parking slots ###
                parking_slots = self.get_residual_slots(slots_df, csID, hour, charging_len)
                if parking_slots <= 0:
                    continue

                # coupon = 0 代表不給獎勵
                N_i = self.charging_csID_avg_request_nums.loc[
                    self.charging_csID_avg_request_nums['locationId'] == csID, 'request_count'].values[0]
                csID_curr_budget = self.budget[csID]
                coupon = min(max(csID_curr_budget // N_i, 1), 10)
                if coupon > csID_curr_budget:
                    continue

                score, willingness, personal = self.expected_score(user_list, userID, csID, hour, coupon)
                schedule_cost = self.incentive_cost[coupon]
                combinations.append((csID, hour, score, schedule_cost, personal, willingness))
            

        return combinations 


if __name__ == "__main__":

    for random_counter in range(random_start_counter, random_end_counter+1):
        
        print(f"counter = {random_counter}")
        start = time.time()
        agent = EpsilonGreedy(epsilon=EPSILON_RATE)
        model = EL_MISP_QLearn()
        
        agent.initialize()
        model.current_date = TESTING_START_DATE # 測試開始時間，可以自行調整

        user_choose_station = defaultdict(lambda: 0)


        for day in range(7):

            ### User interaction matrix ### 
            user_list = model.get_user_list(model.current_date)
            model.get_user_interaction_value(user_list)

            ### Spendable parking slots prediction ###
            slots_df = model.get_parking_slots(model.building_predict_data, model.current_date)
            slots_df = model.reserve_parking_slots(slots_df, model.current_date)

            ### Utilization ###
            model.calculate_utilization(slots_df=slots_df, first=True)
            model.average_utilization()

            ### Get building budget and threshold ###
            model.set_budgets()

            ### EV charging request ###
            charging_request = model.get_charging_request(model.current_date)

            ### schedule ###
            progress = 1
            
            columns = [
                "requestID", 
                "userID", 
                "datetime", 
                "locationID",
                "chargingLen", 
                "score", 
                "incentive", 
                "originLocationID", 
                "originHour",
                "threshold",
                "personal",
                "willingness",
            ]

            schedule_df = pd.DataFrame([], columns=columns)
            
            # 接單
            for rID, userID, charging_len, origin_hour, origin_cs in charging_request:
                
                try:
                    q_learn_recommend, q_learn_index, score_max_recommend, score_max_index =\
                                model.OGAP_QLearn(user_list, slots_df, userID, int(charging_len), agent.q_table)
                    
                    if q_learn_index == 0:
                        user_choose = q_learn_recommend[-1]

                    else:

                        select_arm_result, reward = agent.select_arm(q_learn_recommend, score_max_recommend)
                        user_choose = select_arm_result

                        # update Q table     
                        agent.update(q_learn_index, reward)
                        if reward == 1:
                            agent.updateEpsilon()

                        incentive_nums = model.incentive_cost.index(user_choose[3]) if reward else 0
                        user_choose_station[user_choose[0]] += 1
                        print("q_value: ", agent.q_table[q_learn_index])
                    
                    print("user_choose=", user_choose)
                    
                    schedule_df.loc[len(schedule_df)] = [
                        rID,
                        userID,
                        model.current_date + timedelta(hours=user_choose[1]),
                        user_choose[0],
                        charging_len,
                        user_choose[2],
                        incentive_nums,
                        origin_cs,
                        origin_hour,
                        user_choose[6],
                        user_choose[4],
                        user_choose[5]
                    ]

                    slots_df = model.update_user_selection(slots_df, model.current_date, user_choose, charging_len)
                    model.calculate_utilization(schedule=user_choose, charging_len=charging_len)
                    model.average_utilization()
                    print(f"progress: {progress}/{len(charging_request)}")

                except Exception as e:
                    print(f"{userID}, {charging_len}, {origin_hour}, {origin_cs} ERROR: {e}")

                progress += 1

            for item in user_choose_station.keys():
                print(item, "-", model.location.loc[item, "buildingID"], ":", user_choose_station[item])


            path = f"../Result/Carl/MISP/{TESTING_NAME}/{TEST_DAY}/alpha_{ALPHA}/{random_counter}/"

            if not os.path.isdir(path):
                os.makedirs(path)

            schedule_df.to_csv(path + f"{model.current_date.strftime('%m%d')}.csv", index=None)

            print(f"{day+1} default count: {model.default_count}")
            print(f"========== {day+1} done ==========")
            model.current_date += timedelta(days=1)

            # update average request number
            model.update_average_request_num(model.current_date, user_choose_station)

            file_path = f'../Result/Carl/MISP/{TESTING_NAME}/{TEST_DAY}/alpha_{ALPHA}/{random_counter}/q_table.json'

            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(f'../Result/Carl/MISP/{TESTING_NAME}/{TEST_DAY}/alpha_{ALPHA}/{random_counter}/q_table.json', 'w') as json_file:
                    json.dump(agent.q_table, json_file)

    end = time.time()
    print(f"Time: {(end-start)/60} min")
