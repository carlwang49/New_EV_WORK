import os
import json
import time
import pandas as pd
import numpy as np
from datetime import timedelta
from dateutil import parser
from collections import defaultdict
from EpsilonGreedy import EpsilonGreedy
from MISP_incentive_025_cost_01_sigmoid import MISP
from UserBehavior import UserBehavior
import random

### global variable ###
ALPHA = 0.2
TESTING_NAME = "MISMF_Qlearn_e05_random"
TEST_DAY = "2023-07-15"
 
INCENTIVE_UNIT = 0.25
COST_UNIT = 0.1

EMB_DIM = 10
SIGMOID_INCENTIVE_UNIT_COST = 0.2

EPSILON_RATE = 0.5
TESTING_START_DATE = parser.parse("2018-07-01")
TESTING_END_DATE = parser.parse("2018-07-08")




class QLearn(MISP):

    def __init__(self):
        
        super().__init__()
        self.user_facility_perc_dic = pd.read_json("../Dataset/user_facility_perc_dic.json").to_dict()
    
    

    def expected_score(self, user_list, userID, csID, hour, incentive_num):
        '''
        計算每個充電選項對系統的期望分數
        user_list: 所有使用者 ID
        userID: 要排程的那個使用者 ID
        csID: 選項中的充電站 ID
        hour: 選項中的充電開始時間
        incentive: 選項中獎勵的數量
        ALPHA = __
        '''
        personal_willingness, personal_origin = self.personal_willingness(userID, csID, hour, incentive_num)
        trend = self.trend_willingness(user_list, userID, csID, hour)
        
        ### (該時刻的總平均使用率 - (cs, t)使用率) / (cs, t)最大使用率 ###
        load_value = self.average_utilization_ratio[hour] - self.load_matrix[int(self.location.loc[csID, "buildingID"])][hour]
        load_value = load_value / self.average_utilization_ratio.max() if self.average_utilization_ratio.max() != 0 else 1

        score = (((ALPHA * personal_willingness) + ((1 - ALPHA) * trend)) * load_value)
        cp_value = score/self.incentive_cost[incentive_num]

        return score, personal_willingness, personal_origin, trend, cp_value
    

    def get_user_most_preference_hour(self, userID):
        
        time_filter = (self.charging_data['createdNew'] >= '2018-01-01') & (self.charging_data['createdNew'] <= '2018-06-30')
        history_charging_data = self.charging_data.loc[time_filter].copy()
        user_history_charging_data = history_charging_data[history_charging_data['userId'] == userID]
        user_most_preference_hour = user_history_charging_data['createdHour'].mode()[0]

        return user_most_preference_hour
    

    def find_min_diff(self, dp_time, history_preference_time):
        '''
        計算”option出來得時間點“與“過去使用者偏好的時間點”
        '''
        diff = min(abs(dp_time - history_preference_time),
                   abs(history_preference_time - dp_time + 24))

        return diff


    def convert_percentage(self, num):
        '''
        將 facility type 對應的偏好度(%) 轉換成 Q_table 對應的比例(%)
        '''
        if num <= 0.2:
            return "20%"
        elif num <= 0.4:
            return "40%"
        elif num <= 0.6:
            return "60%"
        elif num <= 0.8:
            return "80%"
        else:
            return "100%"


    def get_q_table_index_option_list(self, userID, combinations):
        '''
        option 轉換成 Q_table 對應的 index
        '''
        q_table_index_option_list = []
    
        for option in combinations:
            
            csID, hour = option[0], option[1]  # csID , hour
            facility_type = self.location.loc[csID, "FacilityType"]
            percentage = self.user_facility_perc_dic.get(str(userID), {}).get(facility_type, 0)
            user_prefer_hours = self.get_user_most_preference_hour(userID)
            perc_interval = self.convert_percentage(percentage) if percentage else "0%"
            time_diff = self.find_min_diff(int(hour), int(user_prefer_hours))
            incentive_num = self.incentive_cost.index(option[3])
            q_table_index_option_list.append({(perc_interval, time_diff, incentive_num): option})
        
            # (計算 user 對於 csID 的 facilityType 過去去過的比例, 推薦時間段對於 user喜好時間得差距, 給予的incetive張數, option score)
            # tup[:3] 前三個 elements 去對應 q-table index

        return q_table_index_option_list
    

    def filter_q_value(self, q_table_list):
        '''
        若 Q-value 一樣，找 score 最大的
        q_table_list: (index:option)
        '''
        index_option_dict = defaultdict()
        for element in q_table_list:
            for key, value in element.items():
                if key not in index_option_dict:
                    index_option_dict[str(key)] = value
                else:
                    # 比較 dict 裡面的對應的 value
                    # value[2]: score
                    if value[2] > index_option_dict[key][2]:
                        index_option_dict[str(key)] = value

        return index_option_dict


    # def get_user_interaction_value(self, user_list):
    #     '''
    #     讀取 relational learning 預測的偏好度
    #     user_list: 所有使用者 ID
    #     '''
    #     # for user_id in user_list:
    #     #     history = pd.read_csv(f"../Result/MISP/Relation/{user_id}.csv", index_col="createdHour")
    #     #     self.user_preference[user_id] = self.normalize_interaction(history)
        
    #     for user_id in user_list:
    #         history = pd.read_csv(f"../Result/MISP/Relation_fix/DIM_{EMB_DIM}/{user_id}.csv", index_col="createdHour")
    #         self.user_preference[user_id] = self.normalize_interaction(history)
    
    

    def OGAP_QLearn(self, user_list, slots_df, userID, charging_len, q_table):
        '''
        user_list
        slots_df
        userID
        charging_len
        '''
        combinations = self.get_all_combinations(user_list, slots_df, userID, charging_len)
        if len(combinations) == 0:
            self.default_count += 1
            return self.default_recommend(self.user_preference[userID], userID, charging_len)
        self.threshold = self.set_threshold(combinations, self.current_date) 
        combinations = self.initial_filter(combinations)
        
        ### 全部的組合都被篩選掉時給 default 
        if len(combinations) == 0:
            self.default_count += 1
            return self.default_recommend(self.user_preference[userID], userID, charging_len)

        # score_max_recommend = max(combinations, key=lambda x: x[2])
        score_max_recommend = random.choice(combinations)
        q_learning_recommend, q_index = self.get_q_learning_recommend(userID, combinations, q_table)

        return q_learning_recommend, q_index, score_max_recommend



    def get_q_learning_recommend(self, userID, combinations, q_table):
        
        q_table_index_option_list = self.get_q_table_index_option_list(userID, combinations)
        index_option_dict = self.filter_q_value(q_table_index_option_list)
        index_option_list = list(index_option_dict.items())
        sort_score_list = sorted(index_option_list, key=lambda x: x[1][2], reverse=True) # score 由大排到小

        q_dict = defaultdict() # index 對應的 q_value

        # 建立 index_option_dict 的 Q-table 表
        for idx in index_option_dict.keys():
            q_value = q_table[str(idx)]
            q_dict[idx] = int(q_value)
        
        max_q_value = max(q_dict.values())  # 最大的 Q value
        q_index = str(sort_score_list[0][0])
        q_learning_recommend = sort_score_list[0][1]
        
        for index, value in dict(sort_score_list).items():
            if q_table[str(index)] == max_q_value:
                q_learning_recommend = value
                q_index = str(index)
                break
        
        return q_learning_recommend, q_index


    def default_recommend(self, preference_df, userID, charging_len):
        '''
        預設推薦
        preference_df: 預測的使用者偏好 (userID 對每一個充電站每個時段的偏好)
        userID: 要排程的那個使用者 ID
        charging_len: 使用者充電時間長度
        '''
        # recommend_cs = str(preference_df.idxmax().idxmax())
        # recommend_hour = int(preference_df.idxmax()[recommend_cs])

        max_index = preference_df.stack().idxmax()
        print("+++++++++")
        print("max_index: ", max_index)
        print("+++++++++")
        recommend_hour, recommend_cs = max_index
        recommend_hour = int(recommend_hour)
        recommend_cs = str(recommend_cs)

        preference_np = preference_df.to_numpy() 
        check = 0
        while check < (20*24):

            hour, locationIdx = np.unravel_index(preference_np.argmax(), preference_np.shape) # perference 最高的 hour 和 locationId
            hour, locationIdx = int(hour), locationIdx

            locationID = preference_df.columns[locationIdx]
            location_budget = self.budget[locationID]
            charging_len = int(charging_len)

            # 1. 確認 budget 至少一張
            # 2. 確認是否和過去 user 喜歡的 facilitType 相同 
            # 3. 建議的時間跟過去 user 喜歡的時間相差不超過兩小時 
            # 4. 確定是否用空位
            print("========================")
            print(location_budget)
            print((all(self.user_history_preference[userID]["facilityType"] == self.location.loc[locationID, "FacilityType"])))
            print(self.check_createdHour(userID, hour))
            print(self.get_residual_slots(slots_df, locationID, hour, charging_len) > 0)
            print("========================")
            if ((location_budget > 0) and 
                (all(self.user_history_preference[userID]["facilityType"] == self.location.loc[locationID, "FacilityType"])) and
                (self.check_createdHour(userID, hour)) and
                (self.get_residual_slots(slots_df, locationID, hour, charging_len) > 0)): 
                
                recommend_cs = locationID
                recommend_hour = hour
                print(f"default recommend = ({recommend_cs}, {recommend_hour})")
                break
            
            check += 1
            preference_np[hour][locationIdx] = -10

        personal_origin, personal_willingness, trend, cp_value, threshold = 0, 0, 0, 0, 0
        print(recommend_cs, locationID)
        return (recommend_cs, 
                recommend_hour, 
                -1, 
                round(1 * COST_UNIT + 0.1, 1), 
                personal_origin, 
                personal_willingness, 
                trend, 
                cp_value, 
                threshold
                ), 0 , 0
    

    def get_user_origin_choose(self, slots_df, origin_cs, origin_hour, charging_len):
        
        hour = origin_hour
        for _ in range(24):
            hour %= 24
            parking_slots = self.get_residual_slots(slots_df, origin_cs, hour, charging_len)
            if parking_slots > 0:
                return (origin_cs, hour, -1, 0.1, 0, 0, 0, 0, 0)
            hour += 1



if __name__ == "__main__":

    # random_start_counter = 1
    # random_end_counter = 5
    # for random_counter in range(random_start_counter, random_end_counter+1):
        
    # print(f"counter = {random_counter}")
    print(f"ALPHA = {ALPHA}")
    print(f"EMB_DIM = {EMB_DIM}")
    print(f"SIGMOID_INCENTIVE_UNIT_COST = {SIGMOID_INCENTIVE_UNIT_COST}")
    print(f"EPSILON_RATE = {EPSILON_RATE}")

    start = time.time()
    agent = EpsilonGreedy(epsilon=EPSILON_RATE)
    agent.initialize()
    model = QLearn()
    user_behavior = UserBehavior()
    model.current_date = TESTING_START_DATE
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
            "personal",
            "willingness",
            "trend",
            "cp_value",
            "threshold",
            "user_accept"
        ]

        schedule_df = pd.DataFrame([], columns=columns)
        for requestID, userID, charging_len, origin_hour, origin_cs in charging_request:
            
            # try:
            q_learn_recommend, q_learn_index, score_max_recommend =\
                    model.OGAP_QLearn(user_list, slots_df, userID, int(charging_len), agent.q_table)
            
            if q_learn_index == 0:
                recommend = q_learn_recommend
            
            else:
                select_arm_result, reward = agent.select_arm(q_learn_recommend, score_max_recommend)
                recommend = select_arm_result

                # update Q table     
                agent.update(q_learn_index, reward)
                if reward == 1:
                    agent.updateEpsilon()
        
                print("q_value: ", agent.q_table[q_learn_index])
            
            factor_time = user_behavior.factor_time(recommend[1], userID, model.charging_data)
            factor_cate = user_behavior.factor_cate(model.location, recommend[0], userID)
            factor_dist = user_behavior.factor_dist(model.location, model.charging_data, recommend[0], userID, TESTING_START_DATE)
            print("factor_time, factor_cate,  factor_dist: ", factor_time, factor_cate, factor_dist)
            dissimilarity = user_behavior.get_dissimilarity(factor_time, factor_cate, factor_dist)
            prob = user_behavior.estimate_willingeness(dissimilarity, model.incentive_cost.index(recommend[3]), SIGMOID_INCENTIVE_UNIT_COST)
            user_accept = user_behavior.get_user_decision(prob)
            print(prob, user_accept)

            user_choose = recommend if user_accept else model.get_user_origin_choose(slots_df, origin_cs, origin_hour, charging_len)
            incentive_nums = model.incentive_cost.index(user_choose[3]) if user_accept else 0
            user_choose_station[user_choose[0]] += 1
            print("user_choose=", user_choose)
            
            schedule_df.loc[len(schedule_df)] = [
                requestID, # requestID
                userID, # userID
                model.current_date + timedelta(hours=user_choose[1]), # datetime
                user_choose[0], # locationID
                charging_len, # chargingLen
                user_choose[2], # score
                model.incentive_cost.index(user_choose[3]), # incentive
                origin_cs, # originLocationID
                origin_hour, # originHour
                user_choose[4], # personal
                user_choose[5], # willingness
                user_choose[6], # trend
                user_choose[7], # cp_value
                user_choose[8], # threshold
                user_accept
            ]

            slots_df = model.update_user_selection(slots_df, model.current_date, user_choose, charging_len)
            model.calculate_utilization(schedule=user_choose, charging_len=charging_len)
            model.average_utilization()
            print(f"progress: {progress}/{len(charging_request)}")

            # except Exception as e:
            #     print(f"{userID}, {charging_len}, {origin_hour}, {origin_cs} ERROR: {e}")

            progress += 1

        for item in user_choose_station.keys():
            print(item, "-", model.location.loc[item, "buildingID"], ":", user_choose_station[item])


        path = f"../Result/Carl/MISP/{TESTING_NAME}/SIGMOID_INCENTIVE_UNIT_COST_{SIGMOID_INCENTIVE_UNIT_COST}/EPSILON_RATE_{EPSILON_RATE}/{TEST_DAY}/alpha_{ALPHA}/"
        # path = f"../Result/Carl/MF/{TESTING_NAME}/SIGMOID_INCENTIVE_UNIT_COST_{SIGMOID_INCENTIVE_UNIT_COST}/DIM_{EMB_DIM}/{TEST_DAY}/alpha_{ALPHA}/"

        if not os.path.isdir(path):
            os.makedirs(path)

        schedule_df.to_csv(path + f"{model.current_date.strftime('%m%d')}.csv", index=None)

        print(f"{day+1} default count: {model.default_count}")
        print(f"========== {day+1} done ==========")
        model.current_date += timedelta(days=1)

        # update average request number
        model.update_average_request_num(model.current_date, user_choose_station)


    # file_path = f'../Result/Carl/MISP/{TESTING_NAME}/SIGMOID_INCENTIVE_UNIT_COST_{SIGMOID_INCENTIVE_UNIT_COST}/{TEST_DAY}/{random_counter}/q_table.json'
    file_path = f'../Result/Carl/MISP/{TESTING_NAME}/SIGMOID_INCENTIVE_UNIT_COST_{SIGMOID_INCENTIVE_UNIT_COST}/EPSILON_RATE_{EPSILON_RATE}/{TEST_DAY}/q_table.json'
    # file_path = f'../Result/Carl/MF/{TESTING_NAME}/SIGMOID_INCENTIVE_UNIT_COST_{SIGMOID_INCENTIVE_UNIT_COST}/DIM_{EMB_DIM}/{TEST_DAY}/q_table.json'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # with open(f'../Result/Carl/MISP/{TESTING_NAME}/SIGMOID_INCENTIVE_UNIT_COST_{SIGMOID_INCENTIVE_UNIT_COST}/{TEST_DAY}/{random_counter}/q_table.json', 'w') as json_file:
    #         json.dump(agent.q_table, json_file)
    with open(file_path, 'w') as json_file:
        json.dump(agent.q_table, json_file)

    end = time.time()
    print(f"Time: {(end-start)/60} min")
