'''
MISP + Q_learning + UserBehavior

小問題:
更新條件: 使用者有選擇其中一個充電option
更新方式 (針對使用者選擇的充電 option 中的 charging station)
b_k -= 付出去的 incentive
if a_k >1
    a_k -= 1

'''
import os
import pandas as pd
import time
from datetime import timedelta
from dateutil import parser
from collections import defaultdict
import json
import time
from collections import defaultdict
from operator import itemgetter
from EpsilonGreedy import EpsilonGreedy
from UserBehavior import UserBehavior
from MISP import MISP

### global variable ###
ALPHA = 0.5
TESTING_NAME = "MISP_QLearn_userBehavior"
TEST_DAY = "2023-06-28"

TESTING_START_DATE = parser.parse("2018-07-01")
TESTING_END_DATE = parser.parse("2018-07-08")

EPSILON_RATE = 0.5
MU = 0
SIGMA_SQUARE = 1


class QLearn(MISP):


    def __init__(self):
        
        super().__init__()
        self.user_most_perference = pd.read_csv("../Dataset/user_history_preference.csv")
        self.user_facility_perc_dic = pd.read_json("../Dataset/user_facility_perc_dic.json").to_dict()
        self.user_most_prefer_csID = pd.read_csv("../Dataset/charging_data_2_move.csv")
        self.run_user_most_prefer_csID()
        print(f"ALPHA = {ALPHA}")


    def run_user_most_prefer_csID(self):
        
        self.user_most_prefer_csID["createdNew"] = pd.to_datetime(self.user_most_prefer_csID["createdNew"])
        self.user_most_prefer_csID = self.user_most_prefer_csID[self.user_most_prefer_csID["createdNew"] < TESTING_START_DATE]


    def find_min_diff(self, dp_time, history_preference_time):
        '''
        計算”dp出來得時間點“與“過去使用者偏好的時間點”
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


    def cal_q_value(self, userID, combinations):
        '''
        計算 DP 出來的 option, 將其轉換成 Q_table 對應的 index
        '''
        q_table_list = []
    
        for option in combinations:
            
            csID, hour = option[0], option[1]  # csID , hour
            facility_type = self.location.loc[csID, "FacilityType"]
            percentage = self.user_facility_perc_dic.get(str(userID), {}).get(facility_type, 0)
            self.user_most_perference.set_index('userId') # 設定 index
            user_prefer_hours = self.user_most_perference[self.user_most_perference['userId'] == userID]['createdHour'].values[0]
            perc_interval = self.convert_percentage(percentage) if percentage else "0%"
            time_diff = self.find_min_diff(int(hour), int(user_prefer_hours))
            incentive_num = self.incentive_cost.index(option[3])
            q_table_list.append({(perc_interval, time_diff, incentive_num): option})
        
            # (計算 user 對於 csID 的 facilityType 過去去過的比例, 推薦時間段對於 user喜好時間得差距, 給予的incetive張數, option score)
            # tup[:3] 前三個 elements 去對應 q-table index

        return q_table_list
    

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
    

    def over_threshold_option(self, combinations, cache):

        temp = set()
        for _ , item in enumerate(cache):
            
            score = item[0]
            incentive_cost = item[1]
            max_threshold = item[2]

            if (score/incentive_cost) > max_threshold:
                temp.add(combinations[item[3]])

        combinations = list(temp)

        return combinations
    

    def OGAP_QLearn(self, user_list, slots_df, userID, charging_len, q_table):
        '''
        user_list:
        slots_df:
        userID:
        charging_len:
        '''
        combinations = self.get_all_combinations(user_list, slots_df, userID, charging_len)
        if len(combinations) == 0:
            self.default_count += 1
            return self.default_recommend(self.user_preference[userID], userID, charging_len)
        
        self.threshold = self.set_threshold(combinations, self.current_date) 
        combinations = self.initial_filter(combinations)
        cache = self.establish_cache(combinations) # (total expected score, total incentive cost, max threshold, index)
        
        ### 全部的組合都被篩選掉時給 default 
        if len(combinations) == 0:
            self.default_count += 1
            return self.default_recommend(self.user_preference[userID], userID, charging_len)
        
        for p0 in range(1, len(combinations)):
            # p0: 1, 2, 3, ...., len(combinations)
            # 第 p0 個之前的所有組合
            candidate = list()
            for p1 in range(p0):
                p0_location_threshold = self.threshold[combinations[p0][0]]
                # tup: (score, incentive, cp_value, sub_threshold)
                tup = self.filter_by_threshold(p0_location_threshold, cache, p0, p1)
                candidate.append(tup)

            ### 避免分母是 0 (這裡後來有加 0.1，所以不用再額外判斷)
            p0_cp_value = combinations[p0][2] / combinations[p0][3]
            candidate.append((combinations[p0][2], combinations[p0][3], p0_cp_value, self.threshold[combinations[p0][0]]))

            ### 從每一輪的 candidate 中找 CP 值最大的取代
            win_idx = candidate.index(max(candidate, key=itemgetter(2)))
            cache[p0] = (candidate[win_idx][0], candidate[win_idx][1], candidate[win_idx][3], win_idx)
        
        combinations = self.over_threshold_option(combinations, cache)

        return self.get_q_learning_recommend(userID, combinations, q_table)


    def get_q_learning_recommend(self, userID, combinations, q_table):
        
        q_table_list = self.cal_q_value(userID, combinations)
        index_option_dict = self.filter_q_value(q_table_list)
        index_option_list = list(index_option_dict.items())
        sort_score = sorted(index_option_list, key=lambda x: x[1][2], reverse=True) # q_value 由大排到小

        q_dict = defaultdict() # index 對應的 q_value

        # index: (計算 user 對於 csID 的 facilityType 過去去過的比例, 推薦時間段對於 user喜好時間得差距, 給予的incetive張數)
        for idx in index_option_dict.keys():
            q_value = q_table[str(idx)]
            q_dict[idx] = int(q_value)
        
        max_q_value = max(q_dict.values()) # 最大的 Q value

        score_max_index = str(sort_score[0][0])
        score_max_recommend = sort_score[0][1]
        
        q_index = str(sort_score[0][0])
        q_learning_recommend = sort_score[0][1]
        for index, value in dict(sort_score).items():
            if q_table[str(index)] == max_q_value:
                q_learning_recommend = value
                q_index = str(index)
                break
        
        return q_learning_recommend, q_index, score_max_recommend, score_max_index
    

    def get_user_origin_choose(self, slots_df, origin_cs, origin_hour, charging_len):
        
        hour = origin_hour
        for _ in range(24):
            hour %= 24
            parking_slots = self.get_residual_slots(slots_df, origin_cs, hour, charging_len)
            if parking_slots > 0:
                return (origin_cs, hour, -1, 0.1)
            hour += 1



if __name__ == "__main__":

    start = time.time()
    agent = EpsilonGreedy(epsilon=EPSILON_RATE)
    agent.initialize() # 紀錄 index 被選的次數
    model = QLearn()
    user_behavior = UserBehavior()
    
    model.current_date = TESTING_START_DATE 

    user_choose_station = defaultdict(lambda: 0)
    MAX_INCENTIVE_NUMS = len(model.incentive_cost)

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
        ]

        schedule_df = pd.DataFrame([], columns=columns)
        

        for rID, userID, charging_len, origin_hour, origin_cs in charging_request:
            
            try:
                q_learn_recommend, q_learn_index, score_max_recommend, score_max_index = model.OGAP_QLearn(user_list, slots_df, userID, int(charging_len), agent.q_table)
                origin_request = model.get_user_origin_choose(slots_df, origin_cs, origin_hour, charging_len)
                select_arm_result, reward = agent.select_arm(q_learn_recommend, score_max_recommend)
        
                incentive_nums = model.incentive_cost.index(q_learn_recommend[3])  # incentive 數量
                incentive_norm = incentive_nums / MAX_INCENTIVE_NUMS
                factor_time = user_behavior.factor_time(q_learn_recommend[1], userID)
                factor_cate = user_behavior.factor_cate(model.location, q_learn_recommend[0], userID)
                factor_dist = user_behavior.factor_dist(model.location, model.user_most_prefer_csID, q_learn_recommend[0], userID)
                dissimilarity = user_behavior.get_dissimilarity(factor_time, factor_cate, factor_dist)
                
                x = user_behavior.get_distribution_x(dissimilarity, incentive_norm)
                y = user_behavior.get_norm_y(x, MU, SIGMA_SQUARE)
                user_accept = user_behavior.get_user_decision(y)

                user_choose = select_arm_result if user_accept else model.get_user_origin_choose(slots_df, origin_cs, origin_hour, charging_len)
                # update Q table     
                agent.update(q_learn_index, reward)
                if reward == 1:
                    agent.updateEpsilon()
        
                incentive_nums = model.incentive_cost.index(user_choose[3]) if user_accept else 0
                user_choose_station[user_choose[0]] += 1

                print("user_choose=", user_choose)
                print("q_value: ", agent.q_table[q_learn_index])
                
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


        path = f"../Result/Carl/MISP/{TESTING_NAME}/{TEST_DAY}/alpha_{ALPHA}/"

        if not os.path.isdir(path):
            os.makedirs(path)

        schedule_df.to_csv(path + f"{model.current_date.strftime('%m%d')}.csv", index=None)

        print(f"{day+1} default count: {model.default_count}")
        print(f"========== {day+1} done ==========")
        model.current_date += timedelta(days=1)

        # update average request number
        model.update_average_request_num(model.current_date, user_choose_station)

    file_path = f'../Result/Carl/MISP/{TESTING_NAME}/alpha_{ALPHA}/q_table.json'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(f'../Result/Carl/MISP/{TESTING_NAME}/alpha_{ALPHA}/q_table.json', 'w') as json_file:
            json.dump(agent.q_table, json_file)


    end = time.time()
    print(f"Time: {(end-start)/60} min")
