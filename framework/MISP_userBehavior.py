'''
MISP + UserBehavior

'''
import os
import math
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from dateutil import parser
from collections import defaultdict
from operator import itemgetter
from base import base
from UserBehavior import UserBehavior

### global variable ###
ALPHA = 0.5
TESTING_NAME = "MISP_userBehavior"
TEST_DAY = "2023-06-27"

INCENTIVE_UPPER = 10
INCENTIVE_BUDGET = 400
TESTING_START_DATE = parser.parse("2018-07-01")
TESTING_END_DATE = parser.parse("2018-07-08")

MU = 0
SIGMA_SQUARE = 1


class MISP(base):

    def __init__(self):

        super().__init__()
        self.budget = None
        self.threshold = None
        self.current_date = None
        self.default_count = 0
        self.incentive_cost = [0.1, 0.12, 0.16, 0.25, 0.4, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05] # 給獎勵的成本
        self.user_preference = defaultdict()
        self.threshold_max = 0
        self.load_matrix = np.zeros((len(self.location), 24))
        self.spendable_parking_slots_matrix = np.zeros((len(self.location), 24))
        self.usage_slots_matrix = np.zeros((len(self.location), 24))
        self.average_utilization_ratio = np.zeros(24)
        self.average_request_df = pd.read_csv("../Dataset/average_request_num.csv", index_col=0)
        self.average_request_df["datetime"] = pd.to_datetime(self.average_request_df["datetime"], format="%Y-%m-%d")
        self.average_request_df["locationId"] = self.average_request_df["locationId"].astype(str)

        self.user_most_prefer_csID = pd.read_csv("../Dataset/charging_data_2_move.csv")
        self.run_user_most_prefer_csID()
        self.get_user_preference()
        print(f"ALPHA = {ALPHA}")


    def run_user_most_prefer_csID(self):
        
        self.user_most_prefer_csID["createdNew"] = pd.to_datetime(self.user_most_prefer_csID["createdNew"])
        self.user_most_prefer_csID = self.user_most_prefer_csID[self.user_most_prefer_csID["createdNew"] < TESTING_START_DATE]


    def set_budgets(self):
        '''
        設定每個充電站的預算
        (依照不同充電站歷史請求數量分配，數量越多分配到的越少; 這裡可以改成用歷史超過契約容量的比例去配)
        '''
        self.budget = pd.Series([0 for _ in range(20)], index=self.location.index, name="budget")

        ratio = pd.Series([0 for _ in range(20)], index=self.location.index, name="ratio")
        request_df = self.average_request_df[self.average_request_df["datetime"] == self.current_date].copy()
        
        for _, row in request_df.iterrows():
            ratio[row["locationId"]] = (request_df["num"].max() + 1) / (row["num"] + 1)

        unit = INCENTIVE_BUDGET / ratio.sum()
        for _, row in request_df.iterrows():
            self.budget[row["locationId"]] = math.floor(unit * ratio[row["locationId"]])

        print("total budget =", self.budget.sum())


    def set_threshold(self, combinations, date):
        '''
        設定每個建物的門檻值
        '''
        max_score = max(combinations, key=lambda x: x[2])[2]
        min_score = min(combinations, key=lambda x: x[2])[2]

        threshold = pd.Series([0 for i in range(20)], index=self.budget.index, name="threshold")
        
        for locationID in self.location.index:
            
            average_day = (self.current_date - TESTING_START_DATE).days
            average_day = average_day if average_day != 0 else 1
            average_request = self.average_request_df.loc[
                (self.average_request_df["locationId"] == locationID) & 
                (self.average_request_df["datetime"] == date), "num"].values[0] / average_day
            
            if self.budget[locationID] == 0:
                threshold[locationID] = self.threshold_max
                continue

            threshold[locationID] = ((max_score + min_score) * average_request) / (2 * self.budget[locationID])

        self.threshold_max = threshold.max()

        return threshold


    def update_average_request_num(self, date, user_choose_station, average_day):
        '''
        更新每個充電站的平均數量
        '''
        if date == TESTING_END_DATE:
            return

        for cs in self.location.index:
            value = 0 if not cs in user_choose_station.keys() else user_choose_station[cs]
            self.average_request_df.loc[
                (self.average_request_df["locationId"] == cs) & 
                (self.average_request_df["datetime"] == date), "num"] = value
    
        print("after average:", self.average_request_df.loc[self.average_request_df["datetime"] == date, "num"])


    def update_user_selection(self, slots_df, date, schedule, charging_len, user_accpet):
        
        if user_accpet:
            self.budget[schedule[0]] -= (self.incentive_cost.index(schedule[3]))

        slots_df.loc[(slots_df["buildingID"] == self.location.loc[schedule[0], "buildingID"]) &
                     (slots_df["datetime"] >= (date + timedelta(hours=schedule[1]))) &
                     (slots_df["datetime"] < (date + timedelta(hours=schedule[1] + charging_len))), "parkingSlots"] -= 1

        return slots_df


    def reserve_parking_slots(self, slots_df, date):
        '''
        保留充電位
        slots_df: 預測隔天的充電位數量
        date: 隔天日期
        '''
        for cs in range(20):
            yesterday_building_df = self.building_truth_data.loc[(self.building_truth_data["buildingID"] == str(cs)) &
                                                                 (self.building_truth_data["datetime"] >= (date - timedelta(days=1))) &
                                                                 (self.building_truth_data["datetime"] < date)].copy()
            
            contract_capacity = self.location.loc[self.location["buildingID"] == str(cs), "contractCapacity"].values[0]

            # 前一天電力使用 30% 保留一個充電位
            for _, raw in yesterday_building_df.iterrows():
                usage_elec = raw["consumption"] - raw["generation"]
                if (contract_capacity * 0.3) <= usage_elec:
                    slots_df.loc[(slots_df["buildingID"] == str(cs)) &
                                (slots_df["datetime"] == (raw["datetime"] + timedelta(days=1))), "parkingSlots"] -= 1

        slots_df["parkingSlots"] = slots_df["parkingSlots"].apply(lambda x: max(x, 0))

        return slots_df



    def average_utilization(self):
        '''
        計算每個小時所有充電站的使用率平均
        '''
        hour_load_sum = self.load_matrix.sum(axis=0)
        for hour in range(24):
            self.average_utilization_ratio[hour] = hour_load_sum[hour] / len(self.location)
        
        return


    def calculate_utilization(self, slots_df=None, schedule=None, charging_len=0, first=False):
        '''
        計算使用率
        slots_df: 隔天的充電位數量
        schedule: 推薦的充電選項 (locationID, charging_hour, expected_score, incentive_cost, hit_count)
        charging_len: 使用者充電長度
        first: 是否為一天的開始 (初始化用)
        '''
        if first:
            for csID in self.location.index:
                for hour in range(24):
                    buildingID = self.location.loc[csID, "buildingID"]
                    self.spendable_parking_slots_matrix[int(buildingID)][hour] = slots_df.loc[
                        (slots_df["datetime"].dt.hour == hour) & 
                        (slots_df["buildingID"] == buildingID), "parkingSlots"]
                    
                    self.load_matrix[int(buildingID)][hour] = 0 \
                        if self.spendable_parking_slots_matrix[int(buildingID)][hour] != 0 else 1
                    self.usage_slots_matrix = np.zeros((len(self.location), 24))
            return


        buildingID = int(self.location.loc[schedule[0], "buildingID"])
        hour = int(schedule[1])
        charging_len = int(charging_len)
    
        self.usage_slots_matrix[buildingID][hour: (hour + charging_len)] += 1
        self.load_matrix[buildingID][hour: (hour + charging_len)] = \
            (self.usage_slots_matrix[buildingID][hour: (hour + charging_len)]) \
            / (self.spendable_parking_slots_matrix[buildingID][hour: (hour + charging_len)])
        
        return


    def normalize_interaction(self, df):
        '''
        使用者偏好表需要 normalize 在 0~1 之間
        df: user preference dataframe (index_col = "createdHour")
        '''
        max_value = max(df.max().tolist())
        min_value = min(df.min().tolist())
        for column in df.columns:
            df[column] = df[column].apply(lambda x: (x-min_value)/(max_value-min_value))
            
        return df


    def get_user_interaction_value(self, user_list):
        '''
        讀取 relational learning 預測的偏好度
        user_list: 所有使用者 ID
        '''
        for user_id in user_list:
            history = pd.read_csv(f"../Result/MISP/Relation/{user_id}.csv", index_col="createdHour")
            self.user_preference[user_id] = self.normalize_interaction(history)


    def marginal_utility(self, incentive):
        '''
        遞減邊際效益
        incentive: 獎勵值數量
        '''
        increased_probability = [0, 1.25, 2.8, 3.25, 3.6, 3.7, 3.71, 3.71, 3.73, 3.74, 3.75] # 3.7 後固定加上 0.01
        incentive = min(incentive, len(increased_probability))

        return increased_probability[incentive]


    def personal_willingness(self, userID, csID, hour, incentive):
        '''
        個人充電意願
        '''
        personal = self.user_preference[userID].loc[hour, csID]
        marginal_value = personal * self.marginal_utility(incentive)

        # 給予 incentive 後的意願最大只會是 1
        willingness = (personal + marginal_value) if (personal + marginal_value) < 1 else 1
        
        return willingness


    def trend_willingness(self, user_list, userID, csID, hour):
        '''
        群眾充電意願
        '''
        sum = 0
        for id in user_list:
            if id != userID:
                sum += self.user_preference[id].loc[hour, csID]
        value = sum / (len(user_list) - 1)
        
        return (1 - value)


    def expected_score(self, user_list, userID, csID, hour, incentive):
        '''
        計算每個充電選項對系統的期望分數 
        user_list: 所有使用者 ID
        userID: 要排程的那個使用者 ID
        csID: 選項中的充電站 ID
        hour: 選項中的充電時間
        incentive: 選項中獎勵的數量
        '''
        personal = self.personal_willingness(userID, csID, hour, incentive)
        trend = self.trend_willingness(user_list, userID, csID, hour)
    
        load_value = self.average_utilization_ratio[hour] - self.load_matrix[int(self.location.loc[csID, "buildingID"])][hour]
        load_value = load_value / self.average_utilization_ratio.max() if self.average_utilization_ratio.max() != 0 else 1

        return ((ALPHA * personal) + ((1 - ALPHA) * trend)) * load_value


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
                for coupon in range(0, INCENTIVE_UPPER+1):
                    score = self.expected_score(user_list, userID, csID, hour, coupon)
                    schedule_cost = self.incentive_cost[coupon]
                    combinations.append((csID, hour, score, schedule_cost))

        return combinations
            

    def initial_filter(self, combinations):
        '''
        利用門檻值初步篩選
        combinations: 所有可能的組合
        '''
        temp = list()
        for _, schedule in enumerate(combinations):
            # 黑名單時段不採用
            if ((schedule[2] / schedule[3]) >= self.threshold.loc[schedule[0]]) and (self.incentive_cost.index(schedule[3]) <= self.budget[schedule[0]]):
                temp.append(schedule)
        combinations = temp

        return combinations


    def establish_cache(self, combinations):
        '''
        建立 DP 用的 cache
        cache: list(tuple)
        tuple: (total expected score, total incentive cost, max threshold, index)
        combinations: 所有可能的組合
        '''
        cache = list()
        for idx, row in enumerate(combinations):
            cache.append((row[2], row[3], self.threshold[row[0]], idx))

        return cache


    def filter_by_threshold(self, threshold, cache, p0, p1):
        '''
        用門檻值再篩選不同組合
        threshold: 每個充電站的門檻值
        cache: list(tuple)
        p0, p1: 指向兩個選項組合的 index
        '''
        score = cache[p1][0] + cache[p0][0]
        incentive = cache[p1][1] + cache[p0][1]
        cp_value = score / incentive
        sub_thres = max(cache[p1][2], threshold)

        if cp_value < sub_thres:
            cp_value = -1

        return (score, incentive, cp_value, sub_thres)


    def recommend_list(self, combinations, cache, userID):
        '''
        從 cache 中挑選一個組合推薦給使用者
        策略：
        1.) 挑最大 CP 值的，但是只會有一個選項
        2.) 挑最大 CP 值的但是選有兩個(含)以上的選項推薦
        3.) 挑最大分數的，但是偏好值好像比較不佳 (待確認)
        4.) average ranking 同時考慮負載平衡和 CP 值

        combinations: 所有可能的選項 (回溯用)
        cache: DP 計算後的結果
        userID: 要排程的那個使用者 ID
        '''

        ### 1.) & 2.) max CP value ####
        max_value = -999
        for idx, item in enumerate(cache):
            ### 1.) 挑最大 CP 值
            score = item[0]
            incentive_cost = item[1]
            max_value = max(round(score/incentive_cost, 1), max_value)

        all_recommend = list()
        for idx, item in enumerate(cache):
            score = item[0]
            incentive_cost = item[1]
            if round(score/incentive_cost, 1) == max_value:
                ptr = idx
                recommend = list()
                while True:
                    recommend.append(combinations[ptr]) # 找到 cp 值最大的後，把原本的那個組合加入到 recommend 裏面
                    ptr2 = cache[ptr][3]
            
                    if ptr == ptr2:
                        break
                    ptr = ptr2

                all_recommend.append(recommend)

        ### 計算每個組合和使用者偏好的 hit 差異 ###
        max_combination_idx = 0
        max_score = 0
        fixed = False
        try:
            for combination_idx, schedules in enumerate(all_recommend):
                for schedule_idx, schedule in enumerate(schedules):
                    score = 0
                    if self.location.loc[schedule[0], "FacilityType"] == self.user_history_preference[userID]["facilityType"]:
                        score += 1
                    if self.check_createdHour(userID, schedule[1], 3):
                        score += 1
                    all_recommend[combination_idx][schedule_idx] += (score,)
                    if max_score <= score and (not fixed):
                        max_score = score
                        max_combination_idx = combination_idx
        except Exception as e:
            print(e)
            for schedule_idx, _ in enumerate(all_recommend[max_combination_idx]):
                all_recommend[max_combination_idx][schedule_idx] += (0,)

        return all_recommend[max_combination_idx]


    def check_createdHour(self, userID, hour, upper_time=2):
        '''
        檢查 default 選項和歷史偏好的差距
        '''
        for prefer_hour in self.user_history_preference[userID]["createdHour"]:
            if abs(prefer_hour - hour) <= upper_time:
                return True
            
        return False


    def default_recommend(self, preference_df, userID, charging_len):
        '''
        預設推薦
        preference_df: 預測的使用者偏好
        userID: 要排程的那個使用者 ID
        charging_len: 使用者充電時間長度
        '''

        recommend_cs = str(preference_df.idxmax().idxmax())
        recommend_hour = int(preference_df.idxmax()[recommend_cs])
        
        preference_np = preference_df.to_numpy() 
        check = 0
        while check < (20*24):

            hour, locationIdx = np.unravel_index(preference_np.argmax(), preference_np.shape) # perference 最高的 hour 和 locationId
            hour, locationIdx = int(hour), locationIdx

            locationID = preference_df.columns[locationIdx]
            location_budget = self.budget[locationID]

            # 1. 確認 budget 還夠 
            # 2. 使否和過去 user 喜歡的 facilitType 相同 
            # 3. 建議的時間跟過去 user 喜歡的時間相差不超過兩小時 
            # 4. 確定是否用空位
            if ((location_budget > 0) and 
                (self.user_history_preference[userID]["facilityType"] == self.location.loc[locationID, "FacilityType"]) and
                (self.check_createdHour(userID, hour)) and
                (self.spendable_parking_slots_matrix[int(self.location.loc[locationID, "buildingID"])][hour:hour+charging_len].all())): 
                recommend_cs = locationID
                recommend_hour = hour
                print(f"default recommend = ({recommend_cs}, {recommend_hour})")
                break
            
            check += 1
            preference_np[hour][locationIdx] = -10

        return [(recommend_cs, recommend_hour, -1, 0.12)]


    def OGAP(self, user_list, slots_df, userID, charging_len):

        combinations = self.get_all_combinations(user_list, slots_df, userID, charging_len)

        ### 如果都沒有合適的，給 default 
        if len(combinations) == 0:
            self.default_count += 1
            return self.default_recommend(self.user_preference[userID], userID, charging_len)
        
        self.threshold = self.set_threshold(combinations, self.current_date)
        combinations = self.initial_filter(combinations)
        cache = self.establish_cache(combinations)

        # 全部的組合都被篩選掉時給 default
        if len(combinations) == 0:
            self.default_count += 1
            return self.default_recommend(self.user_preference[userID], userID, charging_len)

        for p0 in range(1, len(combinations)):
            candidate = list()
            for p1 in range(p0):
                tup = self.filter_by_threshold(self.threshold[combinations[p0][0]], cache, p0, p1)
                candidate.append(tup)

            # 避免分母是 0 (這裡後來有加 0.1，所以不用再額外判斷)
            p0_cp_value = combinations[p0][2] / combinations[p0][3]
            candidate.append((combinations[p0][2], combinations[p0][3], p0_cp_value, self.threshold[combinations[p0][0]]))

            # 從每一輪的 candidate 中找 CP 值最大的取代
            win_idx = candidate.index(max(candidate, key=itemgetter(2)))
            cache[p0] = (candidate[win_idx][0], candidate[win_idx][1], candidate[win_idx][3], win_idx)

        recommend = self.recommend_list(combinations, cache, userID)

        # 推薦的組合 Omega 內依照偏好度 hit 排序
        return sorted(recommend, key=(lambda tup: tup[4]))
    

    def choose_recommend(self, recommend):
        '''
        模擬使用者如何選擇
        策略:
        1.) 選期望分數最大的
        2.) random choice
        3.) weighted random choice
        '''
        # 1.)
        recommend = sorted(recommend, key=lambda tup: tup[2])
        return recommend[-1]
    

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

    misp = MISP()
    user_behavior = UserBehavior()
    misp.current_date = TESTING_START_DATE
    user_choose_station = defaultdict(lambda: 0)
    MAX_INCENTIVE_NUMS = len(misp.incentive_cost)

    for day in range(7):

        ### User interaction matrix ###
        user_list = misp.get_user_list(misp.current_date)
        misp.get_user_interaction_value(user_list)

        ### Spendable parking slots prediction ###
        slots_df = misp.get_parking_slots(misp.building_predict_data, misp.current_date)
        slots_df = misp.reserve_parking_slots(slots_df, misp.current_date)

        ### Utilization ###
        misp.calculate_utilization(slots_df=slots_df, first=True)
        misp.average_utilization()

        ### Get building budget and threshold ###
        misp.set_budgets()

        ### EV charging request ###
        charging_request = misp.get_charging_request(misp.current_date)

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
        
        for requestID, userID, charging_len, origin_hour, origin_cs in charging_request:
            
            try:
                recommend = misp.OGAP(user_list, slots_df, userID, charging_len) # (csID, hour, score, schedule_cost)
                user_choose = misp.choose_recommend(recommend)

                incentive_nums = misp.incentive_cost.index(user_choose[3]) + 1  # incentive 數量
                incentive_norm = incentive_nums / MAX_INCENTIVE_NUMS

                factor_time = user_behavior.factor_time(user_choose[1], userID)
                factor_cate = user_behavior.factor_cate(misp.location, user_choose[0], userID)
                factor_dist = user_behavior.factor_dist(misp.location, misp.user_most_prefer_csID, user_choose[0], userID)
                dissimilarity = user_behavior.get_dissimilarity(factor_time, factor_cate, factor_dist)

                x = user_behavior.get_distribution_x(dissimilarity, incentive_norm)
                y = user_behavior.get_norm_y(x, MU, SIGMA_SQUARE)
                user_accept = user_behavior.get_user_decision(y)

                user_choose = user_choose if user_accept else misp.get_user_origin_choose(slots_df, origin_cs, origin_hour, charging_len)
                
                incentive_nums = misp.incentive_cost.index(user_choose[3]) if user_accept else 0
                user_choose_station[user_choose[0]] += 1
                print("user_choose =", user_choose)

                schedule_df.loc[len(schedule_df)] = [
                    requestID,
                    userID,
                    misp.current_date +
                    timedelta(hours=user_choose[1]), # 使用者選擇的時間
                    user_choose[0], # 使用者選擇的充電站
                    charging_len,
                    user_choose[2], # 分數
                    incentive_nums, # 使用者使用的 incentive 
                    origin_cs,
                    origin_hour,
                ]
                
                slots_df = misp.update_user_selection(slots_df, misp.current_date, user_choose, charging_len, user_accept)
                misp.calculate_utilization(schedule=user_choose, charging_len=charging_len)
                misp.average_utilization()
                print(f"progress: {progress}/{len(charging_request)}")

            except Exception as e:
                print(f"{userID}, {charging_len}, {origin_hour}, {origin_cs} ERROR: {e}")
            
            progress += 1

        for item in user_choose_station.keys():
            print(item, "-", misp.location.loc[item, "buildingID"], ":", user_choose_station[item])

        path = f"../Result/Carl/MISP/{TESTING_NAME}/{TEST_DAY}/alpha_{ALPHA}/"
        if not os.path.isdir(path):
            os.makedirs(path)

        schedule_df.to_csv(path + f"{misp.current_date.strftime('%m%d')}.csv", index=None)

        print(f"{day+1} default count: {misp.default_count}")
        print(f"========== {day+1} done ==========")
        misp.current_date += timedelta(days=1)

        # update average request number
        misp.update_average_request_num(misp.current_date, user_choose_station, day+1)

    end = time.time()
    print(f"Time: {(end-start)/60} min")
