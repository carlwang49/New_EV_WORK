'''
incentive_unit : (0.25, 0.35, 0.45) x cost_unit: (0.1, 0.2, 0.3)
incentive_cost = incentive_num * cost_unit + 0.1
'''
import os
import math
import pandas as pd
import numpy as np
import time
from datetime import timedelta
from dateutil import parser
from collections import defaultdict
from base import base
from numpy import log as ln

### global variable ###
ALPHA = 0.5
TESTING_NAME = "MISP_incentive_045_cost_02"
TEST_DAY = "2023-07-01"

INCENTIVE_UNIT = 0.45
COST_UNIT = 0.2


PERCENTAGE = 100
INCENTIVE_UPPER = 10
INCENTIVE_BUDGET = 400

TESTING_START_DATE = parser.parse("2018-07-01")
TESTING_END_DATE = parser.parse("2018-07-08")


class MISP(base):
    
    def __init__(self):

        super().__init__()
        self.current_date = None
        self.budget = None
        self.threshold = None
        self.threshold_max = 0
        self.default_count = 0
        self.incentive_cost = [round(incentive_num * COST_UNIT + 0.1, 1) for incentive_num in range(0, 11)]
        self.user_preference = defaultdict()
        self.load_matrix = np.zeros((len(self.location), 24)) # 計算各個充電站在每一時刻的使用率
        self.spendable_parking_slots_matrix = np.zeros((len(self.location), 24)) # row: locationId, column: 0~23 (hour) 每天可使用的充電樁
        self.usage_slots_matrix = np.zeros((len(self.location), 24)) # 已使用的充電樁為數量
        self.average_utilization_ratio = np.zeros(24) # 平均所有充電站在某時刻 0~23 (hour) 的使用率 0~23 (hour)
        self.get_user_preference()


    def set_budgets(self):
        '''
        設定每個充電站的預算
        依照不同充電站歷史請求數量分配, 數量越多分配到的越少
        cs_i_budget = ((max_request / request_i) / sum(max_request/request_i)) * total_budget)
        這裡也可以改成用"歷史超過契約容量"的比例去配
        '''
        self.budget = pd.Series([0 for _ in range(20)], index=self.location.index, name="budget")
        ratio_series = pd.Series([0 for _ in range(20)], index=self.location.index, name="ratio")

        location_avg_request_nums_df = self.average_request_df[self.average_request_df["datetime"] == self.current_date].copy()

        for _, row in location_avg_request_nums_df.iterrows():
            ratio_series[row["locationId"]] = (location_avg_request_nums_df["num"].max() + 1) / (row["num"] + 1)

        unit = INCENTIVE_BUDGET / ratio_series.sum()

        for _, row in location_avg_request_nums_df.iterrows():
            self.budget[row["locationId"]] = math.floor(unit * ratio_series[row["locationId"]])

        print("total budget =", self.budget.sum())


    def set_threshold(self, combinations, date):
        '''
        設定每個建物的門檻值
        combinations:[(csID, hour, score, schedule_cost), ....]
        date: 當天的時間
        '''
        max_score = max(combinations, key=lambda x: x[2])[2]
        min_score = min(combinations, key=lambda x: x[2])[2]
    
        threshold = pd.Series([0 for i in range(20)], index=self.budget.index, name="threshold") # 每個建物的門檻值

        for locationID in self.budget.index:
        
            average_day = (self.current_date - TESTING_START_DATE).days
            average_day = average_day if average_day != 0 else 1
            average_request = self.average_request_df.loc[(self.average_request_df["locationId"] == locationID) 
                                                          & (self.average_request_df["datetime"] == date), "num"].values[0] / average_day

            
            # budget 為 0 時，threshold 為 threshold_max
            if self.budget[locationID] == 0:
                threshold[locationID] = self.threshold_max
                continue  
                
            threshold[locationID] = ((max_score + min_score) * average_request) / (2 * self.budget[locationID])

        self.threshold_max = threshold.max()
        
        return threshold # 每個建物的門檻值


    def update_average_request_num(self, date, user_choose_station):
        '''
        更新每個充電站的平均 request 數量
        date: 隔天的時間
        user_choose_station: 所有充電站被選的次數（累積)
        '''
        if date == TESTING_END_DATE:
            return
        
        for cs in self.location.index:
            value = 0 if not cs in user_choose_station.keys() else user_choose_station[cs]
            self.average_request_df.loc[(self.average_request_df["locationId"] == cs) &
                                        (self.average_request_df["datetime"] == date), "num"] = value 
        
        print("after average:", self.average_request_df.loc[self.average_request_df["datetime"] == date, "num"])


    def update_user_selection(self, slots_df, date, schedule, charging_len):
        '''
        schedule: user_choose (csID, hour, score, schedule_cost)
        '''
        # 扣掉用掉的 incentive
        self.budget[schedule[0]] -= (self.incentive_cost.index(schedule[3]))

        # 該時段的充電充減 1
        slots_df.loc[(slots_df["buildingID"] == self.location.loc[schedule[0], "buildingID"]) &
                     (slots_df["datetime"] >= (date + timedelta(hours = schedule[1]))) &
                     (slots_df["datetime"] < (date + timedelta(hours = schedule[1] + charging_len))), "parkingSlots"] -= 1

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
        
        slots_df["parkingSlots"] = slots_df["parkingSlots"].apply(lambda x: max(x, 0)) # 確保充電位不會小於 0
        
        return slots_df


    def average_utilization(self):
        '''
        計算每個小時"所有"充電站的平均使用率
        '''
        hour_load_sum = self.load_matrix.sum(axis=0) # 每個小時，所有充電站的使用率的總和
        for hour in range(24):
            self.average_utilization_ratio[hour] = hour_load_sum[hour] / len(self.location)
        return


    def calculate_utilization(self, slots_df=None, schedule=None, charging_len=0, first=False):
        '''
        計算使用率
        slots_df: 隔天的充電位數量
        schedule: 推薦的充電選項 (locationID, start_hour, expected_score, incentive_cost, hit_count)
        charging_len: 使用者充電長度
        first: 是否為一天的開始 (初始化用)
        '''

        ### 只有初始化會進來 (一天的開始) ###
        if first:
            locationId_list = self.location.index
            for csID in locationId_list:
                for hour in range(24):
                    buildingID = self.location.loc[csID, "buildingID"]
                    # 記錄該 building 該小時可使用的充電為數量
                    self.spendable_parking_slots_matrix[int(buildingID)][hour] = \
                        slots_df.loc[(slots_df["datetime"].dt.hour == hour) &
                        (slots_df["buildingID"] == buildingID), "parkingSlots"]
                    
                    # 初始化使用率
                    self.load_matrix[int(buildingID)][hour] = 0 if self.spendable_parking_slots_matrix[int(buildingID)][hour] != 0 else 1 
                    # 初始化的時候，沒有可用空位代表使用率 100% 
                    self.usage_slots_matrix = np.zeros((len(self.location), 24))
            return
        
        ### 安排一個推薦後，計算使用率 ###
        buildingID = int(self.location.loc[schedule[0], "buildingID"])
        hour = int(schedule[1])
        charging_len = int(charging_len)
        self.usage_slots_matrix[buildingID][hour : (hour + charging_len)] += 1 # 有使用的時段都增加 1
        # 下面為計算使用率 (目前使用數量/總充電站可使用的數量
        self.load_matrix[buildingID][hour : (hour + charging_len)] = self.usage_slots_matrix[buildingID][hour : (hour + charging_len)] / self.spendable_parking_slots_matrix[buildingID][hour : (hour + charging_len)]
        
        return


    def normalize_interaction(self, df):
        '''
        使用者偏好表需要 normalize 在 0~1 之間

        > df: user preference dataframe (index_col = "createdHour")
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


    def _get_sigmoid_y(self, x):
        
        return 1.0 / (1.0 + np.exp(-x))

    
    def _get_sigmoid_x(self, y):
        
        return ln(y/(1-y))


    def estimate_willingeness(self, personal, incentive_num):

        incentive_unit = INCENTIVE_UNIT
        
        # original_willingness = 1 - dissimilarity
        if personal == 0:
            original_x = -5
        elif personal == 1:
            original_x = 5
        else:
            original_x = self._get_sigmoid_x(personal)

        delta_x = incentive_num * incentive_unit
        changed_willingness = self._get_sigmoid_y(original_x + delta_x)

        return changed_willingness # (final) willingness


    def personal_willingness(self, userID, csID, hour, incentive_num):
        '''
        個人充電意願
        userID
        csID
        hour
        incentive
        '''
        personal = self.user_preference[userID].loc[hour, csID]

        ### 給予 incentive 後的意願最大只會是 1 ###
        # willingness = (personal + marginal_value) if (personal + marginal_value) < 1 else 1
        willingness = self.estimate_willingeness(personal, incentive_num)

        return willingness, personal


    def trend_willingness(self, user_list, userID, csID, hour):
        '''
        群眾充電意願
        user_list
        userID
        csID
        hour
        '''
        sum = 0
        for user_id in user_list:
            if user_id != userID:
                sum += self.user_preference[user_id].loc[hour, csID]

        value = sum / (len(user_list) - 1)
        
        return (1 - value)


    def expected_score(self, user_list, userID, csID, hour, incentive_num, requestID):
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
        incentive_cost = self.incentive_cost[incentive_num] 
        cp_value = score/self.incentive_cost[incentive_num]
    
        return score, personal_willingness, personal_origin, trend, incentive_cost, cp_value


    def get_all_combinations(self, user_list, slots_df, userID, charging_len, requestID):
        '''
        取得所有可能的充電組合
        user_list: 所有使用者 ID
        slots_df: 充電位資訊
        userID: 要排程的那個使用者 ID
        charging_len: 使用者充電時間長度
        '''
        combinations = list()
        locationId_list = self.location.index
        
        ### 給每個充電位及每個充電位其 0~23 小時，其推薦的可能
        for csID in locationId_list:
            for hour in range(24):
                ### check residual parking slots ###
                parking_slots = self.get_residual_slots(slots_df, csID, hour, charging_len)
                if parking_slots <= 0:
                    continue

                ### coupon = 0 代表不給獎勵
                for coupon in range(0, INCENTIVE_UPPER+1):
                    score, personal_willingness, personal_origin, trend, incentive_cost, cp_value = self.expected_score(user_list, userID, csID, hour, coupon, requestID)
                    schedule_cost = self.incentive_cost[coupon] # 不給獎勵的 schedule_cost 為 0.1
                    combinations.append((csID, hour, score, schedule_cost, personal_origin, personal_willingness, trend, incentive_cost, cp_value))

        return combinations


    def initial_filter(self, combinations):
        '''
        利用門檻值初步篩選
        combinations: 所有可能的組合 (csID, hour, score, schedule_cost)
        schedule[0]: csID
        schedule[1]: hour
        schedule[2]: score
        schdeule[3]: schdeule_cost (incentive 數量)
        '''
        temp = list()
        for _, schedule in enumerate(combinations):
            if ((schedule[2] / schedule[3]) >= self.threshold.loc[schedule[0]]) and (self.incentive_cost.index(schedule[3]) <= self.budget[schedule[0]]):
                option = schedule
                option += (self.threshold.loc[schedule[0]], )
                temp.append(option)
        
        combinations = temp

        return combinations


    def check_createdHour(self, userID, hour, upper_time=2):
        '''
        檢查 default 選項和歷史偏好的差距
        userID
        hour
        upper_time 不能差超過兩小時
        '''
        for prefer_hour in self.user_history_preference[userID]["createdHour"]:
            # 不能差超過兩小時
            if abs(prefer_hour - hour) <= upper_time:
                return True
        
        return False


    def default_recommend(self, preference_df, userID, charging_len):
        '''
        預設推薦
        preference_df: 預測的使用者偏好 (userID 對每一個充電站每個時段的偏好)
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
            charging_len = int(charging_len)

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
        
        personal_origin, personal_willingness, trend, incentive_cost, cp_value, threshold = 0, 0, 0, 0, 0, 0

        return (recommend_cs, recommend_hour, -1, 0.12, personal_origin, personal_willingness, trend, incentive_cost, cp_value, threshold)


    def OGAP(self, user_list, slots_df, userID, charging_len, requestID):
        '''
        user_list
        slots_df
        userID
        charging_len
        '''
        combinations = self.get_all_combinations(user_list, slots_df, userID, charging_len, requestID) 
        self.threshold = self.set_threshold(combinations, self.current_date) # 每天的 threshold 都不一樣
        combinations = self.initial_filter(combinations)
        
        ### 全部的組合都被篩選掉時給 default 
        if len(combinations) == 0:
            self.default_count += 1
            return self.default_recommend(self.user_preference[userID], userID, charging_len)

        recommend = self.choose_recommend(combinations)

        return recommend


    def choose_recommend(self, recommend):
        '''
        模擬使用者如何選擇
        策略:
        1.) 選期望分數最大的
        2.) random choice
        3.) weighted random choice
        '''
        ### 1.)
        recommend = sorted(recommend, key=lambda tup: tup[2])
        return recommend[-1]


if __name__ == "__main__":
    
    print(f"ALPHA = {ALPHA}")
    start = time.time()

    misp = MISP()
    misp.current_date = TESTING_START_DATE
    user_choose_station = defaultdict(lambda:0)

    for day in range(7):

        ### User interaction matrix ###
        user_list = misp.get_user_list(misp.current_date) # 7/1 以前的 userID list
        misp.get_user_interaction_value(user_list) # self.user_preference

        ### Spendable parking slots prediction ###
        slots_df = misp.get_parking_slots(misp.building_predict_data, misp.current_date)
        slots_df = misp.reserve_parking_slots(slots_df, misp.current_date) # 會根據前一天的使用狀況減少充電位的數量

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
            "threshold",
            "personal",
            "willingness",
            "trend",
            "cost",
            "cp_value"
        ]
        
        schedule_df = pd.DataFrame([], columns=columns)
        for requestID, userID, charging_len, origin_hour, origin_cs in charging_request:
            try:
                user_choose = misp.OGAP(user_list, slots_df, userID, charging_len, requestID)

                user_choose_station[user_choose[0]] += 1
                
                print("user_choose =", user_choose)
                
                # csID, hour, score, schedule_cost, personal, willingness, trend, cost, cp_value, threshold
                schedule_df.loc[len(schedule_df)] = [
                    requestID, # requestID
                    userID, # userID
                    misp.current_date + timedelta(hours=user_choose[1]), # datetime
                    user_choose[0], # locationID
                    charging_len, # chargingLen
                    user_choose[2], # score
                    misp.incentive_cost.index(user_choose[3]), # incentive
                    origin_cs, # originLocationID
                    origin_hour, # originHour
                    user_choose[9], # threshold
                    user_choose[4], # personal
                    user_choose[5], # willingness
                    user_choose[6], # trend
                    user_choose[7], # cost
                    user_choose[8], # cp_value
                ]

                slots_df = misp.update_user_selection(slots_df, misp.current_date, user_choose, charging_len)
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
        misp.current_date+= timedelta(days=1)

        ### update average request number
        misp.update_average_request_num(misp.current_date, user_choose_station)

    end = time.time()
    print(f"Time: {(end-start)/60} min")

