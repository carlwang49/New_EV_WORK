import pandas as pd
import numpy as np
from geopy.distance import great_circle
import numpy as np
from scipy.stats import norm
import random

EPSILON = 0.01

class UserBehavior():


    def __init__(self) -> None:
        
        self.user_history_perference = pd.read_csv("../Dataset/user_history_preference.csv")
        self.user_facility_perc_dic = pd.read_json("../user_facility_perc_dic.json").to_dict()


    def get_max_distence(self, location_df):
        '''
        location_dataframe: 所有的 loaction 資訊
        '''

        max_distance = 0
        for i in range(len(location_df)):
            for j in range(i+1, len(location_df)):
                loc1 = (location_df.iloc[i]['Latitude'], location_df.iloc[i]['Longitude'])
                loc2 = (location_df.iloc[j]['Latitude'], location_df.iloc[j]['Longitude'])
                distance = great_circle(loc1, loc2).km  
                max_distance = max(max_distance, distance)

        return max_distance

    
    def factor_time(self, recommend_hour, userID):
        '''
        recommend_hour:
        userID:
        '''
        user_most_prefer_hour = self.user_history_perference.loc[self.user_history_perference['userId'] == userID, 'createdHour'].item()
        f_time = recommend_hour - user_most_prefer_hour
        f_time_min, f_time_max = 0, 23
        factor_time = (f_time - f_time_min) / (f_time_max - f_time_min)
        factor_time = factor_time if factor_time != 0 else 0.001

        return factor_time


    def factor_cate(self, location_df, recommend_csID, userID):
        
        csID_type = location_df.loc[recommend_csID, 'FacilityType'].item()
        ratio = self.user_facility_perc_dic[userID][csID_type]
        ratio = ratio if not np.isnan(ratio) else 0

        f_cate = 1 - ratio
        f_cate_min, f_cate_max = 0, 1
        factor_cate = (f_cate - f_cate_min) / (f_cate_max - f_cate_min)
        factor_cate = factor_cate if factor_cate != 0 else 0.001
        
        return factor_cate


    def factor_dist(self, location_df, user_most_prefer_csID, recommend_csID, userID):
        
        user_data = user_most_prefer_csID[user_most_prefer_csID['userId'] == userID]
        most_perfer_locationId = user_data['locationId'].value_counts().idxmax()

        recommend_loc = (location_df.loc[str(recommend_csID), 'Latitude'], location_df.loc[str(recommend_csID), 'Longitude'])
        most_prefer_loc = (location_df.loc[str(most_perfer_locationId), 'Latitude'], location_df.loc[str(most_perfer_locationId), 'Longitude'])
        f_dist = great_circle(recommend_loc, most_prefer_loc).km
        f_dist_min = 0
        f_dist_max = self.get_max_distence(location_df)
        factor_dist = (f_dist - f_dist_min) / (f_dist_max - f_dist_min)
        factor_dist = factor_dist if factor_dist != 0 else 0.001
        
        return factor_dist


    def get_dissimilarity(self, factor_time, factor_cate, factor_dist):

        dissimilarity = 0 if factor_time * factor_cate * factor_dist == 0.001 * 0.001 * 0.001 else factor_time * factor_cate * factor_dist

        return dissimilarity
    

    def get_incentive_num(self, remommend_incentive, incentive_cost):
        
        return incentive_cost.index(remommend_incentive) + 1   # incentive 數量


    def get_distribution_x(self, dissimilarity, incentive_norm):

        return dissimilarity / (incentive_norm + EPSILON)
    

   
    def get_norm_y(self, request_x, mu, sigma_square):
        
        x_start = -5
        x_end = 5
        step = 0.001

        # Build normal distribution
        xnormal = np.arange(
            start=x_start,
            stop=x_end+step,
            step=step
        )
        ynormal = norm.pdf(
            x=xnormal,
            loc=mu,
            scale=sigma_square ** 0.5
        )

        x_n = [round(v, 3) for v in xnormal]
        y_n = [round(v, 3) for v in ynormal]


        # Get corresponding Y value
        x_adjust = round(request_x, 3)

        if x_adjust in x_n:
            norm_y = y_n[x_n.index(x_adjust)]

        else:
            norm_y = 0
        
        return norm_y

    def get_user_decision(self, accept_probability):
        
        random_num = random.random()

        # Compare the random number with the threshold
        if random_num <= accept_probability:
            return True
        else:
            return False
        


        