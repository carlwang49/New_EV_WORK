'''
計算各個使用者，過去使用各個 facilty Type 的使用比例
'''
import json
import itertools
import pandas as pd


def map_to_facility(location_file):

    # 讀 location.csv
    location_df = pd.read_csv(location_file)
    location_facilityType_dict = dict(zip(location_df['locationId'], location_df['FacilityType']))

    return location_facilityType_dict


def estimate_charging_facility(location_facilityType_dict, charging_record_file):
    '''
    計算各個使用者，過去使用各個 facilty Type 的次數
    '''
   
    # 充電紀錄
    charg_df = pd.read_csv(charging_record_file)

    # create a new list (facilityType) to charg_df
    facility_type_column = []
    for row in charg_df['locationId']:
        
        # 找到其對應的 Facility type 
        facility_type = location_facilityType_dict[row]

        # append to add_facility_list
        facility_type_column.append(facility_type)

    charg_df['facilityType'] = facility_type_column

    # group by based on userId and facilityType
    group_df = charg_df.groupby(['userId', 'facilityType'])
    user_facility_times_dic = {}


    # for loop groupby result
    for key, item in group_df:

        userId = key[0]  # the first key value is userId
        facilityType = key[1]  # second key value is facilityType

        # assign a default value '{}' only if userId hasn't be a key yet.
        user_facility_times_dic.setdefault(userId, {})

        # df.shape[0] is the number of rows in a dataframe
        # record item.shape[0] (the times of userId charges at facilityType)
        # item.shape[0] 是維度大小
        user_facility_times_dic[userId][facilityType] = item.shape[0]
    
    return user_facility_times_dic


def convert_estimate_charging_facility(user_facility_times_dic):

    user_facility_perc_dic = {}

    for key, value in user_facility_times_dic.items():

        total = sum(value.values())
        user_facility_perc_dic[str(key)] = {str(
            type): times/total for type, times in value.items()}
   
    with open('./Dataset/user_facility_perc_dic.json', 'w') as f:
        json.dump(user_facility_perc_dic, f)


def main(location_file, charging_record_file):

    location_facilityType_dict = map_to_facility(location_file) # locationID 和其對應的 FacilityType

    user_facility_times_dic = estimate_charging_facility(location_facilityType_dict, charging_record_file)
    convert_estimate_charging_facility(user_facility_times_dic)


if __name__ == "__main__":

    location_file = 'Dataset/location_5.csv' # location.csv
    charging_record_file = 'Dataset/charging_data_2_move.csv' # 充電紀錄
    main(location_file, charging_record_file)
