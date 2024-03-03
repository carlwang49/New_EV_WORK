from collections import defaultdict
import pandas as pd
import numpy as np
from haversine import haversine_vector, Unit
from sklearn.utils import shuffle

TYPE_DICT = {
    "咖啡": ["coffee", 1], 
    "雜貨": ["grocery", 2], 
    "餐廳": ["restaurant", 3], 
    "休閒": ["leisure", 4],
    "coffee": 0, 
    "grocery": 1, 
    "restaurant": 2, 
    "leisure": 3
}

FIRST_EXECUTE = False


def generate_relation():
    '''
    generate relationID

    input: None
    output: "./Dataset/relationid.txt"
    '''
    global  TYPE_DICT
    
    relations = ["coffee", "grocery", "restaurant", "leisure", "distance"]
    
    for hour in range(24):
        relations.append(f"time{hour}")
    
    relations.insert(0, len(relations))
    relations = pd.DataFrame(relations, columns=["name"])
    relations["num"] = [i for i in range(len(relations))]
    relations.iat[0, 1] = ""
    relations.to_csv("./Dataset/relationid.txt", sep="\t", header=None, index=None)


def generate_entity():
    '''
    generate entityID (buildingID)

    input: None
    output: "./Dataset/entityid.txt"
    '''
    df = pd.read_csv("./Dataset/location_5.csv", index_col=0)
    df = df.sort_values(by="buildingID")
    df[["locationId", "buildingID"]].to_csv("./Dataset/entityid.txt", sep="\t", header=False, index=False)


def generate_training_data():
    '''
    generate training data of KGE
    each row = ["from_buildingID", "to_buildingID", "relationID"]

    input: "./Dataset/location_5.csv" building information
    output: "./Dataset/train.txt"
    '''

    df = pd.read_csv("./Dataset/location_5.csv", index_col=0)
    df = df.sort_values(by="buildingID")
    train_df = pd.DataFrame([], columns=["from", "to", "relation"])

    ### DISTANCE ###

    ### [old] 每個充電站依照距離建立
    # store_distance = pd.cut(store_distance.flatten(), bins=[-1, 0, 20, 60, 110, 160], labels=[0, 1, 2, 3, 4])
    # distance_matrix = np.array(store_distance).reshape(20, 20)
    # # print(distance_matrix)
    # train_df = pd.DataFrame([], columns=["from", "to", "relation"])
    # for row in range(0, distance_matrix.shape[0]):
    #     for column in range(0, distance_matrix.shape[1]):
    #         if distance_matrix[row][column] != 0:
    #             train_df.loc[len(train_df)] = [row, column, distance_matrix[row][column] + 3]

    store = df[["Latitude", "Longitude"]].to_numpy()
    
    # store_distance 是每個 CS 到其他每個 CS 的距離
    store_distance = haversine_vector([tuple(item) for item in store], [tuple(item) for item in store], Unit.KILOMETERS, comb=True) 

    # distance_sort 會按照大小(由小到大)排對應的 index (跟自己最近，所以第一個的 index 一定是自己的 index)
    distance_sort = np.argsort(store_distance, axis=1)
    
    for cs in distance_sort:
        for other_cs in cs[1:11]:
            train_df.loc[len(train_df)] = [cs[0], other_cs, 4]

    # print(pd.value_counts(store_distance))
    ################

    ### BUSINESS TYPE ###
    for i, i_raw in df.iterrows():
        for j, j_raw in df.iterrows():
            if i != j and i_raw["Typing"] == j_raw["Typing"]:
                # 同一個 type 的記錄下來
                train_df.loc[len(train_df)] = [i_raw["buildingID"], j_raw["buildingID"], TYPE_DICT[i_raw["Typing"]]]
    
    ################

    ### BUSINESS TIME ###

    ### [old] 將營業時間切為多個區間
    # time_period = [[0, 7], [7, 11], [11, 16], [16, 21], [21, 24]]
    # time_label = defaultdict(list)
    # for _, raw in df.iterrows():
    #     label = set()
    #     start_time, end_time = raw["time"].split("-")
    #     start_time = start_time[:2]
    #     end_time = end_time[:2]
    #     for hour in range(int(start_time), int(end_time)):
    #         for idx, item in enumerate(time_period):
    #             if hour in range(item[0], item[1]):
    #                 label.add(idx)
    #     # print(raw["buildingID"], ":", label)
    #     while label:
    #         time_label[label.pop()].append(raw["buildingID"])
    # for key in time_label.keys():
    #     for i in time_label[key]:
    #         for j in time_label[key]:
    #             if i != j:
    #                 train_df.loc[len(train_df)] = [i, j, key + 8]

    business_time = np.zeros((20, 24))
    for _, raw in df.iterrows():
        start_time, end_time = raw["Business_Time"].split("-")
        start_time = int(start_time[:2])
        end_time = int(end_time[:2])
        
        if end_time == 24:
            end_time -= 1
        for hour in range(start_time, end_time+1):
            business_time[raw["buildingID"]][hour] = 1
    

    for hour in range(24):
        for cs in range(20):
            for other_cs in range(20):
                if cs != other_cs and (business_time[cs][hour] == business_time[other_cs][hour] == 1):
                    train_df.loc[len(train_df)] = [cs, other_cs, 5 + hour]

    #####################

    train_df = shuffle(train_df)
    train_df.to_csv("./Dataset/train_222.txt", sep="\t", header=None, index=None)

if __name__ == "__main__":

    if FIRST_EXECUTE:
        generate_relation()
        generate_entity()
    else:
        generate_training_data()