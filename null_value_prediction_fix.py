from email.policy import default
import pandas as pd
import numpy as np
import math
from collections import defaultdict

METHOD = "MISP"
EMB_DIM = 210

MISP_SIMILARITY_PATH = f"./Result/DIM_{EMB_DIM}/user_similarity.csv"
# MISP_PREFERENCE_PATH = f"./Result/ev_cs_preference_correction.csv"
MISP_PREFERENCE_PATH = f"./Result/DIM_{EMB_DIM}/ev_cs_preference_correction.csv"

MF_SIMILARITY_PATH = "./Result/Baseline/MF/user_similarity.csv"
MF_PREFERENCE_PATH = "./Result/Baseline/MF/ev_cs_preference.csv"


### read user similarity
# user_similarity = pd.read_csv(MF_SIMILARITY_PATH, sep="\t", header=None)
user_similarity = pd.read_csv(MISP_SIMILARITY_PATH, sep="\t", header=None)

user_df = pd.read_csv("./Dataset/user_list_2.csv", index_col=0)
location_df = pd.read_csv("./Dataset/location_5.csv", index_col="locationId")["buildingID"]
location_df.index = location_df.index.astype(str)
# preference_df = pd.read_csv(MF_PREFERENCE_PATH, sep="\t", names=["user", "chargingStation", "preference"])
preference_df = pd.read_csv(MISP_PREFERENCE_PATH, sep="\t", names=["user", "chargingStation", "preference"])
# preference_df = preference_df[preference_df.select_dtypes(include=np.number).ge(0).all(axis=1)]
preference_df.set_index(["user", "chargingStation"], drop=True, inplace=True)


### check user with charging station records
user_interaction = defaultdict(list)
user_charging_record = defaultdict(dict)
for id, row in user_df.iterrows():
    df = pd.read_csv(f"./Dataset/Relation_training_data/{row['name']}.csv", index_col="createdHour")
    for cs in location_df.index:
        check = df.index[df[cs].ge(0)]
        if not check.empty:
            for hour in check.tolist():
                user_charging_record[id][(location_df.at[cs], hour)] = df.at[(hour, cs)]
                user_interaction[(location_df.at[cs], hour)].append(id)

    # 充電紀錄 normalize
    max_value = max(user_charging_record[id].values())
    min_value = min(user_charging_record[id].values())
    for key in user_charging_record[id].keys():
        if max_value == min_value == 1:
            continue
        user_charging_record[id][key] = math.ceil(((user_charging_record[id][key] - min_value) / (max_value - min_value)) * 4 + 1)

for id, row in user_df.iterrows():
    df = pd.read_csv(f"./Dataset/Relation_training_data/{row['name']}.csv", index_col="createdHour")

    for hour in range(24):
        for cs in location_df.index.tolist():
            cs_id = location_df.at[cs]
            if not (cs_id, hour) in user_interaction.keys():
                continue

            masses = 0
            for related_user_id in user_interaction[(cs_id, hour)]:
                masses += user_charging_record[related_user_id][(cs_id, hour)] * user_similarity.loc[id, related_user_id]
            masses /= len(user_interaction[(cs_id, hour)])

            df.at[hour, cs] = preference_df.at[(id, cs_id), "preference"] * masses

    if METHOD == "MISP":
        df.to_csv(f"./Result/MISP/Relation_fix/DIM_{EMB_DIM}/{row['name']}.csv")
    elif METHOD == "MF":
        df.to_csv(f"./Result/Baseline/MF/Relation_0818_fix/{row['name']}.csv")
    print(f"{id} done")
