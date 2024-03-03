from email.policy import default
import pandas as pd
import numpy as np
from collections import defaultdict

METHOD = "MISP"

MISP_SIMILARITY_PATH = "./Result/user_similarity.csv"
MISP_PREFERENCE_PATH = "./Result/ev_cs_preference_correction.csv"

MF_SIMILARITY_PATH = "./Result/Baseline/MF/user_similarity.csv"
MF_PREFERENCE_PATH = "./Result/Baseline/MF/ev_cs_preference.csv"


### read user similarity
user_similarity = pd.read_csv(MISP_SIMILARITY_PATH, sep="\t", header=None)

user_df = pd.read_csv("./Dataset/user_list_2.csv", index_col=0)
location_df = pd.read_csv("./Dataset/location_5.csv", index_col="locationId")["buildingID"]
location_df.index = location_df.index.astype(str)
preference_df = pd.read_csv(MISP_PREFERENCE_PATH, sep="\t", names=["user", "chargingStation", "preference"])
# preference_df = preference_df[preference_df.select_dtypes(include=np.number).ge(0).all(axis=1)]
preference_df.set_index(["user", "chargingStation"], drop=True, inplace=True)


### check user with charging station records
user_interaction = defaultdict(list)
for id, row in user_df.iterrows():
    df = pd.read_csv(f"./Dataset/Relation_training_data/{row['name']}.csv", index_col="createdHour")
    for cs in location_df.index:
        check = df.index[df[cs].ge(0)]
        if not check.empty:
            for hour in check.tolist():
                user_interaction[(location_df.at[cs], hour)].append(id)

for id, row in user_df.iterrows():
    df = pd.read_csv(f"./Dataset/Relation_training_data/{row['name']}.csv", index_col="createdHour")

    for hour in range(24):
        for cs in location_df.index.tolist():
            # if user have charging records in this station and time slot
            key = (id, location_df.at[cs])
            if id in user_interaction[(location_df.at[cs], hour)]:
                df.at[hour, cs] = preference_df.at[key, "preference"]
                continue

            unknown_preference = 0
            # related_cs_list = preference_df[(preference_df.index.get_level_values("user") == id) &
            #                                 (preference_df.index.get_level_values("chargingStation") != location_df.at[cs])].index.tolist()
            related_cs_list = preference_df.index.tolist()[:10]
            related_cs_len = len(related_cs_list)
            for _, related_cs_id in related_cs_list:

                other_user_preference = 0
                related_user_list = user_interaction[(location_df.at[cs], hour)]
                related_user_len = len(related_user_list)
                if related_user_len == 0:
                    related_cs_len -= 1
                    continue

                for related_user_id in related_user_list:
                    if not (related_user_id, related_cs_id) in preference_df.index:
                        # temp.add((related_user_id, related_cs_id))
                        related_user_len -= 1
                        continue
                    other_user_preference += (preference_df.at[(related_user_id, related_cs_id), "preference"] * user_similarity.loc[id, related_user_id])
                if related_user_len == 0:
                    related_cs_len -= 1
                    continue

                unknown_preference += (preference_df.at[(id, related_cs_id), "preference"] * (other_user_preference / related_user_len))
            if related_cs_len != 0:
                unknown_preference /= related_cs_len

            df.at[hour, cs] = unknown_preference

    if METHOD == "MISP":
        df.to_csv(f"./Result/MISP/Relation/{row['name']}.csv")
    elif METHOD == "MF":
        df.to_csv(f"./Result/Baseline/MF/Relation/{row['name']}.csv")
    print(f"{id} done")
