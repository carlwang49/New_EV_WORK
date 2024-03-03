from email.policy import default
import pandas as pd
import numpy as np
from collections import defaultdict

METHOD = "MF"
MF_PREFERENCE_PATH = "./Result/Baseline/MF/0721_new_relation/ev_cs_preference.csv"
# MF_PREFERENCE_PATH = f"./Result/DIM_{EMB_DIM}/ev_cs_preference_correction.csv"


user_df = pd.read_csv("./Dataset/user_list_2.csv", index_col=0)
location_df = pd.read_csv("./Dataset/location_5.csv", index_col="locationId")["buildingID"]
location_df.index = location_df.index.astype(str)
preference_df = pd.read_csv(MF_PREFERENCE_PATH, sep="\t", names=["user", "prefer_item", "preference"])
print(preference_df)
# preference_df = preference_df[preference_df.select_dtypes(include=np.number).ge(0).all(axis=1)]
# preference_df.set_index(["user", "chargingStation"], drop=True, inplace=True)

for id, row in user_df.iterrows():
    df = pd.read_csv(f"./Dataset/Relation_training_data/{row['name']}.csv", index_col="createdHour")
    user_prefer_df = preference_df.loc[preference_df["user"] == row["label"]].sort_values(by="prefer_item")

    for hour in range(24):
        for locationID in df.columns:
            buildingID = location_df[locationID]
            df.loc[hour, locationID] = user_prefer_df.loc[user_prefer_df["prefer_item"] == int(buildingID)*20+hour, "preference"].values[0]
            exit(0)
            


    df.to_csv(f"./Result/Baseline/MF/0721_new_relation/Relation/{row['name']}.csv")
    print(f"{id} done")
