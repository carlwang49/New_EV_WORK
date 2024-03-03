import pandas as pd
import numpy as np

### 更改 ev_cs_preference ，避免出現負值
EMB_DIM = 210
preference_df = pd.read_csv(f"./Result/DIM_{EMB_DIM}/ev_cs_preference.csv", sep="\t", names=["user", "chargingStation", "preference"])
for i in range(136):
    min_value = preference_df.groupby("user").get_group(i)["preference"].tolist()[-1]
    preference_df.loc[preference_df["user"] == i, "preference"] -= min_value

preference_df.to_csv(f"./Result/DIM_{EMB_DIM}/ev_cs_preference_correction.csv", sep="\t", index=None, header=None)