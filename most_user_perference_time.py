import pandas as pd

df = pd.read_csv("./Dataset/charging_data_2_move.csv")
most_charging_time_per_user = df.groupby('userId')['createdHour'].agg(lambda x: x.value_counts().index[0])
print(most_charging_time_per_user)
