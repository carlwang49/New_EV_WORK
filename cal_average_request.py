import pandas as pd
import math
from datetime import timedelta
from dateutil import parser
from collections import defaultdict


if __name__ == "__main__":
    
    charging_data = pd.read_csv("./Dataset/charging_data_2_move.csv")
    charging_data['createdNew'] = pd.to_datetime(charging_data['createdNew'])

    # 篩選出 7/1 到 7/7 的數據
    mask = (charging_data['createdNew'] >= '2018-07-01') & (charging_data['createdNew'] <= '2018-07-07')
    df = charging_data.loc[mask]
    daily_requests = df.groupby(['locationId', df['createdNew'].dt.date]).size()

    daily_requests = daily_requests.reset_index(name='request_count')
    print(daily_requests)
    daily_requests.to_csv('daily_requests.csv', index=False)
