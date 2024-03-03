from Evaluaiton import Evaluation
from datetime import timedelta
from collections import namedtuple, defaultdict
import pandas as pd
import time
from dateutil import parser

eval = Evaluation()

columns = [
    "datetime",          # charging time, ex: 2018-07-01 00:00:00
    "userID",            # userID 
    "locationID",        # charging station ID
    "buildingID",        # building ID
    "chargingLen",       # charging duration
    "originChargingHour" # original request charging time 
]

unschedule_df = pd.DataFrame([], columns=columns)

testing_start_time = eval.charging_start_time
building_end_time = eval.charging_end_time

Request = namedtuple('Request', ['id', 'userId', 'chargingHour', 'originChargingHour', 'locationId'])

for day in range(7):

    charging_requests = eval.get_charging_request(testing_start_time) # requests in a day

    for request in charging_requests:
        
        request = Request(*request)
        buildingID = eval.location.loc[eval.location.index == request.locationId, "buildingID"].values[0]
        
        unschedule_df.loc[len(unschedule_df)] = [testing_start_time + timedelta(hours=request[3]),
                                                 request.userId,
                                                 request.locationId,
                                                 buildingID,
                                                 request.chargingHour,
                                                 request.originChargingHour]

    testing_start_time += timedelta(days=1)

# unschedule_df = unschedule_df[unschedule_df["userID"] != "603475"]
# unschedule_df.loc[433, "datetime"] = parser.parse("2018-07-06 17:00:00")
# unschedule_df.loc[471, "datetime"] = parser.parse("2018-07-06 17:00:00")
# unschedule_df


overage = 0
electricity_price = pd.DataFrame([], columns=["basic_tariff", "current_tariff", "overload_penalty", "total"])
unschedule_statistic_count = pd.DataFrame([], columns=[eval.charging_start_time + timedelta(hours=hour) for hour in range(0, 7*24)])

overload_percentage = list()

unschedule_revenue = defaultdict()
unschedule_ev_charging_volume = list()
unschedule_electricity_cost = defaultdict() 
unschedule_ev_revenue = defaultdict()

for cs in eval.location.index:

    buildingID = eval.building_data["buildingID"] == eval.location.loc[cs, "buildingID"]
    start_time = eval.building_data["datetime"] >= eval.building_start_time
    end_time = eval.building_data["datetime"] < building_end_time
    
    info = eval.building_data[buildingID & start_time & end_time].copy()

    ### EV charging data ###
    ev_info = [0 for i in range(24*7)]
    current = eval.charging_start_time
    
    for day in range(7):
        for hour in range(24):
            # hour: current hour
            charging_value = len(unschedule_df[(unschedule_df["locationID"] == cs) &
                                            (unschedule_df["datetime"] >= current) &
                                            (unschedule_df["datetime"] < (current + timedelta(days=1))) &
                                            (unschedule_df["datetime"].dt.hour <= hour) &
                                            ((unschedule_df['datetime'].dt.hour + unschedule_df["chargingLen"]) > hour)]) * eval.charging_speed
            
    
            ev_info[(day * 24) + hour] = charging_value
        current += timedelta(days=1)
    
    unschedule_ev_charging_volume.append(ev_info)
    unschedule_ev_revenue[cs] = sum(ev_info) * eval.charging_fee
    
    info["charging"] = ev_info
    info["total"] = info["consumption"] - info["generation"] + info["charging"]


    ### check number of exceed ###
    info["exceed"] = info["total"] - eval.location.loc[cs, "contractCapacity"]
    info["exceed"] = info["exceed"].apply(lambda x: 0 if x < 0 else x)
   
    overload_slots = 0
    for raw in info["exceed"]:
        if raw != 0:
            overload_slots += 1

    overload_percentage.append(overload_slots / (7 * 24))
    print(int(eval.location.loc[cs, 'buildingID']), ":")
    print(f"overload = {'{:.2f}'.format(info['exceed'].sum())}")
    print(f"overload_percentage = {'{:.4f}'.format(overload_percentage[-1])}")
    
    overage += info["exceed"].sum()

    basic_tariff, current_tariff, overload_penalty = eval.calculate_electricity_price(eval.location, cs, info)
    total_price = basic_tariff + current_tariff + overload_penalty

    electricity_price.loc[len(electricity_price)] = [basic_tariff, current_tariff, overload_penalty, total_price]
    unschedule_revenue[cs] = (-1) * total_price + (eval.charging_fee * sum(ev_info)) # profit
    unschedule_electricity_cost[cs] = total_price
    print(f"revenue = {'{:.2f}'.format(unschedule_revenue[cs])}")

    info["chargingCount"] = info["charging"].apply(lambda x: x/eval.charging_speed)
    info.set_index("datetime", inplace=True)
    unschedule_statistic_count.loc[len(unschedule_statistic_count)] = info["chargingCount"].T


print("==========================")

print(f"average revenue = {int(sum(unschedule_revenue.values()) / 20)}")
print(electricity_price.mean(axis=0))