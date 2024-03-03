import pandas as pd
import numpy as np

from Recommender.matrix_factor_model import ProductRecommender

def read_data():
    # df = pd.read_csv("../Dataset/BPRMF_train_5.csv", names=["user", "cs", "like"], index_col=["user", "cs"])
    # df = df.unstack(fill_value=0)
    df = pd.read_csv("../Dataset/MF_train_5.csv", names=["user", "cs", "hour", "like"], index_col=["user", "cs", "hour"])
    train_data_list = list()
    for user in range(136):
        temp = list()
        for cs in range(20):
            for hour in range(24):
                if (user, cs, hour) in df.index.tolist():
                    temp.append(df.loc[(user, cs, hour)].values[0])
                else:
                    temp.append(0)
        train_data_list.append(temp)
    # return df.values.tolist()
    return train_data_list

# fake data. (2 users, three products)
# user_1 = [1, 2, 3, 0]
# user_2 = [0, 2, 3, 4]
# data = [user_1, user_2]
data = read_data()

# train model
model = ProductRecommender()
model.fit(data)

# predict
result = pd.DataFrame([], columns=[i for i in range(len(data[0]))])
for user in range(len(data)):
    result.loc[len(result)] = model.predict_instance(user)

result = result.stack().sort_values(ascending=False)
print(result)
result.to_csv("../Result/Baseline/MF/0721_new_train/ev_cs_preference.csv", sep="\t", header=None)

user_emb, ent_emb = model.get_models()
np.savetxt("../Result/Baseline/MF/0721_new_train/user_emb.csv", user_emb, delimiter="\t")
np.savetxt("../Result/Baseline/MF/0721_new_train/ent_emb.csv", ent_emb, delimiter="\t")
