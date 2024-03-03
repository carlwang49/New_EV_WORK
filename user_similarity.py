import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

EMB_DIM = 210
MISP_USER_PATH = f"./Result/DIM_{EMB_DIM}/user_emb.csv"
MISP_SIMILARITY_PATH = f"./Result/DIM_{EMB_DIM}/user_similarity.csv"

# MF_USER_PATH = "./Result/Baseline/MF/0721_new_relation/user_emb.csv"
# MF_SIMILARITY_PATH = "./Result/Baseline/MF/0721_new_relation/user_similarity.csv"


### calculate user similarity
user_emb = pd.read_csv(MISP_USER_PATH, sep="\t", header=None)
user_similarity = np.zeros((len(user_emb), len(user_emb)))

for i in range(len(user_emb)):
    for j in range(len(user_emb)):
        user_similarity[i][j] = cosine_similarity(user_emb.loc[i].to_numpy().reshape(1, -1), user_emb.loc[j].to_numpy().reshape(1, -1)) + 1

np.savetxt(MISP_SIMILARITY_PATH, user_similarity, delimiter="\t")