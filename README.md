# An Incentive-based Scheduling Algorithm for Balancing Multiple Charging Stations (平衡多充電站間的充電排程獎勵演算法)

## File Structure
```
┣━ CaseRecommender
┣━ torchkge
┣━ Matrix-Factorization-Recommender-Systems-Netflix-Paper-Implementation
┣━ Dataset
┣━ Result
┃ ┣━ Baseline
┃ ┃ ┣━ ISP
┃ ┃ ┗━ MF
┃ ┃   ┣━ 0721/alpha_0.5/..
┃ ┃   ┗━ 0721_new_relation
┃ ┃     ┣━ Relation
┃ ┃     ┃ ┗━ {user_id}.csv
┃ ┃     ┣━ ent_emb.csv
┃ ┃     ┣━ user_emb.csv
┃ ┃     ┗━ ev_cs_preference.csv
┃ ┗━ MISP
┃   ┣━ 0719_expected_score_6
┃   ┃ ┗━ alpha_{num}
┃   ┃   ┗━ {testing_date}.csv
┃   ┣━ Relation
┃   ┃ ┗━ {user_id}.csv
┃   ┣━ ent_emb.csv
┃   ┣━ user_emb.csv
┃   ┣━ rel_emb.csv
┃   ┣━ ev_cs_preference.csv
┃   ┣━ ev_cs_preference_correction.csv
┃   ┗━ user_similarity.csv 
┣━ Framework
┃ ┣━ base.py
┃ ┣━ ISP.py
┃ ┣━ MISMF.py
┃ ┗━ MISP.py
┣━ preprocessing.py
┣━ model.py
┣━ postprocessing.py
┣━ user_similarity.py
┣━ null_value_prediction.py
┣━ null_value_prediciton_fix.py
┣━ MF_user_relation.py
┗━ evaluation.ipynb
```

## Code Flow
### Preprocessing
* `preprocessing.py`: 產生 KGE 訓練資料，包含 relationid.txt, entity.txt, train.txt
* `model.py`: BPRMF + KGE 模型，產生四份資料 entity embedding (CS embedding), user embedding (EV embedding), relation embedding, ev_cs_preference (每個使用者對不同充電站的偏好值)
* `postprocessing.py`: 平移偏好值，避免出現負值
* `user_similarity.py`: 用 EV embedding 算用戶相似度
* `null_value_prediciton.py`: MISP 補齊時段缺失值 (這是舊的方法)
* `null_value_prediction_fix.py`: MISP 補齊時段缺失值 (這是投影片上新的方法，可是結果好像比較爛)
* `MF_user_relation.py`: MF 補齊使用者對每個充電站的偏好值

### Main Code
* `base.py`: 所有方法共用的一些 function
* `MISP.py`: 論文內的演算法
* `ISP.py`: baseline，只針對各別充電站推薦
* `MISMF.py`: baseline，偏好值用 MF 預測

### Evaluation
* `evaluation.ipynb`: 跑其他基本的 baseline 以及計算各種 metric 結果 (用 [colab](https://colab.research.google.com/drive/1CpYfAaqI51j1Jgs50z_QmXu2yu59kAzX?usp=sharing) 開比較好)

## Document
大部分說明都在程式中，有一些額外資料集紀錄在 [Notion](https://www.notion.so/0908-766b18d7d2c34a89b7759ea5efb98894)
>>>>>>> a3f82da (initial commit)
# New_EV_WORK
