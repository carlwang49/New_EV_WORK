import pandas as pd

satisfaction_df = pd.read_csv("./Result/satisfaction_score/s15_only_accept.csv")

satisfaction_df['faviorate_ratio'] = satisfaction_df['faviorate_ratio'].str.rstrip('%').astype('float') / 100.0

min_distance = 0.010980711346939084
max_distance = 9.949851106124495

print(satisfaction_df.columns)


satisfaction_df['satisfaction_score'] = \
    (satisfaction_df['faviorate_ratio'] + 
     (1 - ((satisfaction_df['MovingTime'] - 0)/(10 - 0))) + 
     (1 - ((satisfaction_df['MovingDistance'] - min_distance) / (max_distance - min_distance)))) / 3

# satisfaction_score_2 = (favor_ratio + (1 - ((average_time - 0)/(10-0))) + (1 - ((average_distance - min_distance) / (max_distance - min_distance)))) / 3
print(satisfaction_df)
satisfaction_df.to_csv("./Result/satisfaction_score/s15_with_satisfaction_score_2.csv", index=False)
