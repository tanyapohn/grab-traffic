import os

from funcs.split_data import reader


this_dir = os.path.abspath("")
data_path = os.path.join(this_dir, "data")
gen_feats_path = os.path.join(data_path, "gen_features/checkpoint_2")

data_train = reader("gen_features/checkpoint_1/train_feats.csv")

data_train = data_train.sort_values(by=['geohash6', 'day', 'time'])
data_train = data_train.reset_index(drop=True)

tmp = data_train[['day', 'time', 'geohash6', 'demand']].copy()
tmp = tmp.groupby('geohash6')['demand']

series_train_lag_diff = tmp.transform(lambda x: x.shift(1))
series_train_lag_diff = series_train_lag_diff.fillna(0)

series_train_rolling_diff = tmp.transform(lambda x: x.rolling(3).mean())
series_train_rolling_diff = series_train_rolling_diff.fillna(0)

data_train['u_diff_lag'] = series_train_lag_diff - data_train['mean_demand_per_hour']
data_train['median_diff_lag'] = series_train_lag_diff - data_train['median_demand_per_hour']
# ---------------------
data_train['u_diff_lag_week'] = series_train_lag_diff - data_train['mean_demand_per_week']
data_train['median_diff_lag_week'] = series_train_lag_diff - data_train['median_demand_per_week']
# ---------------------
data_train['u_diff_lag_geo'] = series_train_lag_diff - data_train['mean_demand_per_geo']
data_train['median_diff_lag_geo'] = series_train_lag_diff - data_train['median_demand_per_geo']
# --------------------
data_train['u_diff_rolling'] = series_train_rolling_diff - data_train['mean_demand_per_hour']
data_train['u_diff_rolling_week'] = series_train_rolling_diff - data_train['mean_demand_per_week']
data_train['u_diff_rolling_geo'] = series_train_rolling_diff - data_train['mean_demand_per_geo']


# ----------------------------- #
# Save features to checkpoint_2 #
# ----------------------------- #

print('============= saving to checkpoint 2 =================')
data_train.to_csv(os.path.join(gen_feats_path, "train_feats_diff.csv"), index=False)
print('====================== FINISHED ======================')