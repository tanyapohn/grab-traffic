import os

from funcs.geo_features import convert_geohash, get_xyz
from funcs.helper import add_mean_median, count_consec_zeros
from funcs.prepare_train import fill_na, mean_or_median
from funcs.split_data import splitter
from funcs.time_features import make_day_feat, make_hour_minute_feat, make_time_feat, make_week_feat

this_dir = os.path.abspath("")
data_path = os.path.join(this_dir, "data")
gen_feats_path = os.path.join(data_path, "gen_features/checkpoint_1")
u_median_feats_path = os.path.join(data_path, "u_median_features")


data_train, data_val = splitter("training.csv", 46, val=False)

data_train['time'] = make_time_feat(data_train['timestamp'])
data_train.drop(['timestamp'], axis=1, inplace=True)

data_train = fill_na(data_train)
data_train['day_mod'] = make_day_feat(data_train['day'])
data_train = count_consec_zeros(data_train)
data_train['week_day'] = make_week_feat(data_train['day'])
data_train['hour'], data_train['minute'] = make_hour_minute_feat(data_train['time'])

print('Saving consecutive features ... \n')

# -------------------------------- #
# Save consecutive zeros features
# -------------------------------- #

zeros_feats = data_train[['day_mod', 'time', 'geohash6', 'consec_zeros']].copy()
zeros_feats.to_csv(os.path.join(u_median_feats_path, "consec_zeros.csv"), index=False)

print('Generating mean and median features ... \n')

# -------------------------------- #
# Generate mean and median features
# -------------------------------- #

mean_demand_by_time = mean_or_median(data_train, index=['geohash6', 'week_day'],
                                     columns=['hour'], value_name= 'mean_demand_per_hour',
                                     func='mean')

mean_demand_by_week = mean_or_median(data_train, index=['geohash6'], columns=['week_day'],
                                     value_name='mean_demand_per_week', func='mean')

mean_demand_by_geo = mean_or_median(data_train, index=['geohash6'], columns=['hour'],
                                    value_name='mean_demand_per_geo', func='mean')

median_demand_by_time = mean_or_median(data_train, index=['geohash6', 'week_day'],
                                       columns=['hour'], value_name= 'median_demand_per_hour',
                                       func='median')

median_demand_by_week = mean_or_median(data_train, index=['geohash6'], columns=['week_day'],
                                       value_name='median_demand_per_week', func='median')

median_demand_by_geo = mean_or_median(data_train, index=['geohash6'], columns=['hour'],
                                      value_name='median_demand_per_geo', func='median')

print('Joining mean and median to training set ... \n')

# ------------------------- #
# Join them to training data
# ------------------------- #

data_train = add_mean_median([data_train, mean_demand_by_time, median_demand_by_time],
                             how='inner', operate_on=['geohash6', 'week_day', 'hour'])
data_train = add_mean_median([data_train, mean_demand_by_week, median_demand_by_week],
                             how='inner', operate_on=['geohash6', 'week_day'])
data_train = add_mean_median([data_train, mean_demand_by_geo, median_demand_by_geo],
                             how='inner', operate_on=['geohash6', 'hour'])

print('Converting latitude-longitude pairs ... \n')

# ------------------------------------ #
# Convert latitude-longitude to x, y, z
# ------------------------------------ #

lat_long_series = convert_geohash(data_train['geohash6'])
data_train['x_coord'] = lat_long_series.apply(lambda x: get_xyz(x[0], x[1])[0])
data_train['y_coord'] = lat_long_series.apply(lambda x: get_xyz(x[0], x[1])[1])
data_train['z_coord'] = lat_long_series.apply(lambda x: get_xyz(x[0], x[1])[2])


# ----------------------- #
# Save it to checkpoint_1 #
# ----------------------- #

print('============= saving to checkpoint 1 ================= \n')
data_train.to_csv(os.path.join(gen_feats_path, "train_feats.csv"), index=False)
data_val.to_csv(os.path.join(gen_feats_path, "val.csv"), index=False)


# ------------------------ #
# Save features to a folder
# ------------------------ #

print('============= saving features =================== \n')
mean_demand_by_time.to_csv(os.path.join(u_median_feats_path, "u_demand_time.csv"), index=False)
mean_demand_by_week.to_csv(os.path.join(u_median_feats_path, "u_demand_week.csv"), index=False)
mean_demand_by_geo.to_csv(os.path.join(u_median_feats_path, "u_demand_geo.csv"), index=False)

median_demand_by_time.to_csv(os.path.join(u_median_feats_path, "median_demand_time.csv"), index=False)
median_demand_by_week.to_csv(os.path.join(u_median_feats_path, "median_demand_week.csv"), index=False)
median_demand_by_geo.to_csv(os.path.join(u_median_feats_path, "median_demand_geo.csv"), index=False)

print('============= FINISHED =================== \n')
