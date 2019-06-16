import gc
import pandas as pd

from funcs.geo_features import convert_geohash, get_xyz
from funcs.helper import add_mean_median
from funcs.split_data import reader
from funcs.time_features import make_day_feat, make_hour_minute_feat, make_time_feat, make_week_feat


def gen_based_features(test_path: str) -> pd.DataFrame:

    med_geo = reader("u_median_features/median_demand_geo.csv")
    med_time = reader("u_median_features/median_demand_time.csv")
    med_week = reader("u_median_features/median_demand_week.csv")

    u_geo = reader("u_median_features/u_demand_geo.csv")
    u_time = reader("u_median_features/u_demand_time.csv")
    u_week = reader("u_median_features/u_demand_week.csv")

    cons_zeros = reader("u_median_features/consec_zeros.csv")

    data_test = pd.read_csv(test_path)
    data_test['time'] = make_time_feat(data_test['timestamp'])

    data_test['day_mod'] = make_day_feat(data_test['day'])

    data_test['week_day'] = make_week_feat(data_test['day'])
    data_test['hour'], data_test['minute'] = make_hour_minute_feat(data_test['time'])

    data_test['time'] = data_test['time'].astype(str)

    data_test = pd.merge(data_test, cons_zeros, how='left', on=['day_mod', 'time', 'geohash6'])

    data_test = add_mean_median([data_test, u_time, med_time],
                                how='left', operate_on=['geohash6', 'week_day', 'hour'])
    data_test = add_mean_median([data_test, u_week, med_week],
                                how='left', operate_on=['geohash6', 'week_day'])
    data_test = add_mean_median([data_test, u_geo, med_geo],
                                how='left', operate_on=['geohash6', 'hour'])

    lat_long_series = convert_geohash(data_test['geohash6'])
    data_test['x_coord'] = lat_long_series.apply(lambda x: get_xyz(x[0], x[1])[0])
    data_test['y_coord'] = lat_long_series.apply(lambda x: get_xyz(x[0], x[1])[1])
    data_test['z_coord'] = lat_long_series.apply(lambda x: get_xyz(x[0], x[1])[2])

    return data_test


def gen_features_t_plus_one(df: pd.DataFrame) -> pd.DataFrame:

    df = df.sort_values(by=['geohash6', 'day', 'time'])
    df = df.reset_index(drop=True)

    lag_feat = df[['day', 'time', 'geohash6', 'y_pred']].copy()
    lag_feat = lag_feat.set_index(['day', 'time', 'geohash6'])
    lag_feat = lag_feat.unstack().shift(1)  # pull out the groups, shift with lag step=1
    lag_feat = lag_feat.stack(dropna=False)
    lag_feat = lag_feat.reset_index()
    lag_feat.rename(columns={'y_pred': 'lag_demand_1'}, inplace=True)

    tmp2 = pd.merge(df, lag_feat, how='inner', on=['day', 'time', 'geohash6'])

    tmp2['lag_demand_1'] = tmp2['lag_demand_1'].fillna(0)
    series_test_lag_diff = tmp2['lag_demand_1']
    del tmp2
    gc.collect()

    roll_feat = df[['day', 'time', 'geohash6', 'y_pred']].copy()
    roll_feat = roll_feat.set_index(['day', 'time', 'geohash6'])
    roll_feat = roll_feat.unstack().rolling(3).mean()  # pull out the groups, shift with lag step=1
    roll_feat = roll_feat.stack(dropna=False)
    roll_feat = roll_feat.reset_index()
    roll_feat.rename(columns={'y_pred': 'roll_demand'}, inplace=True)

    tmp3 = pd.merge(df, roll_feat, how='inner', on=['day', 'time', 'geohash6'])
    tmp3['roll_demand'] = tmp3['roll_demand'].fillna(0)
    series_test_rolling_diff = tmp3['roll_demand']

    df['u_diff_lag'] = series_test_lag_diff - df['mean_demand_per_hour']
    df['median_diff_lag'] = series_test_lag_diff - df['median_demand_per_hour']
    # ----------------------
    df['u_diff_lag_week'] = series_test_lag_diff - df['mean_demand_per_week']
    df['median_diff_lag_week'] = series_test_lag_diff - df['median_demand_per_week']
    # ---------------------
    df['u_diff_lag_geo'] = series_test_lag_diff - df['mean_demand_per_geo']
    df['median_diff_lag_geo'] = series_test_lag_diff - df['median_demand_per_geo']
    # ----------------------
    df['u_diff_rolling'] = series_test_rolling_diff - df['mean_demand_per_hour']
    df['u_diff_rolling_week'] = series_test_rolling_diff - df['mean_demand_per_week']
    df['u_diff_rolling_geo'] = series_test_rolling_diff - df['mean_demand_per_geo']

    del tmp3, roll_feat, lag_feat
    gc.collect()

    return df
