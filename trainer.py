import xgboost as xgb

from funcs.split_data import reader

data_train = reader("gen_features/checkpoint_2/train_feats_diff.csv")

X_train = data_train.drop(['geohash6', 'time', 'demand', 'day'], axis=1)
y_train = data_train['demand']

xgb_reg = xgb.XGBRegressor(n_estimators=200, learning_rate=0.08, objective='reg:squarederror', booster='dart')
xgb_reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train)],
            eval_metric='rmse', early_stopping_rounds=10)
