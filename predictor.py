import argparse
import os

import gc
import pandas as pd
import xgboost as xgb

from gen_feats import gen_based_features, gen_features_t_plus_one


this_dir = os.path.abspath("")
model_1_path = os.path.join(this_dir, "model/xgb_t_plus_1.model")
model_2_path = os.path.join(this_dir, "model/xgb_t_plus_1_2.model")
result_path = os.path.join(this_dir, "results")


BASED_COLS = ['day_mod', 'consec_zeros', 'week_day', 'hour', 'minute',
              'mean_demand_per_hour', 'median_demand_per_hour',
              'mean_demand_per_week', 'median_demand_per_week',
              'mean_demand_per_geo', 'median_demand_per_geo', 'x_coord',
              'y_coord', 'z_coord']

DIFF_COLS = ['day_mod', 'consec_zeros', 'week_day', 'hour', 'minute',
             'mean_demand_per_hour', 'median_demand_per_hour',
             'mean_demand_per_week', 'median_demand_per_week',
             'mean_demand_per_geo', 'median_demand_per_geo', 'x_coord',
             'y_coord', 'z_coord', 'u_diff_lag', 'median_diff_lag',
             'u_diff_lag_week', 'median_diff_lag_week', 'u_diff_lag_geo',
             'median_diff_lag_geo', 'u_diff_rolling', 'u_diff_rolling_week',
             'u_diff_rolling_geo']


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Predict demand \n the results will be stored under ./results dir')
    parser.add_argument('--path', default='s32', help='File path of a test set')
    input_arg = parser.parse_args()

    print('\nStart generating features ... \n')
    data_test = gen_based_features(input_arg.path)

    print('\nStart predicting base demand ... \n')
    # Load model
    bst = xgb.Booster()
    bst.load_model(model_1_path)

    X_test = data_test[BASED_COLS]
    matrix_test = xgb.DMatrix(X_test)

    demand_pred = bst.predict(matrix_test, ntree_limit=int(bst.attributes()['best_iteration']))
    # create base demand prediction
    demand_pred_series = pd.Series(demand_pred, name='y_pred')

    del bst
    gc.collect()

    bst_2 = xgb.Booster()
    bst_2.load_model(model_2_path)

    print('\nStart predicting t plus 1 to 5 ... \n')

    df_res = pd.concat([data_test, demand_pred_series], axis=1)

    for t in range(1, 6):

        print('\nPredicting T + ' + str(t) + ' ...\n')

        df_t = gen_features_t_plus_one(df_res)

        del df_res

        df_t_test = df_t[DIFF_COLS]
        matrix_test_t_plus = xgb.DMatrix(df_t_test)
        demand_pred_t_plus = bst_2.predict(matrix_test_t_plus, ntree_limit=int(bst_2.attributes()['best_iteration']))

        demand_pred_series = pd.Series(demand_pred_t_plus, name='y_pred')

        if t == 1:
            result = pd.concat([df_t[['geohash6', 'day', 'timestamp']], abs(demand_pred_series)], axis=1)
            result.to_csv(os.path.join(result_path, "t_plus_one.csv"), index=False)
        elif t == 5:
            result = pd.concat([df_t[['geohash6', 'day', 'timestamp']], abs(demand_pred_series)], axis=1)
            result.to_csv(os.path.join(result_path, "t_plus_five.csv"), index=False)

        df_res = pd.concat([df_t.drop(['y_pred', 'u_diff_lag', 'median_diff_lag', 'u_diff_lag_week',
                                       'median_diff_lag_week', 'u_diff_lag_geo',
                                       'median_diff_lag_geo', 'u_diff_rolling',
                                       'u_diff_rolling_week', 'u_diff_rolling_geo'], axis=1),
                            demand_pred_series], axis=1)

        del df_t, df_t_test
        gc.collect()

    del bst_2
    gc.collect()

    print('\n--------- FINISHED ---------- \n')
    print('The results have been saved in results directory')