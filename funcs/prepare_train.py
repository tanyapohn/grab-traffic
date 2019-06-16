from typing import List

import numpy as np
import pandas as pd


def fill_na(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill na of train set DataFrame corresponding to each day
    """
    train_df = train_df.set_index(['day', 'time', 'geohash6']).unstack()
    train_df = train_df.fillna(0)
    train_df = train_df.stack(dropna=False)
    train_df = train_df.reset_index().sort_values(by=['geohash6', 'day', 'time']).reset_index(drop=True)
    return train_df


def mean_or_median(train_make_feats_df: pd.DataFrame, index: List,
                   columns: List, value_name: str, func='mean') -> pd.DataFrame:
    """
    Create mean or median features of each hour, day of week, and geohash6

    :param train_make_feats_df: A DataFrame of training set contained features of day,
    hour, minute, day of week
    :param index: A list of indice
    :param columns: A list of columns
    :param value_name: Specify the name of output column
    :param func: A function to operate 'mean' or 'median'
    :return: A DataFrame
    """
    if func == 'mean':
        mean_hour_interval = pd.pivot_table(train_make_feats_df, index=index,
                                            columns=columns, values='demand',
                                            aggfunc=np.mean)

        mean_demand_hour = pd.melt(mean_hour_interval.reset_index(),
                                   id_vars=index,
                                   value_name=value_name)
        return mean_demand_hour

    elif func == 'median':
        median_hour_interval = pd.pivot_table(train_make_feats_df, index=index,
                                              columns=columns, values='demand',
                                              aggfunc=lambda x: np.median(x))

        median_demand_hour = pd.melt(median_hour_interval.reset_index(),
                                     id_vars=index,
                                     value_name=value_name)
        return median_demand_hour

    else:
        raise ValueError
