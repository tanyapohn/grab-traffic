from functools import reduce
from typing import List

import numpy as np
import pandas as pd


def add_mean_median(df_list: List, how: str, operate_on: List) -> pd.DataFrame:
    """
    Join mean and median DataFrame to training set DataFrame
    :param df_list: A list of DataFrames: 'training_df, mean_df, median_df'
    :param how: 'inner', 'left', 'right'
    :param operate_on: A list of columns to operate on
    :return: Output DataFrame containing both mean and median
    """
    output_df = reduce(lambda left, right: pd.merge(left, right, how=how, on=operate_on), df_list)

    return output_df


def count_consec_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count consecutive demand values of 0s
    :param df: A DataFrame that is sorted by 'geohash6', 'day', 'time'
    :return: An increment number of consecutive zeros
    """

    g = df['demand'].ne(df['demand'].shift(1)).cumsum()
    counter = df.groupby(['geohash6', 'day', g])['demand'].cumcount() + 1
    df['consec_zeros'] = np.where(df['demand'].eq(0), counter, 0)

    masker = df['consec_zeros'].mask(df['consec_zeros'].eq(0))

    df['consec_zeros'] = (np.where(masker.isna(),
                                   masker.groupby([df['geohash6'], df['day']]).ffill(limit=1) + 1,
                                   df['consec_zeros']) - 1)
    df['consec_zeros'] = df['consec_zeros'].fillna(0).astype(float)

    return df
