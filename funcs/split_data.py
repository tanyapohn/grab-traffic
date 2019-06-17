import os

import pandas as pd

work_dir = os.path.abspath("")
data_path = os.path.join(work_dir, "data")


def reader(filename: str) -> pd.DataFrame:
    """
    Easily to read any files within root dir
    """
    raw_data_path = os.path.join(data_path, filename)
    data = pd.read_csv(raw_data_path)
    return data


def splitter(filename: str, day_split: int, val=True):
    """
    To generate features from training set, I decide to use demand value from day 15 to 46
    """
    data = reader(filename)
    data_train = data.loc[(14 < data.day) & (data.day <= day_split)].copy()
    data_val = data.loc[day_split < data.day].copy()
    if val:
        return data_train, data_val

    else:
        return data_train
