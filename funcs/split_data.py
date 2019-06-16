import os

import pandas as pd

work_dir = os.path.abspath("")
data_path = os.path.join(work_dir, "data")


def reader(filename: str) -> pd.DataFrame:
    raw_data_path = os.path.join(data_path, filename)
    data = pd.read_csv(raw_data_path)
    return data


def splitter(filename: str, day_split: int, val=True):

    data = reader(filename)
    data_train = data.loc[(14 < data.day) & (data.day <= day_split)].copy()
    data_val = data.loc[day_split < data.day].copy()
    if val:
        return data_train, data_val

    else:
        return data_train
