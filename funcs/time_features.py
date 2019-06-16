import pandas as pd
from typing import Tuple


def make_day_feat(input_day: pd.Series) -> pd.Series:
    """
    Convert ordered day integer like to be in [1,31] range
    and day to day of week
    """
    def mod_day(int_day: int) -> float:
        mod_result = int_day % 31.0
        if mod_result == 0.0:
            return 31.0
        else:
            return mod_result

    day_feat = input_day.apply(lambda x: mod_day(x))

    return day_feat


def make_time_feat(input_timestamp: pd.Series) -> pd.Series:
    """
    Convert timestamp to be in datetime format 'hour:min'
    """
    time_feat = pd.to_datetime(input_timestamp, format='%H:%M').dt.time

    return time_feat


def make_hour_minute_feat(time_feat: pd.Series) -> Tuple:
    """
    Extract time in 'hour:min' format to 'hour' and 'min' (float like)
    """
    hour_feat = time_feat.apply(lambda x: x.hour)
    minute_feat = time_feat.apply(lambda x: x.minute)

    hour_feat = hour_feat.astype(float)
    minute_feat = minute_feat.astype(float)

    return hour_feat, minute_feat


def make_week_feat(input_day: pd.Series) -> pd.Series:
    week_feat = input_day.apply(lambda x: x % 7.0)
    return week_feat
