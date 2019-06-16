import math
from typing import Tuple
import geohash
import pandas as pd
import numpy as np


def convert_geohash(geo: pd.Series) -> pd.Series:
    """
    Convert geohash6 field to be in latitude-longitude pair
    :param geo: Geohash6 field
    :return: Latitude-longitude pair wise
    """
    lat_long = geo.apply(lambda x: np.round(geohash.decode(x)[0:2], 8))
    return lat_long


def get_xyz(lat: float, lon: float) -> Tuple:
    """
    Convert latitude and longitude to be in cartesian x, y, z coordinates
    :param lat: Latitude (degree)
    :param lon: Longitude (degree)
    :return: A tuple of floats indicating x, y, z coordinates
    """
    cos_lat = math.cos(lat * math.pi / 180.0)
    sin_lat = math.sin(lat * math.pi / 180.0)
    cos_lon = math.cos(lon * math.pi / 180.0)
    sin_lon = math.sin(lon * math.pi / 180.0)
    rad = 500.0
    x = rad * cos_lat * cos_lon
    y = rad * cos_lat * sin_lon
    z = rad * sin_lat
    return x, y, z
