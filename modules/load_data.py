"""
Module: load_data.py

This module contains utility functions for loading and preprocessing data
used in urban heat modeling. The functions support loading meteorological
data, station data, Zarr datasets, and sensor points, and ensure compatibility
with the model's requirements.
"""


from pathlib import Path
import xarray as xr
import pandas as pd
import re


def load_meteo() -> xr.Dataset:
    """
    Load meteorological data from NetCDF files for minimum, mean, and maximum temperatures.

    The data is filtered for the 'BER' station and preprocessed for further analysis.

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing temperature data for 'min', 'mean', and 'max'.
    """

    das = {}
    for dtype in ['min', 'mean', 'max']:
        with xr.open_dataset(Path.cwd() / 'data' / 'meteo' / f'meteo_summer23_{dtype}.nc') as ds:
            data = ds.temperature.sel(stn='BER').load()
            data.name = f't_{dtype}'
            data = data.drop_vars('stn')
        das[f't_{dtype}'] = data
    ds = xr.Dataset(das)
    return ds

def load_station_temp() -> pd.DataFrame:
    """
    Load station temperature data and preprocess it for analysis.

    The preprocessing includes removing unwanted loggers, calculating tropical nights,
    and deriving urban heat island (UHI) metrics using a reference logger.

    Returns
    -------
    pd.DataFrame
        A DataFrame with processed temperature data, including UHI and tropical nights.
    """

    path = Path.cwd() / 'data' / 'station_data' / 'station_stats.csv'
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"]).dt.date
    df = df[~df.logger.isin([240, 239, 231])]
    df['tropical_nights'] = df.daily_min > 20.5
    ref_logger = df[df.logger == 206][["time", "daily_min"]].rename(columns={"daily_min": "ref_daily_min"})

    # Step 2: Merge reference logger data with the main dataframe
    df = df.merge(ref_logger, on="time", how="left")

    # Step 3: Calculate UHI as the difference
    df["uhi"] = df["daily_min"] - df["ref_daily_min"]
    return df

def load_zarr_data() -> xr.Dataset:
    """
    Load geospatial model data from a Zarr file.

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing loaded Zarr data.
    """

    path = Path.cwd() / 'data' / 'model_data.zarr'
    with xr.open_zarr(path) as zr:
        return zr.load()

def load_sensor_points() -> pd.DataFrame:
    """
    Load and preprocess sensor data from a CSV file.

    The function maps numerical column names to descriptive names based on
    predefined mappings and handles renaming for better clarity.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing preprocessed sensor data with descriptive column names.
    """

    # Load the sensor data CSV
    csv_path = Path.cwd() / 'data' / 'station_data' / 'sensor_data.csv'
    df = pd.read_csv(csv_path)

    # Mapping for renaming based on descriptions
    description_dict = {
        20: "Buildings",
        22: "Paved",
        14: "Water",
        7: "Rails",
        9: "Grass",
        15: "Vines",
        24: "Developed",
        25: "Forest",
        16: "Rock",
        27: "Barren"
    }
    s_dict = {str(k): v for k, v in description_dict.items()}

    # Prepare a rename dictionary
    rname_dict = {}
    for col in df.columns:
        match = re.search(r'(\d+)_\d+', col)  # Match patterns like `XX_YY`
        if match:
            old_name = match.group(1)  # Extract the first number (e.g., `20` from `20_50`)
            if old_name in s_dict:
                newname = s_dict[old_name]  # Map to description
                _, suffix = col.split('_')  # Extract the suffix (e.g., `50` from `20_50`)
                rname_dict[col] = f"{newname}_{suffix}"

    # Rename columns
    df = df.rename(columns=rname_dict)
    return df