from pathlib import Path
import xarray as xr
import pandas as pd
import re
import streamlit as st


def load_meteo():
    das = {}
    for dtype in ['min', 'mean', 'max']:
        with xr.open_dataset(Path.cwd() / 'data' / 'meteo' / f'meteo_summer23_{dtype}.nc') as ds:
            data = ds.temperature.sel(stn='BER').load()
            data.name = f't_{dtype}'
            data = data.drop_vars('stn')
        das[f't_{dtype}'] = data
    ds = xr.Dataset(das)
    return ds

def load_station_temp():
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

def load_zarr_data():
    path = Path.cwd() / 'data' / 'model_data.zarr'
    with xr.open_zarr(path) as zr:
        return zr.load()

def load_sensor_points():
    """
    Loads sensor data from a CSV file, renames columns based on a description
    dictionary, and returns the updated DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns for sensor data.
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