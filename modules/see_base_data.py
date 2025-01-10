"""
Module: see_base_data.py

This module provides functionality for visualizing raw raster data and sensor
locations interactively. It includes tools for selecting and filtering raster
data layers and overlaying scatter plots of sensor points.
"""


import numpy as np
import pandas as pd
import plotly.express as px
from modules.load_data import load_zarr_data, load_sensor_points
import streamlit as st
import plotly.graph_objects as go
import xarray as xr


def zarr_menu(a: xr.Dataset) -> xr.DataArray | None:
    """
    Display a Streamlit form for selecting and visualizing raster data layers.

    Parameters
    ----------
    a : xr.Dataset
        The input xarray Dataset containing raster data variables.

    Returns
    -------
    xr.DataArray | None
        The selected xarray DataArray filtered by buffer, or None if no selection is made.
    """

    with st.form('select geo data'):

        selected_zarr = st.selectbox('Select Raster Data To Visualize', a.data_vars)
        buffer = st.selectbox('Select Buffer', np.unique(a.buffer))
        submit = st.form_submit_button(label='Submit')
    if submit:
        st.session_state['selected_zarr'] = {'layer': selected_zarr, 'buffer': buffer}
        return a[selected_zarr].sel(buffer=buffer)
    else:
        st.info('Select Raster data to get started')
        return None

def main_viz() -> None:
    """
    Load and visualize raw raster data alongside sensor locations.

    This function utilizes the `zarr_menu` function for raster data selection
    and overlays sensor points interactively.
    """

    data = load_zarr_data()
    sensors = load_sensor_points()
    zarr_menu(data)
    if st.session_state.get('selected_zarr'):
        st.info('Use the interactive tools to zoom in on the map !')
        map_raw_data_with_sensors(data, sensors, f'Map of {st.session_state.selected_zarr['layer']} at buffer {st.session_state.selected_zarr['buffer']}')


def map_raw_data_with_sensors(ds: xr.Dataset, sensors: pd.DataFrame, title: str = "Interactive Raster and Sensor Map") -> None:
    """
    Map raw raster data interactively and overlay sensor locations with hover information.

    Parameters
    ----------
    ds : xr.Dataset
        The raster data to be plotted, reshaped into a 2D grid.
    sensors : pd.DataFrame
        DataFrame containing sensor data. Must include 'X', 'Y', and additional columns
        for hover data.
    title : str, optional
        Title of the map. Default is "Interactive Raster and Sensor Map".

    Returns
    -------
    None
    """
    sensors = sensors[~ sensors.logger.isin([207, 240])]
    # Ensure required columns are in the sensors DataFrame
    if not {'X', 'Y'}.issubset(sensors.columns):
        raise ValueError("Sensors DataFrame must contain 'X' and 'Y' columns.")

    # Convert xarray DataArray to DataFrame for Plotly
    raster_df = ds[st.session_state.selected_zarr['layer']].sel(buffer=st.session_state.selected_zarr['buffer']).to_dataframe().reset_index()

    # Create a density heatmap
    fig = px.density_heatmap(
        raster_df,
        x="X",
        y="Y",
        z=st.session_state.selected_zarr['layer'],  # Use the name of the DataArray
        nbinsx=500,
        nbinsy=500,
        histfunc='avg',
        color_continuous_scale="Viridis",
        labels=st.session_state.selected_zarr['layer'],
        title=title
    )

    # Add sensors overlay
    fig.add_trace(
        go.Scatter(
            x=sensors["X"],
            y=sensors["Y"],
            mode="markers",
            marker=dict(size=12, color="red", symbol="circle"),
            name="Sensors",
            hovertext=sensors.apply(
                lambda row: f"X: {row['X']}, Y: {row['Y']}<br>" +
                            "<br>".join([f"{col}: {row[col]}" for col in sensors.columns if col not in ['X', 'Y']]),
                axis=1
            ),
            hoverinfo="text"
        )
    )

    fig.update_layout(
        font=dict(size=18),
        title=dict(font=dict(size=24)),
        xaxis=dict(title="X Coord (Swiss)", titlefont=dict(size=18)),
        yaxis=dict(title="Y Coord (Swiss)", titlefont=dict(size=18)),
        height=500,
        width= 1000
    )

    st.plotly_chart(fig)
