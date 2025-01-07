import xarray as xr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px
from modules.load_data import load_zarr_data, load_meteo, load_station_temp, load_sensor_points
import streamlit as st

def zarr_menu(a):
    with st.form('select geo data'):

        selected_zarr = st.selectbox('Select Raster Data To Visualize', a.data_vars)
        buffer = st.selectbox('Select Buffer', np.unique(a.buffer))
        overlay_scatter = st.checkbox('Overlay Scatter')
        submit = st.form_submit_button(label='Submit')
    if submit:
        st.session_state['selected_zarr'] = {'layer': selected_zarr, 'buffer': buffer, 'scatter': overlay_scatter}
        return a[selected_zarr].sel(buffer=buffer)
    else:
        st.info('Select a Raster data to get started')
        return None

def main_viz():
    data = load_zarr_data()
    sensors = load_sensor_points()
    zarr_menu(data)
    if st.session_state.get('selected_zarr'):
        map_raw_data_with_sensors(data, sensors, f'jk')

import plotly.graph_objects as go

def map_raw_data_with_sensors(ds, sensors, title="Interactive Raster and Sensor Map"):
    sensors = sensors[~ sensors.logger.isin([207, 240])]
    """
    Maps raw raster data interactively and overlays sensor locations with hover information.

    Parameters
    ----------
    raster_da : xr.DataArray
        The raster data to be plotted, reshaped into a 2D grid.
    sensors : pd.DataFrame
        DataFrame containing sensor data. Must have columns 'X', 'Y', and additional hover data columns.
    title : str, optional
        Title of the map. Default is "Interactive Raster and Sensor Map".

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The interactive Plotly map figure.
    """
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
        labels={st.session_state.selected_zarr['layer']: "Raster Data"},
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

    # Update layout for better visuals
    fig.update_layout(
        font=dict(size=18),  # Big font size
        title=dict(font=dict(size=24)),  # Larger title font size
        xaxis=dict(title="X Coordinate", titlefont=dict(size=20)),
        yaxis=dict(title="Y Coordinate", titlefont=dict(size=20)),
    )

    st.plotly_chart(fig)
