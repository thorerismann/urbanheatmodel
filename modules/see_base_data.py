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

def plot_zar_data(ds):
    fig, ax = plt.subplots(figsize=(8, 6))

    ds[st.session_state.selected_zarr['layer']].sel(buffer=st.session_state.selected_zarr['buffer']).plot(ax=ax, cmap="viridis", cbar_kwargs={"label": st.session_state.selected_zarr['layer']}, x='X', y='Y')

    # Add plot details
    ax.set_title(f" layer {st.session_state.selected_zarr['layer']} at buffer: {st.session_state.selected_zarr['buffer']}")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.legend()

    # Adjust layout and render
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def main_viz():
    data = load_zarr_data()
    zarr_menu(data)
    if st.session_state.get('selected_zarr'):
        plot_zar_data(data)