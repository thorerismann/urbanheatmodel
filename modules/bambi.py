from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px
from modules.load_data import load_zarr_data, load_station_temp, load_sensor_points
import streamlit as st
import bambi as bmb
import arviz as az
import xarray as xr


def load_base_data():
    df = load_station_temp()
    s = load_sensor_points()
    z = load_zarr_data()
    return df, s, z

def display_bambi_menu():
    with st.form('variable_menu'):
        st.markdown('##### Specify model')
        # Available buffers and geospatial variables
        buffers_list = [50, 250]
        numvars = 5
        max_interactions=2
        max_polynomials = 3
        geo_vars = [
            "Water", "Buildings", "Paved", "Developed", "Forest",
            "Grass", "fitnah_ss", 'fitnah_sv', 'fitnah_temp', 'dem'
        ]
        var_buff_combinations = [f"{var}_{buf}" for var in geo_vars for buf in buffers_list]
        response = st.selectbox(
            'Select Response Variable:',
            ['tropical_nights', 'uhi']
        )
        variables = st.multiselect('Select variables', var_buff_combinations, max_selections=numvars)
        interactions = [st.multiselect(f'Select terms for interaction term {x+1}', var_buff_combinations, max_selections=2, key = f'interaction_{x}') for x in range(0, max_interactions+1)]
        polynomials = st.multiselect('Select polynomials', var_buff_combinations, max_selections=max_polynomials)
        st.divider()
        st.markdown('##### Specify model parameters')

        chains = st.number_input(
            'Select the number of Chains',
            min_value=2, max_value=8, value=2
        )
        draws = st.number_input(
            'Select the number of Draws',
            min_value=500, max_value=5000, value=1000
        )
        family = st.selectbox(
            'Select Distribution',
            ['t', 'negativebinomial', 'poisson', 'gaussian']
        )
        st.divider()

        st.markdown('##### Priors')
        c1, c2 = st.columns(2)
        with c1:
            use_priors = st.toggle('Use priors')
            sigma = st.number_input('Sigma', value=1.0, min_value=0.25)
        with c2:
            heavy_tails = st.checkbox('Fat tails', False)
            mu = st.number_input('Mu', value=0.0, min_value=-100.0)
        submit_model = st.form_submit_button('Submit Model')

    if submit_model:
        all_vars = variables + polynomials + [item for sublist in interactions for item in sublist]
        all_vars = list(set(all_vars))
        st.session_state['bambi_selection']  = {'dep_var': response,
                                                'vars': variables,
                                                'family': family,
                                                'chains': chains,
                                                'draws': draws,
                                                'use_priors': use_priors,
                                                'sigma': sigma,
                                                'heavy_tails': heavy_tails,
                                                'mu': mu,
                                                'interactions': interactions,
                                                'polynomials': polynomials,
                                                'all_vars':all_vars
                                                }
        st.write('Expand below to see the selected parameters.')
        st.json(st.session_state['bambi_selection'], expanded=False)

def create_priors(meta):
    """
    Creates priors for variables based on the model's family.

    Parameters:
        var_list (list): List of predictor variable names.
        meta (dict): Dictionary with metadata, including the family.
        sigma (float): Standard deviation for the normal prior (default: 0.5).
        mu (float): Mean for the normal prior (default: 0).
        heavy_tails (bool): Use priors with heavier tails for Gaussian/Student's t (default: False).

    Returns:
        dict: A dictionary of priors for the specified variables.
    """
    priors = {}
    family = st.session_state.bambi_selection['family']
    sigma = st.session_state.bambi_selection['sigma']
    mu = st.session_state.bambi_selection['mu']
    heavy_tails = st.session_state.bambi_selection['heavy_tails']
    var_list = st.session_state.bambi_selection['vars']

    if family in ['poisson', 'negativebinomial']:
        # Normal priors for count models
        for var in var_list:
            priors[var] = bmb.Prior("Normal", mu=mu, sigma=sigma)

    elif family == 'gaussian':
        # Priors for Gaussian models
        for var in var_list:
            if heavy_tails:
                # Heavy-tailed Student's t prior
                priors[var] = bmb.Prior("StudentT", mu=mu, sigma=sigma, nu=3)
            else:
                # Standard normal prior
                priors[var] = bmb.Prior("Normal", mu=mu, sigma=sigma, nu=3)

    elif family == 't':
        # Priors for Student's t-distribution models
        for var in var_list:
            priors[var] = bmb.Prior("StudentT", mu=mu, sigma=sigma, nu=3)

    return priors

def prepare_uhi_data(station_data, sensors, excluded_loggers):
    """
    """
    dep = station_data.groupby('logger')['uhi'].mean()
    data = sensors[['logger'] + st.session_state.bambi_selection['all_vars']]
    full = data.merge(pd.DataFrame(dep).reset_index(), on='logger')
    full = full[~full.logger.isin(excluded_loggers)]

    total_na = full.isna().sum()
    if total_na.sum() > 0:
        st.write('total na values', total_na)
        full = full.dropna(how='any')
    full = full.sort_values(by=['logger'])
    return full


def prepare_tn_data(station_data, sensors, excluded_loggers):
    """
    """
    dep = station_data.groupby('logger')['tropical_nights'].sum()
    data = sensors[['logger'] + st.session_state.bambi_selection['all_vars']]
    full = data.merge(pd.DataFrame(dep).reset_index(), on='logger')
    full = full[~full.logger.isin(excluded_loggers)]

    total_na = full.isna().sum()
    if total_na.sum() > 0:
        st.write('total na values', total_na)
        full = full.dropna(how='any')
    full = full.sort_values(by=['logger'])

    return full


@st.cache_resource
def create_bambi_model(data, _meta):
    y = _meta['dep_var']
    draws = _meta['draws']
    chains = _meta['chains']
    family = _meta['family']
    if _meta['use_priors']:
        _meta['priors'] = create_priors(_meta)
    else:
        _meta['priors'] = None

    predictors = " + ".join(_meta['vars'])
    formula = f"{y} ~ {predictors}"

    for x in _meta['polynomials']:
        data[f'{x}_sq'] = data[x]**2
        formula = formula + " + " + f'{x}_sq'

    for x in _meta['interactions']:
        if len(x) == 2:
            formula = formula + " + " + f'{x[0]}:{x[1]}'
    st.write('the family is', _meta['family'])
    if _meta['priors']:
        st.write('the priors used are:')
        st.write(_meta['priors'])
    else:
        st.write('No priors used')
    st.write('The BAMBI string formula is', formula)

    model = bmb.Model(formula, data, family=family, priors=_meta['priors'])
    st.write('model created. Fiting data...')
    results = model.fit(draws=draws, chains=chains)
    st.write('finished fitting the model to the data')
    return model, results


def visualize_basic_results(results, plot_type):
    """
    Visualize ArviZ plots based on the plot type.

    Parameters
    ----------
    results : arviz.InferenceData
        The model's inference data.
    plot_type : str
        Type of plot: 'forest', 'trace', or 'posterior_predictive'.
    model : bambi.Model, optional
        The Bambi model (required for posterior predictive plots).

    Returns
    -------
    matplotlib.figure.Figure
        The generated plot figure.
    """
    if plot_type == 'forest':
        axes = az.plot_forest(results)
        fig = axes.ravel()[0].figure
        plt.tight_layout()

        st.pyplot(fig)
        plt.close()
    elif plot_type == 'trace':
        axes = az.plot_trace(results)
        fig = axes.ravel()[0].figure
        plt.tight_layout()

        st.pyplot(fig)
        plt.close()
    elif plot_type == 'posterior_predictive':
        axes = az.plot_ppc(results)
        # Extract the figure from the plot
        fig = axes.figure  # az.plot_ppc returns a list of axes, not a grid
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        raise ValueError(f"Unsupported plot type: {plot_type}")


def calculate_metrics(observed, predicted):
    """
    Calculate model performance metrics.

    Parameters
    ----------
    observed : np.array or pd.Series
        Array of observed values.
    predicted : np.array or pd.Series
        Array of predicted values.

    Returns
    -------
    dict
        A dictionary containing R-squared, RMSE, and Willmott's D.
    """
    observed = np.array(observed)
    predicted = np.array(predicted)

    # Mean of observed
    mean_obs = np.mean(observed)

    # Metrics
    r_squared = 1 - np.sum((observed - predicted) ** 2) / np.sum((observed - mean_obs) ** 2)
    rmse = np.sqrt(np.mean((observed - predicted) ** 2))
    willmott_d = 1 - (np.sum((observed - predicted) ** 2) / np.sum((np.abs(predicted - mean_obs) + np.abs(observed - mean_obs)) ** 2))

    return {
        "R-squared": r_squared,
        "RMSE": rmse,
        "Willmott's D": willmott_d
    }

def plot_predicted_vs_actual(results, sensors):
    """
    Extract posterior predictive means and actual values, compute metrics, and create plots.

    Parameters
    ----------
    results : arviz.InferenceData
        The InferenceData object containing observed and predicted data.

    Returns
    -------
    None
        Displays metrics and plots using Streamlit.
    """
    # Extract observed data
    target_variable = st.session_state.bambi_selection['dep_var']
    if "observed_data" not in results.groups() or target_variable not in results.observed_data:
        raise ValueError(f"Observed data for '{target_variable}' not found in InferenceData.")

    obs = results.observed_data
    res = results.posterior_predictive[target_variable].mean(dim=["chain", "draw"])
    obs = obs.to_dataframe().rename(columns={target_variable: 'observed'})
    obs.columns = ['observed']
    res = res.to_dataframe().rename(columns={target_variable: 'predicted'})

    df = pd.concat([obs, res], axis=1)
    st.write("Predicted vs Observed Data")
    st.write(df)

    # Calculate Metrics
    metrics = calculate_metrics(df['observed'], df['predicted'])
    st.subheader("Model Metrics")
    for metric, value in metrics.items():
        st.write(f"{metric}: {value:.3f}")

    # Combined Histogram
    st.subheader("Combined Histogram of Actual and Predicted Values")
    fig_hist = px.histogram(
        df,
        x=['observed', 'predicted'],
        barmode="overlay",
        nbins=20,
        title="Combined Histogram of Actual vs Predicted Values"
    )
    st.plotly_chart(fig_hist)

    # Combined Scatter Plot
    st.subheader("Combined Scatter Plot of Actual vs Predicted")
    fig_scatter = px.scatter(
        df,
        x="observed",
        y="predicted",
        title="Scatter Plot of Actual vs Predicted Values",
        labels={"observed": "Observed", "predicted": "Predicted"}
    )
    maxi = max(df['predicted'].max(), df['observed'].max())
    fig_scatter.add_shape(
        type="line",
        x0=0, y0=0, x1=maxi, y1=maxi,
        line=dict(color="red", dash="dash")
    )
    fig_scatter.update_layout(
        height=500,
        width=500
    )
    st.plotly_chart(fig_scatter)



def visualize_bambi_model(model, results, sensors):
    st.markdown('##### Summary table of results')
    summary_df = az.summary(results)
    st.dataframe(summary_df)
    model.predict(results, kind="pps")
    selection = st.selectbox('Select plot type :', ['trace', 'forest', 'posterior_predictive', 'observed_vs_predicted'])

    if selection == 'trace':
        visualize_basic_results(results, 'trace')

    if selection == 'forest':
        visualize_basic_results(results, 'forest')

    if selection == 'posterior_predictive':
        visualize_basic_results(results, 'posterior_predictive')

    if selection == 'observed_vs_predicted':
        plot_predicted_vs_actual(results, sensors)

def prepare_interaction_raster(meta, data):
    interactions = []
    for interaction in [meta['interaction_one'], meta['interaction_two']]:
        if len(interaction) == 2:
            split_v1 = interaction[0].split('_')
            if len(split_v1) == 2:
                var1 = split_v1[0]
                buf1 = split_v1[1]
            else:
                var1 = split_v1[0] + '_' + split_v1[1]
                buf1 = split_v1[2]
            split_v2 = interaction[1].split('_')
            if len(split_v2) == 2:
                var2 = split_v2[0]
                buf2 = split_v2[1]
            else:
                var2 = split_v2[0] + '_' + split_v2[1]
                buf2 = split_v2[2]
            da = data[var1].sel(buffer=int(buf1)) * data[var2].sel(buffer=int(buf2))
            da.name=("_".join([var1, buf1, var2, buf2]))
            interactions.append(da)
    if len(interactions) == 2:
        interactions_ds = xr.Dataset({
            interactions[0].name: interactions[0],
            interactions[1].name: interactions[1]
        })
    elif len(interactions) == 1:
        interactions_ds = xr.Dataset({interactions[0].name: interactions[0]})
    else:
        interactions_ds = None
    return interactions_ds

def extract_predictors(oldds, buffer_dict, selectionx, selectiony):
    """
    Extract predictor variables from a NetCDF dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset containing variables and buffers.
    buffer_dict : dict
        A mapping of variable names to buffer sizes (or None for non-buffered).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing predictors with (X, Y) indices.
    """

    # Slice the dataset for a subset
    ds = oldds.sel(X=slice(selectionx[0], selectionx[1]), Y=slice(selectiony[1],selectiony[0]))
    predictors = []
    for varname, buffer in buffer_dict.values():
        # Select the appropriate buffer layer or the variable directly
        data = ds[varname].sel(buffer=buffer).to_dataframe().reset_index().drop('buffer',axis=1)
        data = data.rename(columns={varname: varname + '_' + str(buffer)}).set_index(['X','Y'])
        predictors.append(data)

    # Combine all predictors into a single xarray.Dataset
    combined = pd.concat(predictors,axis=1)

    # Convert to a pandas DataFrame with (X, Y) indices
    return combined

@st.cache_data
def apply_bambi_model_to_netcdf(_model, _ds, _interactions, _results, buffer_dict):
    """
    Applies a fitted Bambi model to a NetCDF dataset for prediction, keeping track of cell indices.

    Parameters
    ----------
    _model : bambi.Model
        The fitted Bambi model.
    ds : xr.Dataset
        The NetCDF dataset containing predictor variables.
    buffer_dict : dict
        A mapping of predictor variable names to buffer sizes.
        Example: {"fitnah_temp": [50, 250], "Forest": [250]}
    results : arviz.InferenceData
        The fitted model's posterior samples.

    Returns
    -------
    xr.DataArray
        A 2D xarray.DataArray of predictions, reshaped to match the NetCDF grid.
    pd.DataFrame
        A DataFrame of the flattened data including `(Y, X)` indices and predictions.
    """

    # Extract grid information
    ystart = 1.218
    yend = 1.223
    xstart = 2.584
    xend = 2.589
    xs = [x*1e6 for x in np.arange(xstart, xend, 0.001)]
    ys = [y*1e6 for y in np.arange(ystart, yend, 0.001)]

    # Create a list of squares
    squares = [
        [(x, x + 1e3), (y, y + 1e3)] for x in xs for y in ys
    ]

    dfs = []
    for i, square in enumerate(squares):
        predictors = extract_predictors(_ds, buffer_dict, square[0], square[1])
        predictors_ds = predictors.to_xarray()
        filtered = predictors[predictors.notna().all(axis=1)]
        pred = _model.predict(_results, 'pps',  filtered, inplace=False)
        # Extract posterior predictive data
        posterior_predictive = pred.posterior_predictive
        # Compute mean and standard deviation across chains and draws
        pred_mean = posterior_predictive.mean(dim=("chain", "draw")).to_dataframe().reset_index(drop=True)
        pred_std = posterior_predictive.std(dim=("chain", "draw")).to_dataframe().reset_index(drop=True)
        pred_mean['X'] = filtered.reset_index()['X']
        pred_mean['Y'] = filtered.reset_index()['Y']
        damean = pred_mean.set_index(['X','Y'])[st.session_state.bambi_selection['dep_var']].to_xarray()
        damean = damean.reindex_like(predictors_ds)
        damean = damean.where((damean >= 0) & (damean <= 10), np.nan)
        damean.name=st.session_state.bambi_selection['dep_var']
        pred_std['X'] = filtered.reset_index()['X']
        pred_std['Y'] = filtered.reset_index()['Y']
        dastd = pred_std.set_index(['X', 'Y'])[st.session_state.bambi_selection['dep_var']].to_xarray()
        dastd.name = st.session_state.bambi_selection['dep_var']
        # rasters.append(dastd)
        test = damean.reindex
        dfs.append(pred_mean.set_index(['X','Y'])[st.session_state.bambi_selection['dep_var']])

    #total = xr.concat(rasters, dim=['X','Y'])
    total = pd.concat(dfs,axis=0)
    sorted = total.sort_index().reset_index()
    sorted = sorted.drop_duplicates(['X','Y'])
    dstotal = sorted.set_index(['X','Y']).to_xarray()


    return dstotal



def visualize_predictions(pred_da, sensor_locs):
    """
    Visualizes predictions and overlays sensor locations on the plot.

    Parameters
    ----------
    pred_da : xr.DataArray
        The mean predictions reshaped into a 2D grid as an xarray.DataArray.
    sensor_locs : pd.DataFrame
        A DataFrame with sensor locations, must contain 'X' and 'Y' columns.

    Returns
    -------
    None
    """
    target_variable = st.session_state.bambi_selection['dep_var']
    # Ensure sensor_locs has the required columns
    if not {'X', 'Y'}.issubset(sensor_locs.columns):
        raise ValueError("sensor_locs must contain 'X' and 'Y' columns.")

    # Plot the predictions using Matplotlib

    fig, ax = plt.subplots(figsize=(8, 6))
    pred_da[target_variable].plot(ax=ax, cmap="viridis", cbar_kwargs={"label": "Prediction Mean"}, x='X',y='Y')

    # Overlay sensor locations
    ax.scatter(x=sensor_locs['X'], y=sensor_locs['Y'], color='red', s=50, label='Sensor Locations', edgecolor='black')

    # Add plot details
    ax.set_title("Predictions for Biel")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.legend()

    # Adjust layout and render
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def visualize_predictions_interactive(pred_da, sensors):
    """
    Visualizes predictions and overlays sensor locations on an interactive Plotly map with hover data.

    Parameters
    ----------
    pred_da : xr.DataArray
        The mean predictions reshaped into a 2D grid as an xarray.DataArray.
    sensors : pd.DataFrame
        A DataFrame with sensor locations, must contain 'X', 'Y', and additional hover data columns.

    Returns
    -------
    None
    """
    # Filter sensor locations
    sensor_locs = sensors[~sensors.logger.isin([207, 239, 240, 231])]
    st.write(sensor_locs)

    # Validate columns
    required_columns = {'X', 'Y', 'logger', 'Long_name', st.session_state.bambi_selection['dep_var']}
    if not required_columns.issubset(sensor_locs.columns):
        raise ValueError(f"sensor_locs must contain columns: {required_columns}")
    # Convert the xarray DataArray to a DataFrame for use with Plotly
    pred_df = pred_da.to_dataframe().reset_index()

    # Create a scatter map using Plotly Express
    fig = px.density_heatmap(
        pred_df,
        x="X",
        y="Y",
        z=st.session_state.bambi_selection['dep_var'],  # Use the actual DataArray name
        nbinsx=500,
        nbinsy=500,
        histfunc='avg',
        color_continuous_scale="Viridis",
        labels={st.session_state.bambi_selection['dep_var']: "Prediction Mean"},
        title="Predictions for Biel"
    )
    hd_extra = [f'{x}_{y}' for x, y in st.session_state.bambi_selection['vars']]

    def helper_function(row):
        """
        Generates a hover text string for a given row in a DataFrame.

        Parameters
        ----------
        row : pd.Series
            A single row of the DataFrame containing sensor and prediction data.

        Returns
        -------
        str
            The hover text string for the row.
        """
        # Base hover text with logger, location, and dependent variable
        string = (
            f"Logger: {row['logger']}<br>"
            f"Location: {row['Long_name']}<br>"
            f"{st.session_state.bambi_selection['dep_var']}: {row[st.session_state.bambi_selection['dep_var']]:.2f}<br>"
        )

        # Add extra hover data dynamically based on hd_extra length
        for key in hd_extra:
            string += f"{key}: {row[key]}<br>"

        return string

    # Overlay sensor locations with hover data
    # Overlay sensor locations with hover data
    fig.add_scatter(
        x=sensor_locs["X"],
        y=sensor_locs["Y"],
        mode="markers",
        marker=dict(size=10, color="red", symbol="circle"),
        name="Sensor Locations",
        hovertext=sensor_locs.apply(
            lambda row: helper_function(row),
            axis=1
        )
    )

    # Adjust layout
    fig.update_layout(
        xaxis_title="X (meters)",
        yaxis_title="Y (meters)",
        coloraxis_colorbar=dict(title="Prediction Mean"),
        hovermode='closest',
        height=500,
        width=800

    )


    # Display in Streamlit
    st.plotly_chart(fig)

