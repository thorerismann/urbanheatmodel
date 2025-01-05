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


def display_variable_menu():
    with st.form('variable_menu'):
        st.markdown('##### Select variables')
        st.text('The variable-buffer combinations represent the values of rester grids at 10m taken as either count (landuse categories) or mean (fitnah data) at the 50 or250 meters buffer for the selected variable.')
        st.text('The dependent variables are tropical nights (count data, use Poisson or negative binomial family in the next menu) or urban heat island value (use t or gaussian family in the next menu)')
        # Available buffers and geospatial variables
        buffers_list = [50, 250]
        numvars = 5
        geo_vars = [
            "Water", "Buildings", "Paved", "Developed", "Forest",
            "Grass", "fitnah_ss", 'fitnah_sv', 'fitnah_temp'
        ]
        c1, c2 = st.columns(2)
        with c1:
            # Select buffers
            selected_buffers = [
                st.radio(f'Select Buffer for var{x}', buffers_list, key=f"buffer{x}", horizontal=True)
                for x in range(1, numvars + 1)
            ]
        with c2:
            # Select geospatial variables
            selected_vars = [
                st.selectbox(f'Select variable for var{x}', geo_vars, key=f"var{x}")
                for x in range(1, numvars + 1)
            ]

        number_of_vars = st.number_input(
            'Select the number of variables (excluding interaction terms). Variables are selected in order',
            min_value=1, max_value=5, value=5
        )

        response = st.selectbox(
            'Select Response Variable:',
            ['tropical_nights', 'uhi']
        )
        submit_model = st.form_submit_button('Submit Step One')
    if submit_model:
        indep_dict = {}
        final_vars = selected_vars[:number_of_vars]
        final_bufs = selected_buffers[:number_of_vars]

        for (var, buf) in zip(final_vars, final_bufs):
            key = f"{var}_{buf}"  # e.g. "Water_50"
            indep_dict[key] = (var, buf)

        st.session_state['first_menu']  = {'dep_var': response, 'indep_dict': indep_dict, 'final_vars': final_vars, 'final_bufs': final_bufs}

def display_model_menu():
    if not st.session_state.get('first_menu'):
        st.info("Select variables above to get started with modelling")
    if st.session_state.get('first_menu'):

        selected_vars = st.session_state.first_menu['final_vars']
        selected_buffers = st.session_state.first_menu['final_bufs']
        dep_var = st.session_state.first_menu['dep_var']
        indep_dict = st.session_state.first_menu['indep_dict']
        with st.form('model_menu'):
            st.markdown('##### Select Details About the Model')
            st.text('Change the model here. Increasing the chains and draws will increase accuracy and cost computing time and memory.')
            st.text('Note that squaring the variable will replace the original value.')
            st.text('Choose the appropriate family for the uhi or for tropical nights.')
            st.text('fat tails, mu and sigma are only relevant if priors are toggled.')
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

            # Using the first `number_of_vars` only
            squared_terms = st.multiselect(
                'select variables to square',
                [f'{v}_{b}' for v, b in zip(selected_vars, selected_buffers)]
            )
            interaction_one = st.multiselect("Select first interaction term (select two!)", [f'{v}_{b}' for v, b in zip(selected_vars, selected_buffers)], max_selections=2, key='interaction_one')
            interaction_two = st.multiselect("Select second interaction term (select two!", [f'{v}_{b}' for v, b in zip(selected_vars, selected_buffers)], max_selections=2, key='interaction_two')

            c1, c2 = st.columns(2)
            with c1:
                use_priors = st.toggle('Use priors')
                sigma = st.number_input('Sigma', 1.0)
            with c2:
                heavy_tails = st.checkbox('Fat tails', False)
                mu = st.number_input('Mu', 0.0)



            submit = st.form_submit_button('create model')
        if submit:
            st.cache_data.clear()
            st.cache_resource.clear()
            if use_priors:
                priors = create_priors(list(indep_dict.keys()), family, sigma, mu, heavy_tails)
            else:
                priors=None
            # Store everything in session state
            st.session_state['bambi_selection'] = {
                'indep_dict': indep_dict,
                'dep_var': dep_var,
                'chains': chains,
                'draws': draws,
                'family': family,
                'squared_terms': squared_terms,
                'interaction_one': interaction_one,
                'interaction_two': interaction_two,
                'priors': priors
            }

            st.json(st.session_state['bambi_selection'], expanded=False)


def create_priors(var_list, family, sigma=0.5, mu=0, heavy_tails=False):
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
                priors[var] = bmb.Prior("Normal", mu=mu, sigma=sigma)

    elif family == 't':
        # Priors for Student's t-distribution models
        for var in var_list:
            if heavy_tails:
                priors[var] = bmb.Prior("StudentT", mu=mu, sigma=sigma, nu=3)
            else:
                priors[var] = bmb.Prior("StudentT", mu=mu, sigma=sigma)

    return priors

def prepare_uhi_data(station_data, sensors, excluded_loggers):
    """
    Example usage of the stored 'indep_dict' to build the data.
    """
    dep = station_data.groupby('logger')['uhi'].mean()

    # Grab the dictionary from session_state
    vb = st.session_state.bambi_selection['indep_dict'].values()
    # vb is something like [(50, "Water"), (250, "Buildings"), ...]

    # Convert (buf, var) => "var_buf"
    columns_needed = [f"{var}_{buf}" for (var, buf) in vb]

    # Merge with logger
    data = sensors[['logger'] + columns_needed]
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
    Example usage of the stored 'indep_dict' to build the data.
    """
    dep = station_data.groupby('logger')['tropical_nights'].sum()

    # Grab the dictionary from session_state
    vb = st.session_state.bambi_selection['indep_dict'].values()
    # vb is something like [(50, "Water"), (250, "Buildings"), ...]

    # Convert (buf, var) => "var_buf"
    columns_needed = [f"{var}_{buf}" for (var, buf) in vb]

    data = sensors[['logger'] + columns_needed]
    full = data.merge(pd.DataFrame(dep).reset_index(), on='logger')
    full = full[~full.logger.isin(excluded_loggers)]

    total_na = full.isna().sum()
    if total_na.sum() > 0:
        st.write('total na values', total_na)
        full = full.dropna(how='any')
    full = full.sort_values(by=['logger'])

    return full

def create_interactions(meta, data):
    if len(meta['interaction_one']) == 2:
        data["_".join(meta['interaction_one'])] = data[meta['interaction_one'][0]] * data[meta['interaction_one'][1]]
    if len (meta['interaction_two']) == 2:
        data['_'.join(meta['interaction_two'])] = data[meta['interaction_two'][0]] * data[meta['interaction_two'][1]]
    return data

def add_squared(meta, data):
    if len(meta['squared_terms']) == 0:
        st.write('No terms to square')
        return data
    else:
        st.write('Squaring terms: ' + str(meta['squared_terms']))
        for col in meta['squared_terms']:
            data[col] = data[col]**2



@st.cache_resource
def create_bambi_model(data, _meta, draws=500, chains=2):

    y = st.session_state.bambi_selection['dep_var']
    z = st.session_state.bambi_selection['indep_dict'].values()
    # Add interaction terms to the dataset
    data = create_interactions(_meta, data)
    st.write('final dataset is:')
    st.write(data)
    base_predictors = " + ".join([f"{var}_{buf}" for var, buf in z])

    # Interaction terms (if any)
    interaction_terms = []
    if len(_meta.get('interaction_one', [])) == 2:
        interaction_terms.append("_".join(_meta['interaction_one']))
    if len(_meta.get('interaction_two', [])) == 2:
        interaction_terms.append("_".join(_meta['interaction_two']))

    # Combine predictors and interactions into the formula
    all_predictors = base_predictors
    if interaction_terms:
        all_predictors += " + " + " + ".join(interaction_terms)

    formula = f"{y} ~ {all_predictors}"
    st.write('the formula is', formula)
    st.write('the family is', _meta['family'])
    if _meta['priors']:
        st.write('the priors used are:')
        st.write(_meta['priors'])
    else:
        st.write('No priors used')
    model = bmb.Model(formula, data, family=_meta['family'], priors=_meta['priors'])
    st.write('model created. Fiting data...')
    results = model.fit(draws=draws, chains=chains)
    st.write('finished fitting the model to the data')
    return model, results


def visualize_basic_results(results, plot_type, model):
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
        visualize_basic_results(results, 'trace', model)

    if selection == 'forest':
        visualize_basic_results(results, 'forest', model)

    if selection == 'posterior_predictive':
        visualize_basic_results(results, 'posterior_predictive', model)

    if selection == 'observed_vs_predicted':
        plot_predicted_vs_actual(results, sensors)


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
def apply_bambi_model_to_netcdf(_model, _ds, _results, buffer_dict, squares):
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
    hd_extra = [f'{x}_{y}' for x, y in st.session_state.bambi_selection['indep_dict'].values()]

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
