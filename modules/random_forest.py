import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def train_random_forest(station_data, sensors, excluded_loggers=None,
                        n_estimators=100, max_depth=None, random_state=42):
    """
    Trains a Random Forest model on the selected features and target.

    Parameters
    ----------
    station_data : pd.DataFrame
        Contains your station observations, including the target variable.
    sensors : pd.DataFrame
        Contains the merged sensor data and predictor columns.
    excluded_loggers : list of int, optional
        Station logger IDs to exclude from training.
    n_estimators : int, optional
        Number of trees in the random forest.
    max_depth : int, optional
        Maximum depth of the tree.
    random_state : int, optional
        Random state for reproducibility.

    Returns
    -------
    model : RandomForestRegressor
        The trained random forest model.
    X_train, X_test, y_train, y_test : pd.DataFrame
        The train/test split data, useful for evaluating performance.
    """
    if excluded_loggers is None:
        excluded_loggers = []

    resp = st.session_state.bambi_selection['dep_var']  # e.g., 'tropical_nights'

    # Get predictor columns from the bambi_selection dict
    indep_cols = [f"{var}_{buf}" for var, buf in st.session_state.bambi_selection['indep_dict'].values()]

    # Merge station_data + sensors to get the final DataFrame
    dep = station_data.groupby('logger')[resp].sum().reset_index()
    merged = sensors.merge(dep, on='logger')
    merged = merged[~merged.logger.isin(excluded_loggers)]

    # Drop rows with missing data
    merged = merged.dropna(subset=indep_cols + [resp])

    # Prepare X (predictors) and y (target)
    x = merged[indep_cols]
    y = merged[resp]

    # Split into train/test for demonstration
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=random_state
    )

    # Create and fit the RandomForestRegressor
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(x_train, y_train)

    st.write("Random Forest model trained successfully!")
    st.write(f"Train size: {x_train.shape}, Test size: {x_test.shape}")

    return model, x_train, x_test, y_train, y_test
