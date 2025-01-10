# Simple Modelling of Biel/Bienne's Hot Spots

Use the data layers provided by the canton of Bern and SwissTopo and the empirical data from the summer 2023 measurement campaign season to model the distribution of urban heat in Biel/Bienne.

---

## Tabs in the Application

### 1. **Bambi Modeling**

   - Build Bayesian models using the Bambi library.
   - Select up to five geospatial variables (e.g., "Water", "Buildings") paired with buffer distances.
   - Define model parameters like chains, draws, and likelihood distributions (e.g., "gaussian", "poisson").
   - Include squared terms or interaction terms for selected variables.
   - Optionally apply priors (e.g., for sigma or mu) for Bayesian inference.
   - Visualize model results and apply the model to city-wide raster data for prediction.

### 2. **Raw Data Visualization**
   - Explore raw raster data interactively.
   - Overlay sensor points on raster maps with hover information for additional context.
   - Select specific raster layers and buffer distances for analysis.

### 3. **Explanation**
   - Provides a comprehensive overview of the modeling workflow.
   - Details the roles of modules like `load_data`, `bambi`, and `see_base_data`.

---

## Modules Overview

### `load_data`
Handles data loading and preprocessing:
   - Load meteorological data (e.g., temperature readings).
   - Process station temperature data for urban heat island analysis.
   - Load geospatial raster data and sensor information.

### `bambi`
Facilitates Bayesian modeling:
   - Define models with selected variables, interactions, and priors.
   - Visualize results and calculate metrics like R-squared and RMSE.
   - Apply models to city-wide raster datasets for prediction.

### `see_base_data`
Supports data visualization:
   - Visualize raw raster data layers interactively.
   - Overlay sensor points on maps with detailed hover data.
   - Filter and select specific data for analysis.

---

## Packages Used
The application uses the following Python packages:
   - **Streamlit**: For building the user interface.
   - **Bambi**: For Bayesian model construction and fitting.
   - **ArviZ**: For visualizing and summarizing Bayesian models.
   - **xarray**: For handling geospatial raster data.
   - **Pandas**: For data manipulation and analysis.
   - **NumPy**: For numerical computations.
   - **Plotly**: For interactive data visualization.
   - **Matplotlib**: For static visualizations.
   - **Scikit-learn**: For machine learning utilities (in future integrations).

---

## Acknowledgments
This would not have been possible without:

- **The City of Biel** for supporting this urban heat modeling initiative, in particular **Miro Meyer**
- **The Geographical Institute of the University of Bern**, for providing the materials and the science, in particular **Prof. Stefan Brönnimann** and ** Dr. Moritz Gubler**
- **Roger Erismann**, for providing coding advice, hacks and application review
- **Annelise Erismann**, for giving me the time to get good at this programming stuff.

---

Thank you for using this application!
