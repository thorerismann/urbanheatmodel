"""# Streamlit Model Setup Explanation

Below is a detailed explanation of how the Streamlit forms gather user selections and store them in a dictionary for Bayesian model construction using Bambi.

## Overview

1. **Variable Selection**  
   - The user selects up to five geospatial variables (e.g., `"Water"`, `"Buildings"`, etc.) and pairs each with a buffer distance (either 50m or 250m).  
   - Only the first `number_of_vars` are ultimately kept for the model.  
   - The combination of a geospatial variable and a buffer distance is stored in `indep_dict` as a key (e.g., `"Water_50"`).

2. **Response Variable**  
   - The user chooses either `"tropical_nights"` or `"uhi"` as the dependent variable in the model (`dep_var`).

3. **Model Menu**  
   - **Chains** and **Draws**: Control how many Markov Chain Monte Carlo (MCMC) iterations to run (`chains` and `draws` keys). Higher values increase accuracy but also computational cost.  
   - **Family**: Select the likelihood distribution that suits the data (`family` key). Examples include `"t"`, `"negativebinomial"`, `"poisson"`, or `"gaussian"`.  
   - **Squared Terms**: Any selected variable can be squared instead of using its linear value (`squared_terms`).  
   - **Interaction Terms**: Users can optionally pick two pairs of variables to create interaction terms (`interaction_one` and `interaction_two`), which capture multiplicative effects in the model.

4. **Priors**  
   - Users can toggle optional priors, which allow specifying sigma (`sigma`), mu (`mu`), and whether to use fat tails (`heavy_tails`) if needed.  
   - These are stored as part of `priors`, or `None` if not used.

5. **Session State Storage**  
   - Once submitted, the chosen configurations are saved in `st.session_state['bambi_selection']`, a dictionary containing all the key settings needed to build and fit the Bayesian model.

---

## Dictionary Keys in `st.session_state['bambi_selection']`:

- **`indep_dict`**: A dictionary mapping `"var_buf"` strings (e.g., `"Water_50"`) to tuples of `(var, buf)` representing the selected independent variables and buffers.  
- **`dep_var`**: The dependent (response) variable chosen by the user (`"tropical_nights"` or `"uhi"`).  
- **`chains`**: Integer representing how many MCMC chains will be run.  
- **`draws`**: Integer for how many draws (iterations) will be done in each chain.  
- **`family`**: The probability distribution family selected (e.g., `"gaussian"`, `"poisson"`, etc.).  
- **`squared_terms`**: A list of selected variables (in `"var_buf"` format) that will be squared in the model formula.  
- **`interaction_one`**: A list containing two `"var_buf"` strings selected for the first interaction term.  
- **`interaction_two`**: A list containing two `"var_buf"` strings selected for the second interaction term.  
- **`priors`**: If priors are toggled on, contains user-specified prior settings for sigma, mu, and whether to use fat tails (or `None` if no priors were selected).

Use the following snippet with a Streamlit expander to display this information:

```python"""