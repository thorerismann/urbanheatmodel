import streamlit as st
from modules.bambi import load_base_data, display_variable_menu, \
    create_bambi_model, prepare_tn_data, visualize_bambi_model, apply_bambi_model_to_netcdf, \
    visualize_predictions_interactive, prepare_uhi_data, display_model_menu
from modules.params import main_squares
from modules.see_base_data import main_viz

st.title("Modeling Urban Heat in Biel - A simple approach")

tabs = st.tabs(["Bambi Modeling", "Forest Model", "Other Model", "Raw Data Viz", "Explanation"])

# 1. Bambi Modeling Tab
with tabs[0]:
    st.header("Bambi Modeling")
    df, s, z = load_base_data()
    m1, m2 = st.columns(2)
    with m1:
        display_variable_menu()
    with m2:
        display_model_menu()
    st.warning(
        'This program does not check user inputs. Bayesian modeling can be resource intensive, '
        'so check that inputs make sense before you run. For example, that all buffer/variable '
        'combinations are unique (unless squaring one of them!) before running.'
    )

    meta_bambi = st.session_state.get('bambi_selection')
    if meta_bambi:
        if meta_bambi['dep_var'] == 'tropical_nights':
            data = prepare_tn_data(df, s, [207, 240, 239, 231])
            st.write("Mean of tropical_nights:", data['tropical_nights'].mean())
            st.write("Variance of tropical_nights:", data['tropical_nights'].var())
        else:
            data = prepare_uhi_data(df, s, [207, 240, 239, 231])
        with st.container(border=True):
            model, results = create_bambi_model(data, meta_bambi)
        with st.container(border=True):
            if st.toggle('Plot Model Results', False):
                st.subheader('Model Results')
                visualize_bambi_model(model, results, s)
        with st.container(border=True):
            if st.toggle('Apply model to the city of Biel'):
                st.write('Be patient...takes some time')
                sensors = s.merge(data[['logger',meta_bambi['dep_var']]], on='logger', how='left')
                map = apply_bambi_model_to_netcdf(model, z, results, st.session_state.bambi_selection['indep_dict'], main_squares)
                st.write('prediction map created')
                visualize_predictions_interactive(map, sensors)
# 2. Random Forest Tab
with tabs[1]:
    st.header("Random Forest Model")
    st.write("Train a Random Forest model.")
    # Placeholder for scikit-learn Random Forest

# 3. Other Model Tab
with tabs[2]:
    st.header("Other Models")
    st.write("Placeholder for future models.")

# 4. Raw Data Viz Tab
with tabs[3]:
    st.header("Raw Data Visualization")
    main_viz()

with tabs[4]:
    st.write('Explanation')