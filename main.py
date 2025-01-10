"""
Module: main.py

This is the entry point for the Streamlit application, enabling users to interact
with various modules for urban heat modeling and visualization. The application
includes tabs for Bayesian modeling with Bambi, raw data visualization, and
detailed explanations of model setup and processes.
"""
from pathlib import Path

from modules.bambi import *
from modules.bambi import prepare_rep_data
from modules.see_base_data import main_viz

st.title("Simple Application of Bayesian GLMs to Urban Heat in Biel/Bienne")
st.write('See the explanation tab for project background and instructions.')
tabs = st.tabs(["Bambi Modeling", "Raw Data Viz", "Explanation"])

# 1. Bambi Modeling Tab
with tabs[0]:
    st.header("Bambi Modeling")

    df, s, z = load_base_data()
    with st.expander('Display Bambi Model Menu'):
        display_bambi_menu()
    meta_bambi = st.session_state.get('bambi_selection')
    if meta_bambi:
        data = prepare_rep_data(df, s, [207, 240, 239, 231, 230], meta_bambi['dep_var'])
        with st.container(border=True):
            cache_dict = {k: v for k,v in meta_bambi.items() if k != 'priors'}
            model, results = create_bambi_model(data, meta_bambi, cache_dict)
            if st.button('Run a new model'):
                st.session_state.clear()
                if st.session_state.get('bambi_model'):
                    del st.session_state['bambi_model']
                st.rerun()
        with st.container(border=True):
            if st.toggle('Plot Model Results', False):
                st.subheader('Model Results')
                visualize_bambi_model(model, results)
        with st.container(border=True):
            if st.toggle('Apply model to the city of Biel'):
                st.write('Be patient...takes some time')
                interactions = prepare_interaction_raster(meta_bambi, z)
                st.write('prepped')
                sensors = s.merge(data[['logger',meta_bambi['dep_var']]], on='logger', how='left')
                mappy = apply_bambi_model_to_netcdf(model, z, interactions, results, st.session_state.bambi_selection['vars'])
                st.write('prediction map created')
                visualize_predictions_interactive(mappy, sensors)

with tabs[1]:
    st.header("Raw Data Visualization")
    main_viz()

with tabs[2]:
    markdown_path = Path.cwd() / 'README.md'
    with open(markdown_path, 'r') as file:
        markdown_content = file.read()
    st.markdown(markdown_content)
