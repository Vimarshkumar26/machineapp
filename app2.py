import streamlit as st
import pandas as pd
import os
import pandas_profiling
from pycaret.regression import setup, compare_models, pull, save_model
from streamlit_pandas_profiling import st_profile_report
import sweetviz as sv

with st.sidebar:
    st.image("D:\\SIG\\ML\\Project\\quadratic_img\\car10.jpg")
    st.title("ML Auto:computer:")
    buttons = st.radio("Proceed",['Upload','Describe','Visualization','Preprocessing','Models','Download'])
    st.info("Powered by: Nelumbus Tech")
#st.write(" # We are building AutoML platform!")
st.write("<span style='color:purple; font-size:28px; font-family: Verdana, serif;'>We are bulding AutoML Platform.</span>", unsafe_allow_html=True)

if os.path.exists("org_data.csv"):
    df = pd.read_csv("org_data.csv",  index_col =None)


if buttons == 'Upload':
    st.title("Choose your data for insights")
    data = st.file_uploader("Upload your file here :")
    if data:    
        df = pd.read_csv(data,  index_col =None)
        df.to_csv('org_data.csv', index = None)
        st.dataframe(df)

if buttons == 'Describe':
    st.title("Description of your data:")
    pr = df.profile_report()
    st_profile_report(pr)

if buttons == 'Visualization':
    viz_report = sv.analyze(df)
    viz_report.show_html()

if buttons == 'Models':
    st.title("Performing Algorithms:")
    target = st.selectbox("Choose target variable",df.columns)
    if st.button("Train Model"):   
        setup(df, target = target)
        setup_df = pull()
        st.info("These are settings")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("ML models:")
        st.dataframe(compare_df)
        best_model
        save_model(best_model, 'best')

if buttons == 'Download':
    with open("best.pkl",'rb') as f:
        st.download_button("Download the result", f, 'best.pkl')



