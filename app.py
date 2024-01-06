from PIL import Image
#from streamlit_shap import st_shap
import streamlit as st
import numpy as np 
import pandas as pd 
import time
import plotly.express as px 
import seaborn as sns
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error,confusion_matrix,accuracy_score,recall_score,precision_score,classification_report,roc_auc_score
import shap
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import pickle

plt.style.use('default')

st.set_page_config(
    page_title = 'Real-Time stroke Detection',
    page_icon = '🕵️‍♀️',
    layout = 'wide'
)

data = pd.read_csv("train_data.csv",index_col=False,header=0)
data=data.drop(data.columns[0], axis=1)
X = data.drop("outcome", axis=1)

# dashboard title
#st.title("Real-Time Fraud Detection Dashboard")
st.markdown("<h1 style='text-align: center; color: black;'>Real-time prediction of in-hospital mortality for ischemic stroke patients in ICU</h1>", unsafe_allow_html=True)

# side-bar 

def user_input_features():
    st.sidebar.header('User input parameters below ⬇️')
    a1 = st.sidebar.slider('Age(years)',min_value=18.0, max_value=100.0, value=60.0,step=0.1)
    a2 = st.sidebar.slider('Hematocrit(%)',min_value=10.0, max_value=70.0, value=30.0,step=0.1)
    a3 = st.sidebar.slider('Systolic blood pressure(mmHg)', min_value=70.0, max_value=190.0, value=100.0,step=0.1)
    a4 = st.sidebar.selectbox("Statins", ('Yes', 'No'))
    a5 = st.sidebar.slider('Blood urea nitrogen(mg/dL)', min_value=5.0, max_value=70.0, value=30.0,step=0.1)
    a6 = st.sidebar.slider('White blood cell(10^9/L)', min_value=0.0, max_value=25.0, value=10.0,step=0.1)
    a7 = st.sidebar.selectbox("Warfarin", ('Yes', 'No'))
    a8 = st.sidebar.selectbox("Mechanical ventilation", ('Yes', 'No'))
    a9 = st.sidebar.slider('Bicarbonate(mEq/L)', min_value=10.0, max_value=40.0, value=23.0,step=0.1)
    if a4 == 'Yes':
            a4 = 1
    else:
        a4 = 0
    if a7 == 'Yes':
            a7 = 1
    else:
        a7 = 0
    if a8 == 'Yes':
            a8 = 1
    else:
        a8 = 0

    output = [a1,a2,a3,a4,a5,a6,a7,a8,a9]
    return output


outputdf = user_input_features()

st.subheader('About the model')
st.write('The algorithm used in this freely accessible online calculator is based on random forest.Internal validation of the model showed an area under the curve (AUC) of 0.908 (95% CI: 0.882-0.933) indicating strong predictive performance, with calibration curves and decision curve analysis demonstrating good calibration and clinical yield. Although the model achieved good predictive performance, it is important to note that its use should be limited to research purposes only. This means that the model can be used to gain insights, explore relationships and generate hypotheses in a research setting. However, additional research, external validation and rigorous evaluation are required before the model can be deployed in a real-world setting.')

st.subheader('Guidelines of the Calculator ')
st.write('The calculator consists of 3 main sections.The left sidebar of the first section allows users to input relevant parameters and select model variables. The second displays the predicted probability of in-hospital mortality. The third provides detailed model information, including global and local interpretations using SHAP providing insight into prediction generation. We hope this guide helps you effectively utilize our prediction calculator.')

image4 = Image.open('shap.png')
shapdatadf =pd.read_excel(r'shapdatadf.xlsx')
shapvaluedf =pd.read_excel(r'shapvaluedf.xlsx')
# 这里是查看SHAP值

st.subheader('Make predictions in real time')
outputdf = pd.DataFrame([outputdf], columns= shapdatadf.columns)

#st.write('User input parameters below ⬇️')
#st.write(outputdf)
with open("RF1.pickle", "rb") as file:
    RF = pickle.load(file)

p1 = RF.predict(outputdf)[0]
p2 = RF.predict_proba(outputdf)
p2 = round(p2[0][1], 4)

#st.write('User input parameters below ⬇️')
#st.write(outputdf)
st.write(f'Probability of in-Hospital mortality for ischemic stroke patients in ICU: {p2}')
st.write(' ')

st.subheader("SHAP")
placeholder5 = st.empty()
with placeholder5.container():
    f1,f2 = st.columns(2)

    with f1:
        st.write('Beeswarm plot')
        st.write(' ')
        st.image(image4)
        st.write('The beeswarm plot is designed to display an information-dense summary of how the top features in a dataset impact the model’s output. Each instance the given explanation is represented by a single dot on each feature fow. The x position of the dot is determined by the SHAP value of that feature, and dots “pile up” along each feature row to show density.')     
    with f2:
        st.write('Dependence plot for features')
        cf = st.selectbox("Choose a feature", (shapdatadf.columns))
        fig = px.scatter(x = shapdatadf[cf], 
                         y = shapvaluedf[cf], 
                         color=shapdatadf[cf],
                         color_continuous_scale= ['blue','red'],
                         labels={'x':'Original value', 'y':'shap value'})
        st.write(fig)  


# 这里是查看SHAP和LIME图像的

placeholder6 = st.empty()

explainer   = shap.TreeExplainer(RF)
shap_values = explainer.shap_values(outputdf)

 #st_shap(shap.plots.waterfall(shap_values[0]),  height=500, width=1700)
st.write('Force plots')
st.set_option('deprecation.showPyplotGlobalUse', False)
shap.force_plot(explainer.expected_value[1], shap_values[1][0,:],outputdf.iloc[0,:],link='logit',show=False, matplotlib=True)
st.pyplot(bbox_inches='tight')
st.write('The SHAP force plot can be used to visualise the SHAP value for each feature as a force that can increase (positive) or decrease (negative) the prediction relative to its baseline for the interpretation of individual patient outcome predictions.')



