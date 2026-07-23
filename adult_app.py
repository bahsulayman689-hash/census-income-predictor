#---------------------- adult_app.py---------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.tree import DecisionTreeClassifier

# --------------------------------------------------------------------------------------------------
# 1. PAGE SETUP & GREEN SIDEBAR THEME
# --------------------------------------------------------------------------------------------------
st.set_page_config(
    page_title="💰Adult Income Classifier", 
    page_icon="💰", 
    layout="wide"
)

st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            background-color: #2e7d32;
        }
        [data-testid="stSidebar"] *, [data-testid="stSidebar"] p, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #ffffff !important;
        }
        [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] hr {
            border-color: rgba(255, 255, 255, 0.3) !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)
header_col1, header_col2 = st.columns([4, 1])

with header_col1:
    st.title("💰Census Income Prediction App")
    
    st.write("""
    Hi! I'm Sulayman Bah.
    I'm a mechine learning and deep learning enginner.
    I build Machine Learning and
    Deep Learning applications
    using Python and Streamlit.
    """)
with header_col2:
    if os.path.exists("euro.png"):
        st.image("euro.png", width=200)
    else:
        st.info("💡 Logo asset not found.")
    
st.markdown("---")
# --------------------------------------------------------------------------------------------------
# 2. SIDEBAR CONTENT (LOGO, IMAGE & METADATA)
# --------------------------------------------------------------------------------------------------
with st.sidebar:
    if os.path.exists("euro.png"):
        st.image("euro.png", use_container_width=True)
    else:
        st.subheader("🟢 System Navigation")
        st.caption("(Tip: Add a 'logo.png' image file to your folder)")
    
    st.markdown("---")
    
    if os.path.exists("IMG-20260704-WA0629.jpg"):
        st.image("IMG-20260704-WA0629.jpg", caption="App Operator Profile", use_container_width=True)
    else:
        st.markdown("👤 **Profile Avatar Place-holder**")
        st.caption("(Tip: Save your picture as 'profile.jpg' in this folder)")
     # 💼 Dynamic Social Anchors
    st.markdown(
        """
        <a class="sidebar-btn" href="https://github.com/bahsulayman689-hash" target="_blank">🐙 GitHub</a>
        <a class="sidebar-btn" href="https://www.linkedin.com/in/sulayman-bah-8a7096423" target="_blank">💼 LinkedIn</a>
        <a class="sidebar-btn" href="http://bahsulayman689@gmail.com">📧 Contact</a>
        """, 
        unsafe_allow_html=True
    )
    # ADDED: Who Am I Section
    st.markdown("### 🧑‍💻 Who Am I")
    st.markdown("""
    * **Name:** Sulayman Bah
    * **Role:** Machine Learning Engineer / Developer
    * **Focus:** Data science, predictive analytics, and building intelligent web applications.
    """)
    
    st.markdown("---")
    
    # ADDED: My Skills Section
    st.markdown("### 🛠️ My Skills")
    st.markdown("""
    * 🐍 **Python Programming**
    * 📊 **Data Science & EDA** (Pandas, NumPy)
    * 🤖 **Machine Learning** (Scikit-Learn)
    * 📉 **Data Visualization** (Seaborn, Matplotlib)
    * 🖥️ **Web App Development** (Streamlit)
    * 🧠 **Model Deployment & Pipelines**
    """)

    st.markdown("---")
    st.markdown("### 📊 App Overview")
    st.write("This intelligence dashboard utilizes a trained standard Decision Tree configuration to evaluate demographic vectors against salary thresholds.")

# --------------------------------------------------------------------------------------------------
# 3. MAIN DASHBOARD CONTENT & PIPELINE LOADER (FIXED & PROTECTED)
# --------------------------------------------------------------------------------------------------
st.title("💰 Census Income Prediction App")
st.write("Enter individual demographic profiles below to execute predictive analytics.")
st.markdown("---")

@st.cache_resource
def load_ml_pipeline():
    try:
        loaded_object = joblib.load("model_assets.pkl")
        return loaded_object
    except FileNotFoundError:
        return "NOT_FOUND"

assets = load_ml_pipeline()

# CRITICAL PROTECTION CHECK: Verify if assets is a dict or a lone model object
if assets == "NOT_FOUND":
    st.error("🚨 **Error:** `model_assets.pkl` not found in your directory!")
    st.stop()
elif isinstance(assets, DecisionTreeClassifier):
    st.error("🚨 **Asset Format Mismatch Detected!**")
    st.warning("Your `model_assets.pkl` only contains the model object. You forgot to re-run your updated training script! Run `python train.py` in your terminal to save the correct dictionary structure.")
    st.stop()

# Extract variables safely once confirmed to be a dictionary structure
model = assets['model']
scaler = assets['scaler']
encoders = assets['encoders']
le_income = assets['le_income']
feature_columns = assets['feature_columns']

# --------------------------------------------------------------------------------------------------
# 4. INTERACTIVE INPUT FORMS
# --------------------------------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Select Age:", min_value=17, max_value=90, value=35)
    workclass = st.selectbox("Workclass Sector:", list(encoders['workclass'].classes_))
    education = st.selectbox("Highest Education Level:", list(encoders['education'].classes_))
    
    marital_status = st.selectbox("Marital Status Category:", [
        'Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 
        'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'
    ])
    
    occupation = st.selectbox("Occupation Specialization:", list(encoders['occupation'].classes_))
    relationship = st.selectbox("Household Relationship Role:", list(encoders['relationship'].classes_))

with col2:
    race = st.selectbox("Race Background:", list(encoders['race'].classes_))
    gender = st.selectbox("Gender Identity:", list(encoders['gender'].classes_))
    
    capital_gain = st.number_input("Capital Gains ($):", min_value=0, value=0, step=100)
    capital_loss = st.number_input("Capital Losses ($):", min_value=0, value=0, step=100)
    
    hours_per_week = st.slider("Weekly Working Hours:", min_value=1, max_value=99, value=40)
    native_country = st.selectbox("Country of Origin:", list(encoders['native-country'].classes_))

# --------------------------------------------------------------------------------------------------
# 5. INFERENCE PROCESSING ENGINE
# --------------------------------------------------------------------------------------------------
st.markdown("---")

if st.button("Evaluate Target Income Group", type="primary", use_container_width=True):
    
    input_dict = {
        'age': age, 'workclass': workclass, 'education': education, 'marital-status': marital_status,
        'occupation': occupation, 'relationship': relationship, 'race': race, 'gender': gender,
        'capital-gain': capital_gain, 'capital-loss': capital_loss, 'hours-per-week': hours_per_week,
        'native-country': native_country
    }
    
    if input_dict['marital-status'] in ['Divorced', 'Never-married', 'Separated', 'Widowed']:
        input_dict['marital-status'] = 'not married'
    elif input_dict['marital-status'] in ['Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent']:
        input_dict['marital-status'] = 'married'
        
    input_df = pd.DataFrame([input_dict])[feature_columns]
    
    for col in encoders.keys():
        input_df[col] = encoders[col].transform(input_df[col])
        
    input_scaled = scaler.transform(input_df)
    
    prediction = model.predict(input_scaled)
    predicted_text_label = le_income.inverse_transform(prediction)
    
    st.subheader("Classification Outcome Profile:")
    if predicted_text_label == ">50K":
        st.success(f"**Predicted Annual Bracket:** `{predicted_text_label}` (High-Income Bracket Candidate) 🚀")
    else:
        st.info(f"**Predicted Annual Bracket:** `{predicted_text_label}` (Standard-Income Bracket Candidate) 💵")
