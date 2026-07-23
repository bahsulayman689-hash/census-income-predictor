#-----------------------------------------------import the requirements dependencies--------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score, mean_squared_error

#-------------------printing the data------------------------------------------------------
data = pd.read_csv("adult.csv")

# CRITICAL FIX: Strip all hidden leading/trailing spaces from string columns in the dataset
for col in data.select_dtypes(include=['object']).columns:
    data[col] = data[col].str.strip()

#---------------------checking the first five rows of the data-----------------------------------------
print(data.head())
#-----------------------checking the shape of the data-----------------------------------------------
print(data.shape)

#----------------------handling the missing values------------------------------------------------------
data.replace("?", np.nan, inplace=True)
data.fillna(data.mode().iloc[0], inplace=True)

#------------------simplify marital status-----------------------------------------------------------------
data.replace(['Divorced', 'Married-AF-spouse', 'Married-civ-spouse',
            'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'],
           ['divorced', 'married', 'married', 'married',
            'not married', 'not married', 'not married'], inplace=True)

#--------------- Categorical columns------------------------------------------------------------------------
category_col = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                'relationship', 'gender', 'native-country']

#---------------------label encode the string values to numerical------------------------------------
encoders = {}
mapping_dict = {}
for col in category_col:
    la = LabelEncoder()
    data[col] = la.fit_transform(data[col])
    encoders[col] = la  # Saved to map prediction system inputs
    mapping_dict[col] = dict(enumerate(la.classes_))
print(mapping_dict)

#--------------------------------------------mapping the income to numerical values------------------------------
le_income = LabelEncoder()
data["income"] = le_income.fit_transform(data["income"])

#-------------------------drop the unnecessary columns-------------------------------------------------------------
X = data.drop(["fnlwgt", "educational-num", 'income'], axis=1)
Y = data['income']

#---------------------------split the train and test------------------------------------------------------------------
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

#---------------------building the model-----------------
model = DecisionTreeClassifier(criterion='gini', random_state=42)
model.fit(X_train_scaler, Y_train)

#-------------------------testing_predictions---------------------------------------------
Y_testing_prediction = model.predict(X_test_scaler)
print(f"Test Accuracy: {accuracy_score(Y_test, Y_testing_prediction)}")
print('='*60)

#---------------------------------------building a predictive system to test the model----------------------------
# 1. Raw text sample matching original data format (with spaces or without, it will be stripped cleanly now)
input_data = "39, State-gov, Bachelors, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States"

# 2. Extract clean strings from commas and strip extra spaces
raw_features = [item.strip() for item in input_data.split(',')]

# 3. Form a dictionary utilizing original features
input_dict = {
    'age': int(raw_features[0]),
    'workclass': raw_features[1],
    'education': raw_features[2],
    'marital-status': raw_features[3],
    'occupation': raw_features[4],
    'relationship': raw_features[5],
    'race': raw_features[6],
    'gender': raw_features[7],
    'capital-gain': int(raw_features[8]),
    'capital-loss': int(raw_features[9]),
    'hours-per-week': int(raw_features[10]),
    'native-country': raw_features[11]
}

# 4. Standardize text transformations using target engineering mappings 
if input_dict['marital-status'] in ['Divorced', 'Never-married', 'Separated', 'Widowed']:
    input_dict['marital-status'] = 'not married'
elif input_dict['marital-status'] in ['Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent']:
    input_dict['marital-status'] = 'married'

# 5. Represent entry array as DataFrame matching X's feature scheme sequence
input_df = pd.DataFrame([input_dict])[X.columns]

# 6. Apply saved label encoder targets to matching categorical inputs
for col in category_col:
    input_df[col] = encoders[col].transform(input_df[col])

# 7. Apply standard scaling step using trained scaling parameters
input_scaled = scaler.transform(input_df)

# 8. Complete target classification prediction
predictions = model.predict(input_scaled)
print(f"Numerical Prediction array output: {predictions}")

# 9. Decode output class to original representation text label
predicted_income_label = le_income.inverse_transform(predictions)
print(f"Predicted Income Target Class: {predicted_income_label}")

# --------------------------------------------------------------------------------------------------
# SAVE ALL ASSETS TOGETHER (FIXED)
# --------------------------------------------------------------------------------------------------
model_assets = {
    'model': model,
    'scaler': scaler,
    'encoders': encoders,
    'le_income': le_income,
    'feature_columns': list(X.columns)
}

# Export everything into a single dictionary file
joblib.dump(model_assets, "model_assets.pkl")
print("All machine learning pipeline assets successfully exported to model_assets.pkl!")
