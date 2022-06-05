import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import graph_objs as go
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import GradientBoostingClassifier

def Bmi(val):
	a = []
	if val <= 18.5:
		a = [0,0,1]
	elif val>18.5 and val<=24.9:
		a = [0,0,0]
	elif val>24.9 and val<=29.9:
		a = [0,1,0]
	else:
		a = [1,0,0]

	return a

def blood_pressure(val):
	a = []
	if val < 80.0:
		a = [0, 1]
	elif val>=80.0 and val<=89.0:
		a = [0, 0]
	else:
		a = [1, 0]

	return a

def glucose(val):
	a = []
	if val<140.0:
		a = [0,1]
	elif val>=140.0 and val<=199.0:
		a = [0,0]
	else:
		a = [1,0]

	return a


data = pd.read_csv("data//Diabetes_Processed.csv")
df2 = data.copy()
df2 = df2[~(df2['BloodPressure']<30)]
df2 = df2[~(df2['SkinThickness']==99)]
df2 = df2[~(df2['BMI']==67.1)]
df2 = df2[~(df2['Insulin']>600)]
##BMI FEATURE ENGINEERING
NewBMI = pd.Series(["Underweight", "Normal", "Overweight", "Obesity"], dtype = "category")
df2["NewBMI"] = NewBMI
df2.loc[df2["BMI"] <= 18.5, "NewBMI"] = NewBMI[0]
df2.loc[(df2["BMI"] > 18.5) & (df2["BMI"] <= 24.9), "NewBMI"] = NewBMI[1]
df2.loc[(df2["BMI"] > 24.9) & (df2["BMI"] <= 29.9), "NewBMI"] = NewBMI[2]
df2.loc[(df2["BMI"] > 29.9), "NewBMI"] = NewBMI[3]

##BLOOD PRESSURE FEATURE ENGINEERING
NewBP = pd.Series(["Normal", "AtRisk", "HighBloodPressure"], dtype = "category")
df2["NewBP"] = NewBP
df2.loc[df2["BloodPressure"] < 80, "NewBP"] = NewBP[0]
df2.loc[(df2["BloodPressure"] >=80) & (df2["BloodPressure"] <= 89), "NewBP"] = NewBP[1]
df2.loc[(df2["BloodPressure"] >= 90), "NewBP"] = NewBP[2]

##GLUCOSE FEATURE ENGINEERING
NewGlucose = pd.Series(["Normal", "AtRisk", "MayBe"], dtype = "category")
df2["NewGlucose"] = NewGlucose
df2.loc[df2["Glucose"] < 140.0, "NewGlucose"] = NewGlucose[0]
df2.loc[(df2["Glucose"] >=140.0) & (df2["Glucose"] <= 199.0), "NewGlucose"] = NewGlucose[1]
df2.loc[(df2["Glucose"] >= 200.0), "NewGlucose"] = NewGlucose[2]


df2 = pd.get_dummies(df2, columns =["NewBMI","NewBP", "NewGlucose"], drop_first = True)
y = df2['Outcome']
X = df2.drop(["Outcome", "NewBMI_Obesity", "NewBMI_Overweight", "NewBMI_Underweight", "NewBP_HighBloodPressure", "NewBP_Normal", "NewGlucose_MayBe", "NewGlucose_Normal"], axis=1)
cols = X.columns
index = X.index
scaler = RobustScaler()
scaler.fit(X)
X = scaler.transform(X)
X = pd.DataFrame(X, columns = cols, index = index)
categorical_df = df2[["NewBMI_Obesity", "NewBMI_Overweight", "NewBMI_Underweight", "NewBP_HighBloodPressure", "NewBP_Normal", "NewGlucose_MayBe", "NewGlucose_Normal"]]
X = pd.concat([X,categorical_df], axis = 1)

clf = GradientBoostingClassifier(subsample=0.5, random_state=42, learning_rate=0.2, n_estimators=500, min_samples_split = 6, max_depth=8)
clf.fit(X, y)

st.title("Diabetes Predictor")
st.image("data//diabetes.jpg", width = 900)
nav = st.sidebar.radio("Navigation",["Home","Prediction","Contribute"])
st.set_option('deprecation.showPyplotGlobalUse', False)
if nav == 'Home':
	if st.checkbox("Show Table"):
		st.table(data)
    
	graph = st.selectbox("What kind of Graph ? ",["Non-Interactive","Interactive"])
	if graph =="Non-Interactive":
		st.subheader("We can have a look at data distribution with Histograms, Just select the variable for which you want distribution")
		var = st.selectbox("Select the Feature/Variable",["Pregnancies","Glucose","BloodPressure", "SkinThickness", "BMI", "Insulin", "Age"])
		sns.set_style("whitegrid")
		sns.histplot(data = df2, x = var, hue = 'Outcome',kde = True)
		plt.tight_layout()
		st.pyplot()
	if graph == "Interactive":
		var = st.selectbox("Select the Feature/Variable",["Pregnancies","Glucose","BloodPressure", "SkinThickness", "BMI", "Insulin", "Age"])
		fig = go.Figure(data=go.Histogram(x=df2[var]))
		st.plotly_chart(fig)


if nav == "Prediction":
	st.header("Are You at Risk?")
	box = []
	b = []
	val1 = st.number_input("Number of Times You have been Pregnant",0,18,step = 1)
	box.append(val1)
	val2 = st.number_input("Plasma glucose concentration a 2 hours in an oral glucose tolerance test",0.000,300.000)
	box.append(val2)
	val3 = st.number_input("Diastolic blood pressure (mm Hg)",40)
	box.append(val2)
	val4 = st.number_input("Triceps skin fold thickness (mm)", 7.000, 70.00, step = 0.01)
	box.append(val4)
	val5 = st.number_input("2-Hour serum insulin (mu U/ml)", 10.00, 500.00, step=0.01)
	box.append(val5)
	val6 = st.number_input("Body mass index (weight in kg/(height in m)^2)", 18.00, step=0.01)
	box.append(val6)
	val7 = st.number_input("Diabetes pedigree function",0.00)
	box.append(val7)
	val8 = st.number_input("Your Age in Years", 0, step = 1)
	box.append(val8)

	sample = np.array(box)
	sample = sample.reshape(-1, 1)
	sample = sample.T
	sample = scaler.transform(sample)

	for x in Bmi(val6):
		b.append(x)
	for x in blood_pressure(val3):
		b.append(x)
	for x in glucose(val2):
		b.append(x)

	b = np.array(b)
	b = b.reshape(-1, 1)
	b = b.T

	fsample = np.hstack((sample, b))

	pred = clf.predict(fsample)



	if st.button("Predict"):
		if pred[0]==1:
			st.warning(f"You are probably at a Risk of having diabetes")
		else:
			st.success("Yayy!! You are safe")


if nav=="Contribute":
	st.header("Contribute Your Test Data")
	box = []
	val1 = st.number_input("Number of Times You have been Pregnant",0,step = 1)
	box.append(val1)
	val2 = st.number_input("Plasma glucose concentration a 2 hours in an oral glucose tolerance test",0)
	box.append(val2)
	val3 = st.number_input("Diastolic blood pressure (mm Hg)",40.00)
	box.append(val2)
	val4 = st.number_input("Triceps skin fold thickness (mm)", 7, step = 1)
	box.append(val4)
	val5 = st.number_input("2-Hour serum insulin (mu U/ml)", 10.00, step=0.01)
	box.append(val5)
	val6 = st.number_input("Body mass index (weight in kg/(height in m)^2)", 18.00, step=0.01)
	box.append(val6)
	val7 = st.number_input("Diabetes pedigree function",0.00)
	box.append(val7)
	val8 = st.number_input("Your Age in Years", 0, step = 1)
	box.append(val8)
	val9 = st.selectbox("Were You Diabetic or Not?", ["Yes", "No"], index=1)
	if val9 == "Yes":
		box.append(1)
	else:
		box.append(0)

	if st.button("Submit"):
		to_add = {}
		for index, col in enumerate(data.columns):
			to_add[col] = [box[index]]
		st.write(to_add)
		to_add = pd.DataFrame(to_add)
		to_add.to_csv("data//Diabetes_Processed.csv",mode='a',header = False,index= False)
		st.success("Submitted")

