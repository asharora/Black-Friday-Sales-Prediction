import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv(r"C:\Users\BHupesh\Practice notebooks\BlackFriday.csv")
df["Product_Category_2"].fillna(value=int(df["Product_Category_2"].mean()),inplace=True)
df["Product_Category_3"].fillna(value=int(df["Product_Category_3"].mean()),inplace=True)
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
df["Gender"]=encoder.fit_transform(df["Gender"])
df["Age"]=encoder.fit_transform(df["Age"])
df["Occupation"]=encoder.fit_transform(df["Occupation"])
df["City_Category"]=encoder.fit_transform(df["City_Category"])
df["Stay_In_Current_City_Years"]=encoder.fit_transform(df["Stay_In_Current_City_Years"])
df["Marital_Status"]=encoder.fit_transform(df["Marital_Status"])
df["Product_Category_1"]=encoder.fit_transform(df["Product_Category_1"])
df["Product_Category_2"]=encoder.fit_transform(df["Product_Category_2"])
df["Product_Category_3"]=encoder.fit_transform(df["Product_Category_3"])
df["Purchase"]=encoder.fit_transform(df["Purchase"])
df["Product_ID"]=encoder.fit_transform(df["Product_ID"])
x=df.iloc[:,0:10]
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler
x_train=StandardScaler().fit_transform(x_train)
scaledData=pd.DataFrame(x_train)
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(x_train, y_train)
print(regressor.predict(x_test))

