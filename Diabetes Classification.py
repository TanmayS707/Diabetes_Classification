#!/usr/bin/env python
# coding: utf-8

# Importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Reading the dataset

df=pd.read_csv("diabetes-data.csv")


df.head()


df.info(verbose=True)


df.describe()


# In[7]:


df_copy=df.copy(deep=True)
df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] =df_copy[['Glucose','SkinThickness','BloodPressure','Insulin','BMI']].replace(0,np.NaN)


# In[8]:


print(df_copy.isnull().sum())



hplot=df.hist(figsize=(20,20))


df_copy['Glucose'].fillna(df['Glucose'].mean(),inplace=True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(),inplace=True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(),inplace=True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median(),inplace=True)
df_copy['BMI'].fillna(df_copy['BMI'].median(),inplace=True)
hplot=df_copy.hist(figsize=(20,20))


print(df.Outcome.value_counts())




df.Outcome.value_counts().plot(kind="bar")




from pandas.plotting import scatter_matrix
p=scatter_matrix(df,figsize=(25,25))




p=sns.pairplot(df_copy,hue='Outcome')




plt.figure(figsize=(12,10))
p=sns.heatmap(df_copy.corr(),annot=True,cmap='RdYlGn')



from sklearn.preprocessing import StandardScaler
scale_X=StandardScaler()
X=scale_X.fit_transform(df_copy.drop(["Outcome"],axis=1))
X=pd.DataFrame(X,columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])





X.head()




from sklearn.preprocessing import MinMaxScaler
scale_X=MinMaxScaler()
X=scale_X.fit_transform(df_copy.drop(["Outcome"],axis=1))
X=pd.DataFrame(X,columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])





X.head()




y=df_copy.Outcome
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=1/3,random_state=42,stratify=y)




from sklearn.neighbors import KNeighborsClassifier
testing_score=[]
training_score=[]
for i in range(1,100):
    knn=KNeighborsClassifier(i)
    knn.fit(X_train,Y_train)
    training_score.append(knn.score(X_train,Y_train))
    testing_score.append(knn.score(X_test,Y_test))





max_training_score=max(training_score)
train_scores_ind=[i for i,v in enumerate(training_score) if v==max_training_score]
print('Max training score {} % and k={}'.format(max_training_score*100,list(map(lambda x: x+1,train_scores_ind))))




max_testing_score=max(testing_score)
test_scores_ind=[i for i,v in enumerate(testing_score) if v==max_testing_score]
print('Max testing score {} % and k={}'.format(max_testing_score*100,list(map(lambda x: x+1,test_scores_ind))))





plt.figure(figsize=(12,5))
pplot = sns.lineplot(range(1,100),training_score,marker='*',label='Training Score')
pplot=sns.lineplot(range(1,100),testing_score,marker='o',label='Testing Score')







