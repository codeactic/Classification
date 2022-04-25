import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
df=pd.read_csv("creditcard.csv")

df.head()  #aleardy innormalized form most of the part  except (time and amount) 
df.describe()  #statistical information
df.info()  #no non-null values

#important part is to check the class in classification problems
sns.countplot(df['Class'])   
#quite unbalanced class from the figure

#checking the data through the dataplots on the data
df_temp=df.drop(columns=['Time','Amount','Class'],axis=1)
index=0

fig,ax=plt.subplots(ncols=4, nrows=7,figsize=(20,30))
ax=ax.flatten()

for col in df_temp.columns:
    sns.displot(df_temp[col],ax=ax[index])
    index+=1
plt.tight_layout(pad=0.5,w_pad=0.5,h_pad=5)

sns.distplot(df['Time'])  #from figure we can see that its quite messy here might need to change the value of the time

sns.distplot(df['Amount'])

df.describe()
#correlation matrix
corr=df.corr()
plt.figure(figsize=(40,20))
sns.heatmap(corr, annot=True, cmap='coolwarm')

#inputs
X=df.drop(columns=['Class'],axis=1)
y=df['Class']

#scaling the time 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_s=sc.fit_transform(X)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,f1_score
X_train ,X_test ,y_train,y_test=train_test_split(X_s,y,test_size=0.25,random_state=42,stratify=y)

from sklearn.linear_model import LogisticRegression
lc=LogisticRegression()
lc.fit(X_train,y_train)
y_pred=lc.predict(X_test)
print(classification_report(y_test,y_pred))#reports or performance measure
print(f1_score(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier
rc=RandomForestClassifier()
rc.fit(X_train,y_train)
y_pred=rc.predict(X_test)
print(classification_report(y_test,y_pred))
print(f1_score(y_test, y_pred))

from xgboost import XGBClassifier
rc=XGBClassifier()
rc.fit(X_train,y_train)
y_pred=rc.predict(X_test)
print(classification_report(y_test,y_pred))
print(f1_score(y_test, y_pred))

#class balancing 
from imblearn.over_sampling import SMOTE
over_sample=SMOTE()
x_smote, y_smote = over_sample.fit_resample(X_train, y_train)
#logistic after balancing the classes
model = LogisticRegression()
model.fit(x_smote, y_smote)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("F1 Score:",f1_score(y_test, y_pred))
#random forest
model = RandomForestClassifier(n_jobs=-1)
model.fit(x_smote, y_smote)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("F1 Score:",f1_score(y_test, y_pred))

#XGBClassifier
model = XGBClassifier(n_jobs=-1)
model.fit(x_smote, y_smote)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("F1 Score:",f1_score(y_test, y_pred))



