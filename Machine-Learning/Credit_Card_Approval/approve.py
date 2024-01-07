import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
#add check
tf.compat.v1.logging.set_verbosity(tf._logging.ERROR)
sns.set_theme()
print('Libraries imported.')
sns.__version__
# Load dataset
df = pd.read_csv('clean_dataset.csv')
print(df.shape)
# Obtain dataframe info
df.info()

# Obtain the statistical summary of the dataframe
df.describe()

# For better visualization
df.describe().T

# See how many null values exist in the dataframe
df.isnull().sum()

"""Visualization"""
# employment status vs age
plt.figure(figsize=(10,8))
sns.violinplot(data=df, x='Employed',y='Age', hue = 'Gender', palette='plasma')
plt.legend(title='Gender', labels=['Female', 'Male'])
plt.xticks([0,1],['Not Employed','Employed'])
plt.title('Ratio of Gender vs Age vs Employed')
plt.show()

# bank staus credit score gender debt
plt.figure(figsize=(10,8))
sns.violinplot(data=df, x='BankCustomer',y='CreditScore', hue = 'Gender', palette='Spectral')
plt.legend(title='Gender', labels=['Female', 'Male'])
plt.xticks([0,1],['No Bank Account','Have Bank Account'])
plt.title('Ratio of Gender vs Credit vs Bank Account')
plt.show()

# Approval
plt.figure(figsize=(6, 6))
fraud_proportion = df[df['Approved'] == 1]['Gender'].value_counts()
labels = ['Female','Male']
plt.figure(figsize=(6, 6))
plt.pie(fraud_proportion, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Proportion of Approval by Gender')
plt.show()

# Income vs debt (catplot)
plt.figure(figsize=(10,10))
sns.scatterplot(data=df, y='Income', x='Debt',hue='Income', palette="gist_ncar")
plt.title('Scatter plot Income vs Debt')
plt.legend(title='Income')
plt.ioff()
plt.show()

# PriorDefault vs Industry (scatter)
plt.figure(figsize=(20,8))
sns.countplot(data=df, x='Industry', hue = 'PriorDefault', palette='gist_ncar')
plt.title('Scatter plot Industry Prior default')
plt.legend(title='Income')
plt.ioff()
plt.show()

"""Model Prep"""

# Specify model input features (all data except for the target variable) 
x = df.drop('Approved',axis=1)

# Model output (target variable)
y = df['Approved']
y

from tensorflow.keras.utils import to_categorical
#y = to_categorical(y)
#y

# scale the data before training the model (normalization)
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
encoder = {}
for i in x.select_dtypes('object').columns:
    encoder[i] = LabelEncoder()
    x[i] = encoder[i].fit_transform(x[i])
#scaler_x = StandardScaler()
#X = scaler_x.fit_transform(X)

# spliting the data into testing and training sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=42)

# print the shapes
x_train.shape, x_test.shape, y_train.shape, y_test.shape

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test  = sc.fit_transform(x_test)

#print(X.head())

"""Model Build"""
"""
* 'LR' Logistic Regression Classifier
* 'SVM' Support Vector Machine
* 'KNN' K-Neighbors Classifier
* 'BGC' Bagging Classifier
* 'RF' Random Forest Classifier
"""
#Prepare a Logistic Regression Classifier Model
from sklearn.linear_model import LogisticRegression
reg=LogisticRegression(random_state=42)
reg.fit(x_train,y_train)
y_pred_rl=reg.predict(x_test)
rl = accuracy_score(y_test,y_pred_rl)*100
print(rl)

#Prepare a Support Vector Machine
from sklearn import svm
svm = svm.SVC(kernel='linear',C = 0.01)
svm.fit(x_train,y_train)
y_pred_svm = svm.predict(x_test)
sv = accuracy_score(y_test,y_pred_svm)*100
print(sv)

#Prepare a K-Neighbors Classifier Model
from sklearn.neighbors import KNeighborsClassifier
reg=KNeighborsClassifier(n_neighbors=20)
reg.fit(x_train,y_train)
y_pred_knn=reg.predict(x_test)
kn = accuracy_score(y_test,y_pred_knn)*100
print(kn)

#Prepare a Bagging Classifier Model
from sklearn.ensemble import BaggingClassifier
knn = KNeighborsClassifier(n_neighbors=20)
bgc = BaggingClassifier(base_estimator= knn, n_estimators=5, random_state=42)
bgc.fit(x_train, y_train)
y_pred_bgc = bgc.predict(x_test)
bg = accuracy_score(y_test,y_pred_bgc)*100
print(bg)

#Prepare a Random Forest Classifier Model
from sklearn.ensemble import RandomForestClassifier
y_rfm = RandomForestClassifier(n_estimators=100,random_state=42,max_features=15)
y_rfm.fit(x_train, y_train)
rfm = y_rfm.predict(x_test)
r = accuracy_score(y_test, rfm)*100
print(r)

plt.rcParams["figure.figsize"] = [10, 8]
plt.rcParams["figure.autolayout"] = True

x=['Logistic Regression','Support Vector Machine','K-Neighbors Regressor','Bagging Classifier','Random Forest Classifier']
y=[rl,sv, kn, bg, r]

width = 0.75
fig, ax = plt.subplots()
color = ['indigo', 'teal', 'olive', 'orangered', 'maroon']
pps = ax.bar(x, y, width, align='center', color =color)

for p in pps:
   height = p.get_height()
   ax.text(x=p.get_x() + p.get_width() / 2, y=height+1,
      s="{}%".format(height),
      ha='center')
plt.title('Accuracy of models')
plt.show()
