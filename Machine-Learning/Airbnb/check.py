import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

from utils import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import *  
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score
tf.compat.v1.logging.set_verbosity(tf._logging.ERROR)

print('Libraries imported.')

airbnb= pd.read_csv('AB_NYC_2019.csv')
airbnb.duplicated().sum()
airbnb.drop_duplicates(inplace=True)
airbnb.isnull().sum()
airbnb.drop(['name','id','host_name','last_review'], axis=1, inplace=True)
airbnb.fillna({'reviews_per_month':0}, inplace=True)
#examining changes
airbnb.reviews_per_month.isnull().sum()
airbnb.isnull().sum()
airbnb.dropna(how='any',inplace=True)

"""Visualization"""

#~~~Graphs~~~

#price for neighbourhood_group
#creating a sub-dataframe with no extreme values (less than 500)
sub=airbnb[airbnb.price < 500]
plt.figure(figsize=(10,10))
sns.violinplot(data=sub, x='neighbourhood_group',y='price', hue = 'neighbourhood_group', palette='plasma')
plt.title('Distribution of Prices in NYC up to $500 by Borough')
plt.ylabel('Price')
plt.xlabel('Borough')
plt.show()

#airbnb in each borough
plt.figure(figsize=(10,10))
sns.countplot(data=airbnb, x='neighbourhood_group', hue = 'neighbourhood_group', palette="plasma")
plt.title('Distribution of Airbnb in the 5 Borough')
plt.ylabel('Count')
plt.xlabel('Borough')
plt.show()

#~~~NYC map~~~

#plot price
plt.figure(figsize=(10,6))
sns.scatterplot(data=sub, x='longitude', y='latitude',hue='price', palette="gist_ncar")
plt.title('Distribution of Prices in NYC up to $500')
plt.legend(title='Price')
plt.ioff()
plt.show()

#plot type
plt.figure(figsize=(10,6))
sns.scatterplot(data=airbnb, x='longitude', y='latitude',hue='room_type', palette="brg")
plt.title('Distribution of Room Type in NYC')
plt.legend(title='Room Type')
plt.ioff()
plt.show()

#minimum nights
plt.figure(figsize=(10,6))
ax = sns.scatterplot(data=airbnb, x='longitude', y='latitude',hue='neighbourhood', palette="gist_ncar")
plt.title('Showing the Neighbourhoods that have Airbnb')
plt.legend(title='Neighbourhood')
plt.ioff()
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.show()

"""Model Prep"""
airbnb.drop(['host_id','latitude','longitude','neighbourhood','number_of_reviews','reviews_per_month'], axis=1, inplace=True)

#Encode the input Variables
def Encode(airbnb):
    for column in airbnb.columns[airbnb.columns.isin(['neighbourhood_group', 'room_type'])]:
        airbnb[column] = airbnb[column].factorize()[0]
    return airbnb

airbnb_en = Encode(airbnb.copy())
x = airbnb_en.iloc[:,[0,1,3,4,5]]
y = airbnb_en['price']

"""Models"""
"""
* 'LR' Linear Regression
* 'KNN' K-Neighbors Regressor
* 'DTR' Decision Tree Regressor
* 'RF' Random Forest Regressor
* 'GBM' Gradient Boosting Regressor
"""
#Getting Test and Training Set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=353)
x_train.head()
y_train.head()

#Prepare a Linear Regression Model

reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred_rl=reg.predict(x_test)
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred_rl))

#Prepare a K-Neighbors Regressor Model
from sklearn.neighbors import KNeighborsRegressor
reg=KNeighborsRegressor()
reg.fit(x_train,y_train)
y_pred_knn=reg.predict(x_test)
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred_knn))

#Prepare a DecisionTreeRegressor Model
from sklearn.tree import DecisionTreeRegressor
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=105)
DTree=DecisionTreeRegressor(min_samples_leaf=.0001)
DTree.fit(x_train,y_train)
y_pred_dt=DTree.predict(x_test)

#Prepare a Random Forest Regressor Model
from sklearn.ensemble import RandomForestRegressor
random_forest_model = RandomForestRegressor(n_estimators=30)
random_forest_model.fit(x_train, y_train)
r_squared_rf = random_forest_model.score(x_train, y_train)

#Prepare a Gradient Boosting Regressor Model
from sklearn.ensemble import GradientBoostingRegressor
gradient_boosting_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
gradient_boosting_model.fit(x_train, y_train)
y_pred_gb = gradient_boosting_model.predict(x_test)
r2_gb = r2_score(y_test, y_pred_gb)

# Predict using Random Forest model
y_pred_rf = random_forest_model.predict(x_test)
from sklearn.metrics import r2_score

r2_score(y_test, y_pred_rf)
compare_predictions(y_pred_rl, y_pred_knn, y_pred_dt, y_pred_rf ,y_pred_gb , y_test)
pm = [y_pred_rl, y_pred_knn, y_pred_dt, y_pred_rf ,y_pred_gb]
rmse_scores = []
import numpy as np 
from sklearn import metrics
for p in pm:
    rm = np.sqrt(metrics.mean_squared_error(y_test, p))
    rmse_scores.append(rm)
rmse(rmse_scores)
