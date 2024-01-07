# Machine Learning Predictions of 2019 Airbnb Prices

### This project compares different sklearn library models in order to see how well they can predict the price of NYC Airbnbs based on the data collected from 2019. The models that I am comparing are:

* 'LR' Linear Regression
* 'KNN' K-Neighbors Regressor
* 'DTR' Decision Tree Regressor
* 'RF' Random Forest Regressor
* 'GBM' Gradient Boosting Regressor

Before I trained these models on the AB_NYC_2019.csv dataset, I encoded the dataset using the pandas library (see the check.py file). I also used the matplotlib and seaborn libraries in order to visualize the data and the model training results (see the check.py and utils.py files, respectively). The resulting scatterplots, bar charts, and violin plot can be seen in the Graphs folder.

Dataset obtained from [here](https://www.kaggle.com/datasets/ebrahimelgazar/new-york-city-airbnb-market)