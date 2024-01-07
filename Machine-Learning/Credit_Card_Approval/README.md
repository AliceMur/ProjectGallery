# Machine Learning Predictions of Credit Card Approval

### This project compares different sklearn library models based on the data collected in a dataset. This is done in order to see how well these models can predict the chance of approval for the given user to get a credit card. The models that I am comparing are:

* 'LR' Logistic Regression Classifier
* 'SVM' Support Vector Machine
* 'KNN' K-Neighbors Classifier
* 'BGC' Bagging Classifier
* 'RF' Random Forest Classifier

Before I trained these models on the clean_dataset.csv dataset, I encoded the dataset using the pandas library (see the approve.py file). I also used the matplotlib and seaborn libraries in order to visualize the data and the model training results (see the same file for details). The resulting scatterplots, bar charts, and violin plot can be seen in the Graphs folder.

Dataset obtained from [here](https://www.kaggle.com/datasets/samuelcortinhas/credit-card-approval-clean-data?select=clean_dataset.csv)