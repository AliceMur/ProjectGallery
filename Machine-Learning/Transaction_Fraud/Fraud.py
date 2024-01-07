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
# Add check
tf.compat.v1.logging.set_verbosity(tf._logging.ERROR)
sns.set_theme()
print('Libraries imported.')
sns.__version__
# Load dataset
df = pd.read_csv('Fraud.csv')
print(df.shape)
bank_df = df.sample(n=60000, random_state=0)
print(bank_df.shape)
# Obtain dataframe info
bank_df.info()

# Obtain the statistical summary of the dataframe
bank_df.describe()

# For better visualization
bank_df.describe().T

# See how many null values exist in the dataframe
bank_df.isnull().sum()

"""Visualization"""

plt.figure(figsize = (20, 10))
sns.countplot(data = bank_df, x='type', palette='plasma')
plt.xlabel('Transaction Type')
plt.ylabel('Count')
plt.title('Distribution of Transaction Types')
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(data=bank_df, x='type', y='amount', palette='viridis')
plt.xticks(rotation=0)  
plt.xlabel('Transaction Type')
plt.ylabel('Transaction Amount')
plt.title('Transaction Amount by Transfer Type')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=bank_df, x='amount', y='step', hue='isFraud', alpha=0.5)
plt.xlabel('Hours')
plt.ylabel('Amount')
plt.title('Scatter Plot of Hours vs. Size of Transaction with Fraudulent Status (based on color)')
plt.grid(True)
plt.tight_layout()
plt.legend(title='Is Fraud')
plt.show()

fraud_proportion = bank_df[bank_df['isFraud'] == 1]['type'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(fraud_proportion, labels=fraud_proportion.index, autopct='%1.1f%%', startangle=140)
plt.title('Proportion of Fraudulent Transactions by Transaction Type')
plt.show()

# Correlation plot
plt.figure(figsize = (20, 20))
cm = bank_df.corr()
sns.heatmap(cm,annot = True)
plt.show()

"""Model Prep"""
# Specify model input features (all data except for the target variable) 
#X = bank_df.drop(columns = ['isFraud', 'isFlaggedFraud','step','nameOrig','nameDest'])
X = bank_df

# Model output (target variable)
y = bank_df['isFraud']
y

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
y

# scale the data before training the model (normalization)
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
encoder = {}
for i in X.select_dtypes('object').columns:
    encoder[i] = LabelEncoder()
    X[i] = encoder[i].fit_transform(X[i])
#scaler_x = StandardScaler()
#X = scaler_x.fit_transform(X)

# splitting the data into testing and training sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.3)

# print the shapes
X_train.shape, X_test.shape, y_train.shape, y_test.shape

#print(X.head())

"""Model Build"""
# Create keras sequential model
ANN_model = keras.Sequential()

# Add dense layer
ANN_model.add(Dense(250, input_dim = 11, kernel_initializer = 'normal',activation = 'relu'))
ANN_model.add(Dropout(0.3))
ANN_model.add(Dense(500, activation = 'relu'))

ANN_model.add(Dropout(0.3))
ANN_model.add(Dense(500, activation = 'relu'))

ANN_model.add(Dropout(0.3))
ANN_model.add(Dense(500, activation = 'relu'))

ANN_model.add(Dropout(0.4))
ANN_model.add(Dense(250, activation = 'linear'))

ANN_model.add(Dropout(0.4))
# Add dense layer with softmax activation
ANN_model.add(Dense(2, activation = 'softmax'))
ANN_model.summary()


# Compile the model
ANN_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

history = ANN_model.fit(X_train, y_train, epochs = 10, validation_split = 0.2, verbose = 1)

# Plot the model performance across epochs
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss','val_loss'], loc = 'upper right')
plt.show()

#increase epochs to see where it starts to overfit

# Make predictions
predictions = ANN_model.predict(X_test)
predict = []
for i in predictions:
    predict.append(np.argmax(i))

# Get the accuracy of the model
result = ANN_model.evaluate(X_test, y_test)

print("Accuracy : {}".format(result[1]))

# Get the original values
y_original = []

for i in y_test:
    y_original.append(np.argmax(i))

# Plot Confusion Matrix
# dark values are what it missed

confusion_matrix = metrics.confusion_matrix(y_original, predict)
sns.heatmap(confusion_matrix, annot = True)
plt.show()
# Print out the classification report
from sklearn.metrics import classification_report
print