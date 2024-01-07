import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf._logging.ERROR)
print('Using Tesorflow version', tf.__version__)
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
# check use
from emnist import extract_test_samples
from emnist import extract_training_samples
(x_train, y_train) = extract_training_samples('letters')
(x_test, y_test) = extract_test_samples('letters')

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

"""Visualizations"""

plt.imshow(x_train[0], cmap='binary')
plt.show()
y_train[0]
print(set(y_train))

"""Model"""
#encoding labels
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)
#display encoded labels
y_train_encoded[0]

#Neural Networks

#check size
x_train_reshaped = np.reshape(x_train, (124800, 784))
x_test_reshaped = np.reshape(x_test, (20800, 784))
x_mean = np.mean(x_train_reshaped)
x_std = np.std(x_train_reshaped)

epsilon = 1e-10
x_train_norm = (x_train_reshaped - x_mean)/(x_std + epsilon)
x_test_norm = (x_test_reshaped - x_mean)/(x_std + epsilon)

model = Sequential([
    Dense(128, activation='relu',  input_shape=(784,)),
    Dense(128, activation='relu'),
    Dense(27, activation='softmax')
])


#Compiling the model
model.compile(
    optimizer='adam',
    loss = 'categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

#training model
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)
model.fit(x_train_norm, y_train_encoded, epochs=100, batch_size=32, validation_data=(x_train_norm, y_train_encoded), callbacks=early_stopping_callback)
#evaluating
accuracy = model.evaluate(x_test_norm, y_test_encoded)
print('Test set accuacy:', accuracy * 100)

#predictions
preds= model.predict(x_test_norm)
print('Shape of preds:', preds.shape)

plt.figure(figsize=(12,12))

start_index = 0

for i in range(25):
    plt.subplot(5,5, i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    pred = np.argmax(preds[start_index+1])
    gt = y_test[start_index+i]

    col ='g'
    if pred != gt:
        col = 'r'

    plt.xlabel('i ={}, pred={}, gt={}'.format(start_index+i, pred, gt), color = col)
    plt.imshow(x_test[start_index+i], cmap='binary')  
plt.show() 