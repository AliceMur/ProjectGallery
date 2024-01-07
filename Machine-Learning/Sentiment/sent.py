import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from keras import models
from keras import layers
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from keras.datasets import imdb


(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
print(x_train[0])
print(y_train[0])
class_names = ['Negative', 'Positive']
word_index = imdb.get_word_index()
print(word_index['hello'])
reverse_word_index = dict((value, key) for key, value in word_index.items())

def decode(review):
    text = ''
    for i in review:
        text += reverse_word_index[i]
        text += ' '
    return text
decode(x_train[0])
def show_lengths():
    print('Length of 1st training example: ', len(x_train[0]))
    print('Length of 2nd training example: ',  len(x_train[1]))
    print('Length of 1st test example: ', len(x_test[0]))
    print('Length of 2nd test example: ',  len(x_test[1]))
    
show_lengths()
word_index['the']
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

x_train = pad_sequences(x_train, value = word_index['the'], padding = 'post', maxlen = 256)
x_test = pad_sequences(x_test, value = word_index['the'], padding = 'post', maxlen = 256)
show_lengths()
decode(x_train[0])

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Embedding, Dense, GlobalAveragePooling1D, Dropout

model = Sequential([
    Embedding(10000, 16),
    GlobalAveragePooling1D(),
    Dense(40, activation = 'relu'),
    Dropout(0.1),
    Dense(30, activation = 'relu'),
    Dropout(0.1),
    Dense(24, activation = 'relu'),
    Dropout(0.1),
    Dense(8, activation = 'relu'),
    Dense(1, activation = 'sigmoid')

])

model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['acc']
)

model.summary()

from tensorflow.python.keras.callbacks import LambdaCallback
simple_logging = LambdaCallback(on_epoch_end = lambda e, l: print(e, end='.'))

E = 2

history = model.fit(
    x_train, y_train,
    validation_split = 0.2,
    epochs = E,
    callbacks = [simple_logging],
    verbose = True
)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['Training','Validation'])
plt.title('Training and Validation Accuracy')
plt.xlabel('epoch')
plt.show()

plt.figure(figsize=(8, 8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training','Validation'])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

loss, acc = model.evaluate(x_test,y_test)
print('Test set accuracy: ', acc * 100)

import numpy as np

p=model.predict(np.expand_dims(x_test[0], axis=0))
print(class_names[np.argmax(p[0])])

decode(x_test[0])
