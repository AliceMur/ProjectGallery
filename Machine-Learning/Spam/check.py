import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding, Dropout


import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

df = pd.read_csv('spam_or_not_spam.csv')
df.head()

df.label.value_counts()

df.info()

plt.figure(figsize=(14,6))
sns.set_style('darkgrid')
sns.countplot(x='label',data=df, hue = 'label', palette='gist_ncar')
plt.legend(title='label', labels=['Not Spam', 'Spam'])
plt.xticks([0,1],['Not Spam','Spam'])
plt.title('Number of Spam and Non-spam')
plt.show()

df.dropna(inplace=True)

def tokenizer_sequences(num_words, X):
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    return tokenizer, sequences

    
max_words = 10000 
maxlen = 300

tokenizer, sequences = tokenizer_sequences(max_words, df.email.copy())

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X = pad_sequences(sequences, maxlen=maxlen)

y = df.label.copy()

print('Shape of data tensor:', X.shape)
print('Shape of label tensor:', y.shape)

max_words = len(tokenizer.word_index) + 1

embed_dim = 100 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train.shape, y_train.shape

model = Sequential()
model.add(Embedding(max_words, embed_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(20, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs = 5, validation_split = 0.2, verbose = 1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss','val_loss'], loc = 'upper right')
plt.show()

result = model.evaluate(X_test, y_test)

print("Accuracy : {}".format(result[1]))