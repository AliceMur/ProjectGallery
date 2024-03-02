import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from normalizing import *
import matplotlib as plt
from sklearn.metrics import accuracy_score

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

#ecobee thermostate
benign=pd.read_csv('input/nbaiot-dataset/2.benign.csv')
g_c=pd.read_csv('input/nbaiot-dataset/2.gafgyt.combo.csv')
g_j=pd.read_csv('input/nbaiot-dataset/2.gafgyt.junk.csv')
g_s=pd.read_csv('input/nbaiot-dataset/2.gafgyt.scan.csv')
g_t=pd.read_csv('input/nbaiot-dataset/2.gafgyt.tcp.csv')
g_u=pd.read_csv('input/nbaiot-dataset/2.gafgyt.udp.csv')
m_a=pd.read_csv('input/nbaiot-dataset/2.mirai.ack.csv')
m_sc=pd.read_csv('input/nbaiot-dataset/2.mirai.scan.csv')
m_sy=pd.read_csv('input/nbaiot-dataset/2.mirai.syn.csv')
m_u=pd.read_csv('input/nbaiot-dataset/2.mirai.udp.csv')
m_u_p=pd.read_csv('input/nbaiot-dataset/2.mirai.udpplain.csv')

train_data, labels = norm(benign, g_c, g_j, g_s, g_t, g_u, m_a, m_sc, m_sy, m_u, m_u_p)

# test/train split  25% test
x_train, x_test, y_train, y_test = train_test_split(
    train_data, labels, test_size=0.25, random_state=42)

# baby monitor
benign2=pd.read_csv('input/nbaiot-dataset/4.benign.csv')
g_c2=pd.read_csv('input/nbaiot-dataset/4.gafgyt.combo.csv')
g_j2=pd.read_csv('input/nbaiot-dataset/4.gafgyt.junk.csv')
g_s2=pd.read_csv('input/nbaiot-dataset/4.gafgyt.scan.csv')
g_t2=pd.read_csv('input/nbaiot-dataset/4.gafgyt.tcp.csv')
g_u2=pd.read_csv('input/nbaiot-dataset/4.gafgyt.udp.csv')
m_a2=pd.read_csv('input/nbaiot-dataset/4.mirai.ack.csv')
m_sc2=pd.read_csv('input/nbaiot-dataset/4.mirai.scan.csv')
m_sy2=pd.read_csv('input/nbaiot-dataset/4.mirai.syn.csv')
m_u2=pd.read_csv('input/nbaiot-dataset/4.mirai.udp.csv')
m_u_p2=pd.read_csv('input/nbaiot-dataset/4.mirai.udpplain.csv')

train_data2, labels2 = norm(benign2, g_c2, g_j2, g_s2, g_t2, g_u2, m_a2, m_sc2, m_sy2, m_u2, m_u_p2)

# test/train split  25% test
x_train2, x_test2, y_train2, y_test2 = train_test_split(
    train_data2, labels2, test_size=0.25, random_state=42)

# Samsung_SNH_1011_N_Webcam
benign3=pd.read_csv('input/nbaiot-dataset/5.benign.csv')
g_c3=pd.read_csv('input/nbaiot-dataset/5.gafgyt.combo.csv')
g_j3=pd.read_csv('input/nbaiot-dataset/5.gafgyt.junk.csv')
g_s3=pd.read_csv('input/nbaiot-dataset/5.gafgyt.scan.csv')
g_t3=pd.read_csv('input/nbaiot-dataset/5.gafgyt.tcp.csv')
g_u3=pd.read_csv('input/nbaiot-dataset/5.gafgyt.udp.csv')
m_a3=pd.read_csv('input/nbaiot-dataset/5.mirai.ack.csv')
m_sc3=pd.read_csv('input/nbaiot-dataset/5.mirai.scan.csv')
m_sy3=pd.read_csv('input/nbaiot-dataset/5.mirai.syn.csv')
m_u3=pd.read_csv('input/nbaiot-dataset/5.mirai.udp.csv')
m_u_p3=pd.read_csv('input/nbaiot-dataset/5.mirai.udpplain.csv')

train_data3, labels3 = norm(benign3, g_c3, g_j3, g_s3, g_t3, g_u3, m_a3, m_sc3, m_sy3, m_u3, m_u_p3)

# test/train split  25% test
x_train3, x_test3, y_train3, y_test3 = train_test_split(
    train_data3, labels3, test_size=0.25, random_state=42)

#Danmini_Doorbell
benign4=pd.read_csv('input/nbaiot-dataset/1.benign.csv')
g_c4=pd.read_csv('input/nbaiot-dataset/1.gafgyt.combo.csv')
g_j4=pd.read_csv('input/nbaiot-dataset/1.gafgyt.junk.csv')
g_s4=pd.read_csv('input/nbaiot-dataset/1.gafgyt.scan.csv')
g_t4=pd.read_csv('input/nbaiot-dataset/1.gafgyt.tcp.csv')
g_u4=pd.read_csv('input/nbaiot-dataset/1.gafgyt.udp.csv')
m_a4=pd.read_csv('input/nbaiot-dataset/1.mirai.ack.csv')
m_sc4=pd.read_csv('input/nbaiot-dataset/1.mirai.scan.csv')
m_sy4=pd.read_csv('input/nbaiot-dataset/1.mirai.syn.csv')
m_u4=pd.read_csv('input/nbaiot-dataset/1.mirai.udp.csv')
m_u_p4=pd.read_csv('input/nbaiot-dataset/1.mirai.udpplain.csv')

train_data4, labels4 = norm(benign4, g_c4, g_j4, g_s4, g_t4, g_u4, m_a4, m_sc4, m_sy4, m_u4, m_u_p4)

# test/train split  25% test
x_train4, x_test4, y_train4, y_test4 = train_test_split(
    train_data4, labels4, test_size=0.25, random_state=42)

model = Sequential()
model.add(Dense(10, input_dim=train_data.shape[1],activation='relu'))
model.add(Dense(64, input_dim=train_data.shape[1], activation='relu'))
model.add(Dropout(0.03))
model.add(Dense(32, input_dim=train_data.shape[1], activation='relu'))
model.add(Dropout(0.03))
model.add(Dense(10, input_dim=train_data.shape[1], activation='relu'))
model.add(Dense(labels.shape[1],activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, 
                        patience=5, verbose=1, mode='auto')

#ecobee thermostate
history = model.fit(x_train,y_train,validation_data=(x_test,y_test),
          callbacks=[monitor],verbose=2,epochs=10)
pred = model.predict(x_test)
pred = np.argmax(pred,axis=1)
y_eval = np.argmax(y_test,axis=1)
score = metrics.accuracy_score(y_eval, pred)*100
print("accuracy: {}".format(score))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss','val_loss'], loc = 'upper right')
plt.show()

model2 = Sequential()
model2.add(Dense(10, input_dim=train_data.shape[1],activation='relu'))
model2.add(Dense(64, input_dim=train_data.shape[1], activation='relu'))
model2.add(Dropout(0.03))
model2.add(Dense(32, input_dim=train_data.shape[1], activation='relu'))
model2.add(Dropout(0.03))
model2.add(Dense(10, input_dim=train_data.shape[1], activation='relu'))
model2.add(Dense(labels.shape[1],activation='softmax'))
model2.summary()
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, 
                        patience=5, verbose=1, mode='auto')

history2 = model2.fit(x_train2,y_train2,validation_data=(x_test2,y_test2),
          callbacks=[monitor],verbose=2,epochs=10)

# baby monitor
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss','val_loss'], loc = 'upper right')
plt.show()

pred2 = model2.predict(x_test2)
pred2 = np.argmax(pred2,axis=1)
y_eval2 = np.argmax(y_test2,axis=1)
score2 = metrics.accuracy_score(y_eval2, pred2)*100
print("accuracy: {}".format(score2))

model3 = Sequential()
model3.add(Dense(10, input_dim=train_data.shape[1],activation='relu'))
model3.add(Dense(64, input_dim=train_data.shape[1], activation='relu'))
model3.add(Dropout(0.03))
model3.add(Dense(32, input_dim=train_data.shape[1], activation='relu'))
model3.add(Dropout(0.03))
model3.add(Dense(10, input_dim=train_data.shape[1], activation='relu'))
model3.add(Dense(labels.shape[1],activation='softmax'))
model3.summary()
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, 
                        patience=5, verbose=1, mode='auto')
history3 = model3.fit(x_train3,y_train3,validation_data=(x_test3,y_test3),
          callbacks=[monitor],verbose=2,epochs=10)

# Samsung_SNH_1011_N_Webcam
plt.plot(history3.history['loss'])
plt.plot(history3.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss','val_loss'], loc = 'upper right')
plt.show()
pred3 = model3.predict(x_test3)
pred3 = np.argmax(pred3,axis=1)
y_eval3 = np.argmax(y_test3,axis=1)
score3 = metrics.accuracy_score(y_eval3, pred3)*100
print("accuracy: {}".format(score3))

model4 = Sequential()
model4.add(Dense(10, input_dim=train_data.shape[1],activation='relu'))
model4.add(Dense(64, input_dim=train_data.shape[1], activation='relu'))
model4.add(Dropout(0.03))
model4.add(Dense(32, input_dim=train_data.shape[1], activation='relu'))
model4.add(Dropout(0.03))
model4.add(Dense(10, input_dim=train_data.shape[1], activation='relu'))
model4.add(Dense(labels.shape[1],activation='softmax'))
model4.summary()
model4.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, 
                        patience=5, verbose=1, mode='auto')
history4 = model4.fit(x_train4,y_train4,validation_data=(x_test3,y_test3),
          callbacks=[monitor],verbose=2,epochs=10)

#Danmini_Doorbell
plt.plot(history3.history['loss'])
plt.plot(history3.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss','val_loss'], loc = 'upper right')
plt.show()
pred4 = model4.predict(x_test4)
pred4 = np.argmax(pred4,axis=1)
y_eval4 = np.argmax(y_test4,axis=1)
score4 = metrics.accuracy_score(y_eval4, pred4)*100
print("accuracy: {}".format(score4))

# metrics
print("accuracy: {}".format(score))

plt.rcParams["figure.figsize"] = [10, 8]
plt.rcParams["figure.autolayout"] = True

x=["Ecobee Thermostate", "Philips_B120N10_Baby_Monitor", "Samsung_SNH_1011_N_Webcam","Danmini Doorbell"]
y=[score, score2, score3, score4]

width = 0.75
fig, ax = plt.subplots()
color = ['indigo', 'teal', 'olive', 'maroon']
pps = ax.bar(x, y, width, align='center', color =color)

for p in pps:
   height = p.get_height()
   ax.text(x=p.get_x() + p.get_width() / 2, y=height+1,
      s="{}%".format(height),
      ha='center')
plt.title('Accuracy of models')
plt.show()
