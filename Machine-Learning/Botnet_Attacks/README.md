# Detect IoT Botnet Attacks
### Accuracy all test above ~90% on test set
I created a Keras deep neural network model. I decided to see how well it would predict the likelihood of a bot net attack on a given device if it was only trained on that system. I separately trained the model on different data sets, consisting of Ecobee Thermostate, Philips B120N10 Baby Monitor, Samsung SNH 1011 N Webcam, and Danmini Doorbell. The resulting accuracy score of the model on the test sets was ~92%, ~94%, ~90%, and ~92%, respectively.  

I built the model using Tensorflow, Keras, and Pandas. I also used ADAM as the model optimizer and categorical cross-entropy as the loss function. The resulting model consists of 4,395 Trainable Parameters. 

To visualize the model training results, I used the matplotlib libraries. The resulting plot graphs and bar chart can be seen in the Graphs folder.

Citation:
**Y. Meidan, M. Bohadana, Y. Mathov, Y. Mirsky, D. Breitenbacher**, A. Shabtai, and Y. Elovici 'N-BaIoT: Network-based Detection of IoT Botnet Attacks Using Deep Autoencoders', IEEE Pervasive Computing, Special Issue - Securing the IoT (July/Sep 2018).

Dataset obtained from: https://www.kaggle.com/datasets/mkashifn/nbaiot-dataset