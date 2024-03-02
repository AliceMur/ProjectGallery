import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping
def norm(benign, g_c, g_j, g_s, g_t, g_u, m_a, m_sc, m_sy, m_u, m_u_p):
    benign=benign.sample(frac=0.25,replace=False)
    g_c=g_c.sample(frac=0.25,replace=False)
    g_j=g_j.sample(frac=0.5,replace=False)
    g_s=g_s.sample(frac=0.5,replace=False)
    g_t=g_t.sample(frac=0.15,replace=False)
    g_u=g_u.sample(frac=0.15,replace=False)
    m_a=m_a.sample(frac=0.25,replace=False)
    m_sc=m_sc.sample(frac=0.15,replace=False)
    m_sy=m_sy.sample(frac=0.25,replace=False)
    m_u=m_u.sample(frac=0.1,replace=False)
    m_u_p=m_u_p.sample(frac=0.27,replace=False)

    benign['type']='benign'
    m_u['type']='mirai_udp'
    g_c['type']='gafgyt_combo'
    g_j['type']='gafgyt_junk'
    g_s['type']='gafgyt_scan'
    g_t['type']='gafgyt_tcp'
    g_u['type']='gafgyt_udp'
    m_a['type']='mirai_ack'
    m_sc['type']='mirai_scan'
    m_sy['type']='mirai_syn'
    m_u_p['type']='mirai_udpplain'

    data=pd.concat([benign,m_u,g_c,g_j,g_s,g_t,g_u,m_a,m_sc,m_sy,m_u_p],
                axis=0, sort=False, ignore_index=True)

    #how many instances of each class
    data.groupby('type')['type'].count()

    #shuffle rows of dataframe 
    sampler=np.random.permutation(len(data))
    data=data.take(sampler)
    data.head()

    #dummy encode labels, store separately
    labels_full=pd.get_dummies(data['type'], prefix='type')
    labels_full.head()

    #drop labels from training dataset
    data=data.drop(columns='type')
    data.head()

    #standardize numerical columns
    def standardize(df,col):
        df[col]= (df[col]-df[col].mean())/df[col].std()

    data_st=data.copy()
    for i in (data_st.iloc[:,:-1].columns):
        standardize (data_st,i)

    data_st.head()

    #training data for the neural net
    train_data=data_st.values

    #labels for training
    labels=labels_full.values
    labels

    return train_data, labels
