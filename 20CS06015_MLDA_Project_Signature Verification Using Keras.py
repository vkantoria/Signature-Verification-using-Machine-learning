#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import random
from PIL import Image
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import os
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from keras import backend as K
import cv2
import matplotlib.pyplot as plt


# In[3]:


train_dir="/kaggle/input/signature-verification-dataset/sign_data/train"
train_csv="/kaggle/input/signature-verification-dataset/sign_data/train_data.csv"
test_csv="/kaggle/input/signature-verification-dataset/sign_data/test_data.csv"
test_dir="/kaggle/input/signature-verification-dataset/sign_data/test"


# In[4]:


df_train=pd.read_csv(train_csv)
df_train.sample(10)


# In[5]:


df_test=pd.read_csv(test_csv)
df_test.sample(10)


# In[6]:


print(df_train.shape)
print(df_test.shape)


# In[7]:


# Added new code
def dataset_train(train_csvfile):
    x1=[]
    x2=[]
    y_train=[]
    for i in range(0,2000):
        image1_path=os.path.join(train_dir,train_csvfile.iat[i,0])
        image2_path=os.path.join(train_dir,train_csvfile.iat[i,1])
        img1=cv2.imread(image1_path)
        img1= cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img1=cv2.resize(img1,(150,150))
        x1.append(img1)
        img2=cv2.imread(image2_path)
        img2= cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img2=cv2.resize(img2,(150,150))
        x2.append(img2)
        y_train.append(train_csvfile.iat[i,2])

    x1=np.array(x1).astype(np.float32)/255.0
    x2=np.array(x2).astype(np.float32)/255.0
    y_train=np.array(y_train).astype(np.float32)
    
    return x1,x2,y_train


# In[8]:


# Added new code
def dataset_test(test_csvfile):
    x1=[]
    x2=[]
    y_train=[]
    for i in range(2000,4000):
        image1_path=os.path.join(test_dir,test_csvfile.iat[i,0])
        image2_path=os.path.join(test_dir,test_csvfile.iat[i,1])
        img1=cv2.imread(image1_path)
        img1= cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img1=cv2.resize(img1,(150,150))
        x1.append(img1)
        img2=cv2.imread(image2_path)
        img2= cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img2=cv2.resize(img2,(150,150))
        x2.append(img2)
        y_train.append(test_csvfile.iat[i,2])

    x1=np.array(x1).astype(np.float32)/255.0
    x2=np.array(x2).astype(np.float32)/255.0
    y_train=np.array(y_train).astype(np.float32)
    
    return x1,x2,y_train


# In[9]:


#for test set
xt1,xt2,yt=dataset_test(df_test)


# In[10]:


#for train set
xs1,xs2,ys=dataset_train(df_train)


# In[11]:


def dist1(xy):
    x, y = xy
    sum_abs = K.sum(K.abs(x - y), axis=1, keepdims=True)
#     return sum_abs
    return K.maximum(sum_abs, K.epsilon())
def dist2(xy):
    x,y=xy
    sum_square=K.sum(K.square(x-y),axis=1,keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def dist3(xy):
    x,y=xy
    return K.sqrt(K.mean(K.square(x-y),axis=1,keepdims=True))
def dist4(xy):
    x,y=xy
    ss=K.sum(K.square(x-y),axis=1,keepdims=True)
    return K.sqrt(ss)/x.shape[1]


input1=keras.layers.Input(shape=(150,150,1))
# using relu as a loss function
x=keras.layers.Conv2D(64,(10,10),activation='relu')(input1)
x=keras.layers.MaxPooling2D(2,2)(x)
x=keras.layers.Dropout(0.2)(x)
x=keras.layers.Conv2D(128,(4,4),activation='relu')(x)
x=keras.layers.MaxPooling2D(2,2)(x)
x=keras.layers.Dropout(0.5)(x)
x=keras.layers.Flatten()(x)
x=keras.layers.Dense(500,activation='relu')(x)
dense=keras.models.Model(inputs=input1,outputs=x)


img1=keras.layers.Input(shape=(150,150,1))
img2=keras.layers.Input(shape=(150,150,1))
dense1=dense(img1)
dense2=dense(img2)
fc=keras.layers.Lambda(dist3)([dense1,dense2])
fc=keras.layers.Dense(1,activation='sigmoid')(fc)
m=keras.models.Model(inputs=[img1,img2],outputs=fc)

m.compile(loss = "binary_crossentropy", optimizer="adam", metrics=["accuracy"])
m.summary()


# In[12]:


m.fit([xs1,xs2],ys,epochs=5)
m.evaluate([xt1,xt2],yt)


# In[13]:


def check(stored,image):
    img2=np.array([image])
    for i in range(0,stored.shape[0]):
        img1=np.array([stored[i]])
        pred=m.predict([img1,img2])
        print(pred)
        if(pred<0.5):
            print("matched",i)
            return
    print("unmatched")


# In[14]:


type(img1)


# In[15]:


database_signatures=xt1[:15]
check(database_signatures,xt1[13])


# In[ ]:




