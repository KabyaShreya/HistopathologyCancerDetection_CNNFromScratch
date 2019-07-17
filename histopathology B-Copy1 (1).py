


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from numpy import mean
from numpy import std
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from keras.regularizers import l2
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import SeparableConv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import Adam
import tensorflow

import os;
import datetime  
import cv2 
import seaborn as sns
import fnmatch
import glob


# In[35]:


image = glob.glob('/Users/kavyashreya/Downloads/IDC_regular_ps50_idx5/**/*.png', recursive=True)


# In[36]:


X=list()
Y=list()
WIDTH = 50
HEIGHT = 50
lowerIndex=0

class0 = '*class0.png'
class1 = '*class1.png'
classZero = fnmatch.filter(image, class0)
classOne = fnmatch.filter(image, class1)
upperIndex = len(image)
 #entering loop
for img in image[lowerIndex:upperIndex]:
        full_size_image = cv2.imread(img)
        X.append(cv2.resize(full_size_image, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC))
        if img in classZero:
            Y.append(0)
        elif img in classOne:
            Y.append(1)
IDCn=X[:78786]
IDCp=X[199000:]
X=IDCn+IDCp
Y=Y[:78786]+Y[199000:]


def describeData(a,b):
    print('Total number of images: {}'.format(len(a)))
    print('Number of IDC(-) Images: {}'.format(np.sum(b==0)))
    print('Number of IDC(+) Images: {}'.format(np.sum(b==1)))
    print('Percentage of positive images: {:.2f}%'.format(100*np.mean(b)))
    print('Image shape (Width, Height, Channels): {}'.format(a[0].shape))
describeData(X,Y)


# In[4]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
X_train = X_train / 256.0
X_test = X_test / 256.0

print("Training Data Shape:", X_train.shape, X_train.shape)
print("Testing Data Shape:", X_test.shape, X_test.shape)


# In[12]:


X_train=X
Y_train=Y
X_train = X_train / 256.0


# In[14]:


type(Y_train)


# In[6]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
Y_train = label_encoder.fit_transform(Y_train)


# In[7]:


img_rows, img_cols = X_train.shape[1],X_train.shape[2]
input_shape = (img_rows, img_cols, 3)


# In[16]:


type(img_rows)
type(img_cols)
type(input_shape)


# In[8]:


model = Sequential()
model.add(SeparableConv2D(32, (3, 3), activation='elu', kernel_initializer='he_uniform',input_shape=input_shape,kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(SeparableConv2D(64, (3, 3), activation='elu', kernel_initializer='he_uniform',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(SeparableConv2D(64, (3, 3), activation='elu', kernel_initializer='he_uniform',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(SeparableConv2D(128, (3, 3), activation='elu', kernel_initializer='he_uniform',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='elu', kernel_initializer='he_uniform',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
# compile model
opt = Adam()
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


# In[9]:


scores=list()
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10, shuffle=True)
for train_ix, test_ix in skf.split(X_train,Y_train):
   # select rows for train and test
   trainX, trainY, testX, testY = X_train[train_ix], Y_train[train_ix],X_train[test_ix], Y_train[test_ix]
   trainY = to_categorical(trainY, num_classes = 2)
   testY = to_categorical(testY, num_classes = 2)
   # fit model
   model.fit(trainX, trainY, epochs=12, batch_size=32, validation_data=(testX, testY), verbose=0)
   # evaluate model
   _, acc = model.evaluate(testX, testY, verbose=0)
   print('> %.3f' % (acc * 100.0))
   # stores scores
   scores.append(acc)
print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))


# In[10]:


import sklearn
from sklearn.metrics import confusion_matrix
import itertools
score = model.evaluate(testX, testY, verbose=0)
print('\nKeras CNN #1A - accuracy:', score[1],'\n')
y_pred = model.predict(testX) 
map_characters = {0: 'IDC(-)', 1: 'IDC(+)'}
print('\n', sklearn.metrics.classification_report(np.where(testY > 0)[1], np.argmax(y_pred, axis=1), target_names=list(map_characters.values())), sep='')
Y_pred_classes = np.argmax(y_pred,axis = 1) 
Y_true = np.argmax(testY,axis = 1) 
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize = (5,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
plot_confusion_matrix(confusion_mtx, classes = list(map_characters.values())) 







