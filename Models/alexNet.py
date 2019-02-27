# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Flatten, Dropout
from keras.regularizers import l2

def alexNet_model(image_shape = (32, 32, 3), n_classes = 10, l2_reg=0., weights=None):
    alexnet = Sequential()
    
    #layer1
    alexnet.add(Conv2D(96,(11, 11), 
                input_shape = image_shape, 
                padding = 'same', 
                kernel_regularizer=l2(l2_reg),
                activation='relu',
                strides=(4,4)))
    alexnet.add(MaxPooling2D(pool_size=(2,2)))
    
    #layer2
    alexnet.add(Conv2D(256,(5,5), 
                padding = 'same', 
                kernel_regularizer=l2(l2_reg),
                activation='relu'))
    alexnet.add(MaxPooling2D(pool_size=(2,2)))
    #layer3
    alexnet.add(Conv2D(384, (3, 3),
                padding = 'same',
                activation='relu'))
    
    #layer4
    alexnet.add(Conv2D(384, (3, 3),
                activation='relu',
                padding = 'same'))
    
    #layer5
    alexnet.add(Conv2D(256, (3, 3),
                activation='relu',
                padding = 'same'))
    alexnet.add(MaxPooling2D(pool_size=(2,2)))
    
    #layer6
    alexnet.add(Flatten())
    alexnet.add(Dense(4096,
                activation='relu'))
    alexnet.add(Dropout(0.5))
    
    #layer7
    alexnet.add(Dense(4096,
                activation='relu'))
    alexnet.add(Dropout(0.5))
    
    #layer8
    alexnet.add(Dense(n_classes,
                activation='softmax'))
    return alexnet

model = alexNet_model()