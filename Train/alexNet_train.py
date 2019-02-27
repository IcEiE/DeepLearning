# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 13:46:40 2019

@author: Issa Hijazi
"""

import sys
sys.path.append("..")
from Models import alexNet
from keras.datasets import cifar10
from keras.optimizers import Adamax
from keras.utils import to_categorical

if __name__ == '__main__':
    #Pre-data
    batch_size = 128
    num_classes = 10
    epochs = 20
    img_rows, img_cols, img_depth = 32, 32, 3
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    #Maybe feature normalization on x?
    
    #---------------------------------
    
    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    
    #Model
    model = alexNet.alexNet_model(image_shape=(img_rows, img_cols, img_depth), n_classes = num_classes)
    model.compile(loss = 'categorical_crossentropy',
                  optimizer= Adamax(lr=0.0001, decay=1e-6),
                  metrics=['accuracy'])
    
    history = model.fit(x_train, y_train,
              batch_size = batch_size,
              epochs = epochs,
              verbose = 1,
              validation_data = (x_test, y_test))
    
	# Evaluate the model
    print('[INFO] Evaluating the trained model...')
    (loss, accuracy) = model.evaluate(x_test, 
                            y_test,
                            batch_size=128,
                            verbose=1)
    print('[INFO] accuracy: {:.2f}%'.format(accuracy * 100))