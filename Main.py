# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 13:30:29 2019

@author: The Machine: Hunt Waggoner
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Step 1 lets get our data in order
#First lets import the data that tells us what each image is
dataset = pd.read_csv('Data_Entry_2017.csv')
'''
#now lets restrict the data set to just finding labels
#this is column index 1 and we want 0 for other purposes
Diagnosis = dataset.iloc[1:,0:2]
#variable to hold our list of pictures with mass/nodes
massPictures =[]
#ok very good now we need to search through that list and find "Mass / Nodule"
for i in range(0,1300):
    #split the word on | to deal with double diagnosis
    splitdiagnosis = Diagnosis.get_value(i,1,takeable = True)
    splitdiagnosis = splitdiagnosis.split('|')
    
    for eachDiagnosis in splitdiagnosis:
        if (eachDiagnosis == 'Mass' or eachDiagnosis == 'Nodule'):
            massPictures.append(Diagnosis.iloc[i,0])
'''
'''   
#Ok this is working, now lets pull all the images we want from the folder to make a training and test set
import os
import shutil

for filenamewithmass in massPictures:
    for filename in os.listdir('Chest Xrays'):
        filenametry = 'Chest Xrays/'+filename
        if(filenamewithmass == filename):
            shutil.copy(filenametry,'Mass',follow_symlinks = True)
        else:
            shutil.copy(filenametry, 'NoMass',follow_symlinks = True)
#so now lets filter out NoMass so no pictures with mass are in it

for file in os.listdir('NoMass'):
    for pictures in massPictures:
        if(file == pictures):
            os.remove('NoMass/'+file)

'''


''' The data is how we want it, now lets machine learn!
START HERE if your data is how you want it
'''

#Step 1 BUILDING THE CNN
#Get our libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Start the CNN
classifier = Sequential()

#step 1.1 Convolution
'''Convolution creates a kernal that is convolved with layer input to 
produce our tensor outputs'''

classifier.add(Convolution2D(32,3,3,input_shape = (64,64,3), activation = 'relu'))


#Step 1.2 Pooling
classifier.add(MaxPooling2D(pool_size =(2,2)))

#Step 1.3 Flattening
classifier.add(Flatten())

#Step 1.4
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))   
        
#Compile the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the CNN to our image set
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('data/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 8,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('data/test_set',
                                            target_size = (64, 64),
                                            batch_size = 8,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 60,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 200)