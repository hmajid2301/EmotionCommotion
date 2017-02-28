'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from glob import glob
import pandas as pd

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

def generate_frames(specto_path,label_path):
    labels = pd.read_csv(label_path)
    labels = labels.set_index('session')
    while True:
        for filename in glob(specto_path + '/*.npy'):
            spectos = np.load(filename)
            name = filename[len(specto_path)+1:-11]
            label = labels.loc[name].label
            for frame in spectos:
                if K.image_dim_ordering() == 'th':
                    frame = frame.reshape(1,1, frame.shape[0], frame.shape[1])
                else:
                    frame = frame.reshape(1,frame.shape[0], frame.shape[1],1)
                frame = frame.astype('float32')
                if label == "neu":
                    return_label = np.array([1,0,0,0])
                elif label == "hap":
                    return_label = np.array([0,1,0,0])
                elif label == "sad":
                    return_label = np.array([0,0,1,0])
                else:
                    return_label = np.array([0,0,0,1])
                return_label = return_label.reshape(1,4)
                yield(frame,return_label)

batch_size = 128
nb_classes = 4
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 129, 71
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernelsize
kernel_size = (3, 3)

if K.image_dim_ordering() == 'th':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)


model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit_generator(generate_frames('spectograms/train','allLabels.csv'),
                    samples_per_epoch=64,nb_epoch=1000,verbose=2)

score = model.evaluate_generator(generate_frames('spectograms/test','allLabels.csv'),
                                                 val_samples=943)
print('Test score:', score[0])
print('Test accuracy:', score[1])

