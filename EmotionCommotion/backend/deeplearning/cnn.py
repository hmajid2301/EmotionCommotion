from __future__ import print_function
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

batch_size = 256
nb_classes = 4
nb_epoch = 10

# input image dimensions
img_rows, img_cols = 65, 40
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

X_train = np.load('../../../../local/whitened_data/X_train_whitened.npy')
X_test = np.load('../../../../local/whitened_data/X_test_whitened.npy')
Y_train = np.load('../../../../local/whitened_data/y_train.npy')
Y_test = np.load('../../../../local/whitened_data/y_test.npy')



if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dropout(0.75))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['categorical_accuracy'])

model.fit(X_test, Y_test, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)

preds = model.predict_proba(X_test, batch_size=32, verbose=1)
np.save('preds_10_epoch.npy',preds)

# from keras import backend as K
#
# get_8th_layer_output = K.function([model.layers[0].input,K.learning_phase()],
#                                   [model.layers[7].output])
# layer_output = get_8th_layer_output([X_test[0:1],0])[0]


print('Test score:', score[0])
print('Test accuracy:', score[1])

model.save('cnn_10.h5')
