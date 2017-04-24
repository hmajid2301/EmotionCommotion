from __future__ import print_function
import numpy as np
import math

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

def get_class_weights(y):
  y = np.argmax(y,axis=1)
  class0 = len(y)/sum(y==0)
  class1 = len(y)/sum(y==1)
  class2 = len(y)/sum(y==2)
  class3 = len(y)/sum(y==3)
  return {0:class0,1:class1,2:class2,3:class3}


batch_size = 256
nb_classes = 4
nb_epoch = 5
test_index = 30242
split_index = 23489

# input image dimensions
img_rows, img_cols = 65, 40
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

X_train = np.load('../../../../local/whitened_data/iemo_X_whitened_40.npy')[split_index:]
X_test = np.load('../../../../local/whitened_data/iemo_X_whitened_40.npy')[0:split_index]
Y_train = np.load('../../../../local/whitened_data/iemo_y.npy')[split_index:]
Y_test = np.load('../../../../local/whitened_data/iemo_y.npy')[0:split_index:]

class_weights = get_class_weights(Y_train)


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

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test),class_weight=class_weights)
score = model.evaluate(X_test, Y_test, verbose=0)

train_predictions = model.predict_proba(X_train, batch_size=32, verbose=1)
#np.save('train_predictions_10_epoch_40.npy',train_predictions)
test_predictions = model.predict_proba(X_test, batch_size=32, verbose=1)
#np.save('test_predictions_10_epoch_40.npy',test_predictions)

print('Test score:', score[0])
print('Test accuracy:', score[1])

model.save('yt_5_40_sessions45.h5')

predictions = model.predict(X_test, batch_size=32, verbose=1)

cnf_matrix = confusion_matrix(np.argmax(Y_test,axis=1), np.argmax(predictions,axis=1))
cm_normalized = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
cm_normalized = cm_normalized.round(3) * 100

def plot_confusion_matrix(cm, title='CNN Confusion matrix', cmap=plt.cm.Greens,fontsize=14):
    plt.figure(figsize=(4.5,4.5))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize=fontsize+3)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=fontsize)

    tick_marks = np.arange(len(['Angry','Happy','Neutral','Sad']))
    plt.xticks(tick_marks, ['Angry','Happy','Neutral','Sad'], rotation=45,fontsize=fontsize)
    plt.yticks(tick_marks, ['Angry','Happy','Neutral','Sad'],fontsize=fontsize)
    thresh = cm.max() / 1.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, str(cm[i, j]) + "%",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                    fontsize=fontsize)
    plt.tight_layout()
    plt.ylabel('True label',fontsize=fontsize)
    plt.xlabel('Predicted label',fontsize=fontsize)
    plt.gcf().subplots_adjust(bottom=0.25,left=0.25)
    plt.savefig("NNCM_sessions45.png",transparent=True,figsize=(20,20),dpi=120)


plot_confusion_matrix(cm_normalized,fontsize=14)
plt.show()
