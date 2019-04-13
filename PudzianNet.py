from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import skimage.io as io
from skimage import color

batch_size = 8
num_classes = 3
epochs = 10

img_rows, img_cols = 150, 150

#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#print("TRAIN", x_train.shape, y_train.shape)
#print("TEST", x_test.shape, y_test.shape)

from glob import glob

prefix = "data/PudzianNet/prawy_abs/"
x_all = []
y_all = []

import toolbox
for filename in glob("{}/*/*".format(prefix)):
    img =  color.rgb2gray(io.imread(filename))
    #toolbox.debug(img)
    name = filename.replace(prefix, '')
    label_type = name.split("/")[0]
    #label = [0,0,0]
    if label_type == "A":
        label = 0
    if label_type == "B":
        label = 1
    if label_type == "C":
        label = 2
    #print(label)
    x_all.append(img)
    y_all.append(label)

import numpy as np

break_idx = int(len(x_all)*0.8)
x_train = np.array(x_all[0:break_idx])
y_train = np.array(y_all[0:break_idx])
x_test  = np.array(x_all[break_idx:])
y_test  = np.array(y_all[break_idx:])

print("TRAIN", x_train.shape, y_train.shape)
print("TEST", x_test.shape, y_test.shape)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#print(x_train)
#x_train /= 255
#x_test /= 255
print(x_test)
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_test)

import tensorflow as tf

def f1_score(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='elu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='elu'))
model.add(Conv2D(32, (3, 3), activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# model.add(Convolution2D(10,3,3, border_mode='same'))
# model.add(GlobalAveragePooling2D())
#Adam          = tf.keras.optimizers.Adam
#optimizer = Adam(lr=0.0001, decay=1e-6)

model.compile(loss='hinge',
              optimizer=keras.optimizers.Adadelta(),
              metrics=[f1_score])

"""
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
"""

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
