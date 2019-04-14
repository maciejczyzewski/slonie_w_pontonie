from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import skimage.io as io
from skimage import color

# FIXME: automatic augumentation
# FIXME: more pad???

batch_size = 64
num_classes = 3
epochs = 5

img_rows, img_cols = 100, 100

#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#print("TRAIN", x_train.shape, y_train.shape)
#print("TEST", x_test.shape, y_test.shape)

from glob import glob
from imgaug import augmenters as iaa

seq = iaa.Sequential([
    iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
])

import toolbox
import numpy as np

def get_dataset_for_type(name):
    prefix = "data/PudzianNet/{}/".format(name)
    x_all = []
    y_all = []

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

        for _ in range(0, 30):
            images_aug = seq.augment_images([img])[0]
            x_all.append(img)
            y_all.append(label)



    # https://github.com/aleju/imgaug

    idx = np.random.permutation(len(x_all))
    x_all = np.array(x_all)
    y_all = np.array(y_all)
    x_all,y_all = x_all[idx], y_all[idx]

    # FIXME: shuffle
    break_idx = int(len(x_all)*0.8)
    x_train = np.array(x_all[0:break_idx])
    y_train = np.array(y_all[0:break_idx])
    x_test  = np.array(x_all[break_idx:len(x_all)])
    y_test  = np.array(y_all[break_idx:len(x_all)])

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
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)

import tensorflow as tf
EarlyStopping = tf.keras.callbacks.EarlyStopping

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
model.add(Conv2D(32, (3, 3), activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# model.add(Convolution2D(10,3,3, border_mode='same'))
# model.add(GlobalAveragePooling2D())
#Adam          = tf.keras.optimizers.Adam
#optimizer = Adam(lr=0.0001, decay=1e-6)

model.compile(loss='hinge',
              optimizer=keras.optimizers.Adam(lr=0.0001, decay=1e-6),
              metrics=[f1_score, 'accuracy'])

"""
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
"""

net_names = ['lewy_abs', 'prawy_abs', 'prawa_klatka', 'lewa_klatka', 'prawy_biceps', 'lewy_biceps', 'prawe_ramie', 'lewe_ramie', 'prawe_udo', 'lewe_udo', 'prawa_lydka', 'lewa_lydka']

while 1:
    for name in net_names:
        print("\033[92m=== {} ===\033[m".format(name))
        tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='logs/'+name, histogram_freq=0, write_graph=True, write_images=True)
        if glob("model/"+name) != []:
            model.load_weights("model/"+name)
        x_train, y_train, x_test, y_test = get_dataset_for_type(name)
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  shuffle=True,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test),callbacks=[tbCallBack,
                              EarlyStopping(min_delta=0.00025, patience=2)])
        score = model.evaluate(x_test, y_test, verbose=0)
        model.save_weights("model/"+name)
        print("WHAT", name)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
