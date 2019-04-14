"""generuje dane "results" dla serii danych z input_series/"""

# czyta z input_series/
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf

import numpy as np
import skimage.io as io

from tqdm import tqdm
from glob import glob

# FIXME: jakas obsluga PudzianNet

DEBUG = False

from PudzianNet import net_names, model, f1_score

# 5:30 FIXME: tworzymy z tego dataframe w pandas???? --> time series prediction
# 6:30 FIXME: robimy heatmape --> na cieleeee????????

# FIXME: silne cachowanie predykcji dla zdjecia muscla

import keras.metrics
from keras.models import load_model
keras.metrics.f1_score = f1_score

# model = load_model('model/multi_task/try.h5', custom_objects={'loss_max': loss_max})
img_rows, img_cols = 100, 100

import xxhash

def __action_hash(obj):
    h = xxhash.xxh64()
    h.update(obj)
    return h.intdigest()

print("[[[INIT]]]")
from copy import copy
if DEBUG:
    models = {}
    for net_name in net_names:
        print("wczytuje {}".format(net_name))
        #models[net_name] = load_model("model/"+net_name, custom_objects={'f1_score':
        #    f1_score})
        models[net_name] = copy(model)
        models[net_name].load_weights("model/"+net_name)
    print("------------")

import cv2
from skimage.transform import resize
import toolbox
def dynamic_size(img):
    box = list(img.shape)
    avg_area = 800*800
    cur_area = box[0] * box[1]  # jego wymiary
    factor = (avg_area / cur_area)**(1 / 2)  # aby byly podobne
    print("[GRID] CUR", cur_area, factor)  # do pozostalych zdjec
    box[0] = int(box[0] * factor)
    box[1] = int(box[1] * factor)
    print("NOWE", box)
    img = cv2.resize(img, (box[1], box[0]))
    #img = resize(img, box, anti_aliasing=True)*255 # FIXME:? *255
    #img = img.astype('int')
    #toolbox.debug(img)
    return img

results = []

from muscles import parse_file_for_muscles
import os
from skimage import color
for filename in tqdm(sorted(glob("input_series/*"))): #FIXME: tylko pierwsze narazie
    print(filename)
    img = io.imread(filename)
    #print(img)
    img = dynamic_size(img)
    imgs_muscles = parse_file_for_muscles(img)

    cur_results = {}

    for net_name in imgs_muscles:
        img = imgs_muscles[net_name]
        img = color.rgb2gray(img)

        valhash = __action_hash(img)
        valpath = "cache2/" + str(valhash)
        print("valhash={}".format(valhash))
        if os.path.isfile(valpath + ".npz"):
            result = np.load(valpath + ".npz")["result"]
            print(result)
            cur_results[net_name] = result
            continue
        # FIXME: ------------------------------

        x_input = np.array([img])
        if K.image_data_format() == 'channels_first':
            x_input = x_input.reshape(x_input.shape[0], 1, img_rows, img_cols)
        else:
            x_input = x_input.reshape(x_input.shape[0], img_rows, img_cols, 1)

        #x_input = x_input.reshape(x_input.shape[0], 1, img_rows, img_cols)

        result = models[net_name].predict(x_input)[0]
        result = [round(result[0], 3), round(result[1], 3), round(result[2], 3)]
        print("[{}] net_name={}".format(filename, net_name), result)

        np.savez_compressed(valpath, result=result)
        cur_results[net_name] = result

    results.append(cur_results)
    print("========")

np.savez_compressed("results", results=results)
