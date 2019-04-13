import os
import sys
import xxhash

import numpy as np
import skimage.io as io
import tensorflow as tf

from absl import flags

from hmr.src.util import image as img_util
from hmr.src.RunModel import RunModel

import hmr.src.config

flags.DEFINE_string('cache_path', './cache/', 'Path to cache dir.')


def __action_hash(obj):
    h = xxhash.xxh64()
    h.update(obj)
    return h.intdigest()


def __action_setup():
    config = flags.FLAGS
    config(sys.argv)
    config.load_path = hmr.src.config.PRETRAINED_MODEL
    config.batch_size = 1
    return config


config = __action_setup()
print("MODEL", hmr.src.config.PRETRAINED_MODEL)


def preprocess_image(img):
    global config
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if np.max(img.shape[:2]) != config.img_size:
        print('\033[92mResizing so the max image size is %d..\033[m'
              % config.img_size)
        scale = (float(config.img_size) / np.max(img.shape[:2]))
    else:
        scale = 1.
    center = np.round(np.array(img.shape[:2]) / 2).astype(int)
    # image center in (x,y)
    center = center[::-1]

    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               config.img_size)

    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img


def parse(img_path):
    return parse_core(io.imread(img_path))


def parse_core(img_org):
    global config

    input_img, proc_param, img = preprocess_image(img_org)

    valhash = __action_hash(input_img)
    valpath = config.cache_path + str(valhash)
    print("valhash={}".format(valhash))

    if os.path.isfile(valpath + ".npz"):
        l = np.load(valpath + ".npz")
        joints, verts, cams, joints3d, theta = \
            l["joints"], l["verts"], l["cams"], l["joints3d"], l["theta"]
        return img, proc_param, joints[0], verts[0], cams[0]

    sess = tf.Session()
    model = RunModel(config, sess=sess)

    # add batch dimension: 1 x D x D x 3
    input_img = np.expand_dims(input_img, 0)

    # theta is the 85D vector holding [camera, pose, shape] where camera is 3D
    # [s, tx, ty] pose is 72D vector holding the rotation of 24 joints of SMPL
    # in axis angle format shape is 10D shape coefficients of SMPL
    joints, verts, cams, joints3d, theta = model.predict(
        input_img, get_theta=True)

    # zapisujemy do folderu cache/
    np.savez_compressed(
        valpath,
        joints=joints,
        verts=verts,
        cams=cams,
        joints3d=joints3d,
        theta=theta)

    return img, proc_param, joints[0], verts[0], cams[0]
