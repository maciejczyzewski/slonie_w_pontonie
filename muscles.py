import skimage.io as io
import numpy as np

from hmr.src.util import renderer as vis_util

import joints
import toolbox
from toolbox import pt, pts


def half(x):
    a, b = x[0]
    return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]


def get_muscles(img):
    img, proc_param, _joints, verts, cam = joints.parse_core(img)

    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, _joints, img_size=img.shape[:2])

    pb = [7, 8]  # otoczka
    lb = [9, 10]  # otoczka
    pr = [7, 6]  # otoczka
    lr = [10, 11]  # otoczka
    pu = [2, 1]  # otoczka grubsza niz ramie czy biceps
    lu = [3, 4]  # otoczka grubsza niz ramie czy biceps
    pl = [1, 0]  # otoczka
    ll = [4, 5]  # otoczka

    In = toolbox.inject

    skel_img = toolbox.premask(img, cam_for_render, vert_shifted)

    p23 = half(In(joints_orig, [2, 3]))
    p12_23 = half(pts([In(joints_orig, [12])[0], [p23]]))
    p28 = half(In(joints_orig, [2, 8]))
    p39 = half(In(joints_orig, [3, 9]))
    pkla = pts([*In(joints_orig, [12, 8])[0], p12_23, p28])
    lkla = pts([*In(joints_orig, [12, 9])[0], p12_23, p39])
    labs = pts([*In(joints_orig, [2])[0], p28, p12_23, p23])
    pabs = pts([*In(joints_orig, [3])[0], p39, p12_23, p23])

    prawy_biceps = toolbox.line_crop(
        skel_img,
        *
        toolbox.inject(
            joints_orig,
            pb)[0],
        size=50)
    lewy_biceps = toolbox.line_crop(
        skel_img,
        *
        toolbox.inject(
            joints_orig,
            lb)[0],
        size=50)
    prawa_klatka = toolbox.crop(skel_img, pkla)
    lewa_klatka = toolbox.crop(skel_img, lkla)
    prawe_ramie = toolbox.line_crop(
        skel_img,
        *
        toolbox.inject(
            joints_orig,
            pr)[0],
        size=40)
    lewe_ramie = toolbox.line_crop(
        skel_img,
        *
        toolbox.inject(
            joints_orig,
            lr)[0],
        size=40)
    prawe_udo = toolbox.line_crop(
        skel_img,
        *
        toolbox.inject(
            joints_orig,
            pu)[0],
        size=60)
    lewe_udo = toolbox.line_crop(
        skel_img,
        *
        toolbox.inject(
            joints_orig,
            lu)[0],
        size=60)
    prawa_lydka = toolbox.line_crop(
        skel_img,
        *
        toolbox.inject(
            joints_orig,
            pl)[0],
        size=40)
    lewa_lydka = toolbox.line_crop(
        skel_img,
        *
        toolbox.inject(
            joints_orig,
            ll)[0],
        size=40)
    lewy_abs = toolbox.crop(skel_img, labs)
    prawy_abs = toolbox.crop(skel_img, pabs)

    """
    toolbox.debug([lewy_abs,
                   prawy_abs,
                   prawa_klatka,
                   lewa_klatka,
                   prawy_biceps,
                   lewy_biceps,
                   prawe_ramie,
                   lewe_ramie,
                   prawe_udo,
                   lewe_udo,
                   prawa_lydka,
                   lewa_lydka])
    """

    return {'lewy_abs': lewy_abs,
            'prawy_abs': prawy_abs,
            'prawa_klatka':     prawa_klatka,
            'lewa_klatka':   lewa_klatka,
            'prawy_biceps':   prawy_biceps,
            'lewy_biceps':   lewy_biceps,
            'prawe_ramie' :  prawe_ramie,
            'lewe_ramie':    lewe_ramie,
            'prawe_udo':    prawe_udo,
            'lewe_udo' :  lewe_udo,
            'prawa_lydka':    prawa_lydka,
               'lewa_lydka':  lewa_lydka}

import operator
def cropND(img, bounding):
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]

from PIL import Image

def fit(image, max_size, method=Image.ANTIALIAS):
    """Skaluje do odpowiedniego rozmiaru oraz wysrodkuwuje"""
    image = Image.fromarray(image)
    im_aspect = float(image.size[0]) / float(image.size[1])
    out_aspect = float(max_size[0]) / float(max_size[1])
    # interesuje nasz ratio w stosunku do krawedzi
    if im_aspect >= out_aspect:
        scaled = image.resize(
            (max_size[0], int((float(max_size[0]) / im_aspect) + 0.5)), method)
    else:
        scaled = image.resize(
            (int((float(max_size[1]) * im_aspect) + 0.5), max_size[1]), method)
    # srodek obrazka sie powinnien tutaj znalesc
    offset = (int((max_size[0] - scaled.size[0]) / 2),
              int((max_size[1] - scaled.size[1]) / 2))
    back = Image.new("RGB", max_size, "black")
    back.paste(scaled, offset)  # wklejamy jedno w drugie
    back = np.array(back) 
    #back = back[:, :, ::-1].copy()
    return back

import xxhash
def __action_hash(obj):
    h = xxhash.xxh64()
    h.update(obj)
    return h.intdigest()

def parse_file(filename):
    label_w = filename.split("_")[-2][-1]
    dmuscles = get_muscles(io.imread(filename))
    okay_dmuscles = {}
    for key in dmuscles:
        M = dmuscles[key]
        size = M.shape[0]*M.shape[1]
        # rotation
        if not isinstance(M, np.ndarray): continue
        if size < 100: continue
        if M.shape[0] < M.shape[1]:
            M = np.rot90(M)
            #print(M)
        #print("TYPE", M)
        # FIXME: resize to MAX 100
        M = fit(M, (150, 150))
        #M = cropND(M, (100,100))
        okay_dmuscles[key] = M

    #print(okay_dmuscles.keys())
    #print(okay_dmuscles.values())
    #toolbox.debug(list(okay_dmuscles.values()))
    for key in okay_dmuscles:
        img = okay_dmuscles[key]
        dhash = __action_hash(img)
        place = "data/PudzianNet/{}/{}/{}.png".format(key, label_w, dhash)
        print("SAVE TO", place)
        io.imsave(place, img)

    #return okay_dmuscles

import os
keys_1 = ['lewy_abs', 'prawy_abs', 'prawa_klatka', 'lewa_klatka', 'prawy_biceps', 'lewy_biceps', 'prawe_ramie', 'lewe_ramie', 'prawe_udo', 'lewe_udo', 'prawa_lydka', 'lewa_lydka']
keys_2 = ['A', 'B', 'C']

for a in keys_1:
    for b in keys_2:
        cmd = "mkdir -p data/PudzianNet/{}/{}".format(a, b)
        print(cmd)
        os.system(cmd)

from tqdm import tqdm
from glob import glob

for filename in tqdm(glob("data/dataset men ABC/*")):
    try:
        parse_file(filename)
    except:
        print("ERROR")

# parse_file("data/dataset men ABC/A_566448.png")
