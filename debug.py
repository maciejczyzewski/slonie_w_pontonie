import cv2
import numpy as np

from hmr.src.util import renderer as vis_util

import joints

def crop_from_points(img, points):
    height = img.shape[0]
    width = img.shape[1]

    mask = np.zeros((height, width), dtype=np.uint8)
    #points = np.array([[[10,150],[150,100],[300,150],[350,100],[310,20],[35,10]]])
    cv2.fillPoly(mask, points, (255))
    res = cv2.bitwise_and(img,img,mask = mask)
    rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    return cropped

def norm_points(points):
    # FIXME: faster version
    dpoints = []
    for point in points:
        point = list(map(int, point))
        print(point)
        dpoints.append(point)
    return np.array([dpoints], dtype=np.int32)

def visualize(img, proc_param, joints, verts, cam):
    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])

    # FIXME: @marek trzeba ogarnac te punkty, i co z nimi robic
    # https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md

    print(joints_orig)
    print(len(joints_orig))
    pBL = joints_orig[2]
    pBR = joints_orig[3]
    pCC = joints_orig[12]
    print(pBL, pBR)

    # Render results
    skel_img = vis_util.draw_skeleton(img, joints_orig)

    # FIXME: skel_img add big point at this
    import cv2
    cv2.circle(skel_img, (int(pBL[0]), int(pBL[1])), 10, (255, 0, 0), -1)
    cv2.circle(skel_img, (int(pBR[0]), int(pBR[1])), 10, (0, 255, 0), -1)
    cv2.circle(skel_img, (int(pCC[0]), int(pCC[1])), 10, (0, 0, 255), -1)

    import matplotlib.pyplot as plt
    # plt.ion()
    plt.figure(1)
    plt.clf()

    plt.subplot(231)
    plt.imshow(skel_img)
    plt.title('joint projection')
    plt.axis('off')

    #points2 = [pBL, pBR, pCC]
    #points2 = [[10,150],[150,100],[300,150],[350,100],[310,20],[35,10]]
    npoints2 = norm_points([pBL, pBR, pCC])
    print(npoints2, npoints2.shape, type(npoints2))
    points = np.array([[[10,150],[150,100],[300,150],[350,100],[310,20],[35,10]]])
    print(points, points.shape, type(points))
    cropped = crop_from_points(skel_img, npoints2)

    plt.subplot(232)
    plt.imshow(cropped)
    plt.title('cropped muscle')
    plt.axis('off')

    plt.show()

    # import ipdb
    # ipdb.set_trace()


img, proc_param, joints, verts, cams = joints.parse("data/lewy.jpg")
visualize(img, proc_param, joints, verts, cams)
