import cv2
import numpy as np

from hmr.src.util import renderer as vis_util

import toolbox
from toolbox import pt, pts

import joints

def visualize(img, proc_param, joints, verts, cam):
    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])

    print(joints_orig)
    print(len(joints_orig))
    pBL = pt(joints_orig[2])
    pBR = pt(joints_orig[3])
    pCC = pt(joints_orig[12])
    print(pBL, pBR)

    # Render results
    skel_img = vis_util.draw_skeleton(img, joints_orig)

    # FIXME: skel_img add big point at this
    import cv2
    cv2.circle(skel_img, pBR, 10, (255, 0, 0), -1)
    cv2.circle(skel_img, pBR, 10, (0, 255, 0), -1)
    cv2.circle(skel_img, pCC, 10, (0, 0, 255), -1)

    #cv2.line(skel_img, pBR, pCC, (0,0,0), 50)

    x = toolbox.line_crop(skel_img, \
        pt(joints_orig[9]), pt(joints_orig[10]), size=35)
    toolbox.debug(x)

    x = toolbox.crop(skel_img, toolbox.inject(joints_orig, [6, 7, 8]))
    toolbox.debug(x)

    import sys
    sys.exit()

    import matplotlib.pyplot as plt
    # plt.ion()
    plt.figure(1)
    plt.clf()

    plt.subplot(231)
    plt.imshow(skel_img)
    plt.title('joint projection')
    plt.axis('off')

    cropped = toolbox.crop(skel_img, pts([pBL, pBR, pCC]))

    plt.subplot(232)
    plt.imshow(cropped)
    plt.title('cropped muscle')
    plt.axis('off')

    plt.show()

    # import ipdb
    # ipdb.set_trace()


img, proc_param, joints, verts, cams = joints.parse("data/lewy.jpg")
visualize(img, proc_param, joints, verts, cams)
