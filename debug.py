import cv2
import numpy as np

from hmr.src.util import renderer as vis_util

import joints
import toolbox
from toolbox import pt, pts

def visualize(img, proc_param, joints, verts, cam):
    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])

    skel_img = vis_util.draw_skeleton(img, joints_orig)

    """
    x = toolbox.line_crop(skel_img, \
        pt(joints_orig[9]), pt(joints_orig[10]), size=35)
    toolbox.debug(x)
    """

    x = toolbox.crop(skel_img,
            toolbox.pad(
                toolbox.inject(joints_orig, [2, 3, 8, 9]),
                size=20
            ),
            warped=True)
    toolbox.debug([x, skel_img])
    #toolbox.debug(skel_img)

img, proc_param, joints, verts, cams = joints.parse("data/lewy.jpg")
visualize(img, proc_param, joints, verts, cams)
