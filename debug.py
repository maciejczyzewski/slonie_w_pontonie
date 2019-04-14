import cv2
import numpy as np

from hmr.src.util import renderer as vis_util

import joints
import toolbox
from toolbox import pt, pts


def visualize(img, proc_param, _joints, verts, cam):
    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, _joints, img_size=img.shape[:2])

    preimg = toolbox.premask(img, cam_for_render, vert_shifted, joints_orig)
    # toolbox.debug([img, preimg])

    skel_img = vis_util.draw_skeleton(img, joints_orig)

    from muscles import parse_file_for_muscles
    muscles_imgs = parse_file_for_muscles(img)
    toolbox.debug(list(muscles_imgs.values()))

    x = toolbox.crop(preimg,
                     toolbox.pad(
                         toolbox.inject(joints_orig, [2, 3, 8, 9]),
                         size=0
                     ),
                     warped=True)
    y = toolbox.crop(preimg,
                     toolbox.pad(
                         toolbox.inject(joints_orig, [2, 3, 8, 9]),
                         size=-20
                     ),
                     warped=True)
    toolbox.debug([x, y, preimg])


img, proc_param, _joints, verts, cams = joints.parse("input/lewy.jpg")
visualize(img, proc_param, _joints, verts, cams)
