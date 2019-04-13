import cv2
import numpy as np

from hmr.src.util import renderer as vis_util

import joints
import toolbox
from toolbox import pt, pts

"""
def premask(img, cam_for_render, vert_shifted):
    mask = np.zeros(img.shape, dtype=np.uint8)
    renderer = vis_util.SMPLRenderer(face_path=joints.config.smpl_face_path)
    rend_img_overlay = renderer(
        vert_shifted, cam=cam_for_render, img=mask, do_alpha=False)

    # rend_img_overlay[0<rend_img_overlay] = 1

    black_pixels_mask = np.all(rend_img_overlay == [0, 0, 0], axis=-1)
    non_black_pixels_mask = np.any(rend_img_overlay != [0, 0, 0], axis=-1)
    # or non_black_pixels_mask = ~black_pixels_mask

    image_copy = rend_img_overlay.copy()
    image_copy[black_pixels_mask] = [255, 255, 255]
    image_copy[non_black_pixels_mask] = [0, 0, 0]

    #kernel = np.ones((2,2),np.uint8)
    #image_copy = cv2.erode(image_copy,kernel,iterations = 1)

    image_copy = ~image_copy
    return cv2.bitwise_and(img, image_copy)
"""

def visualize(img, proc_param, _joints, verts, cam):
    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, _joints, img_size=img.shape[:2])

    """
    mask = np.zeros(img.shape, dtype=np.uint8)
    renderer = vis_util.SMPLRenderer(face_path=joints.config.smpl_face_path)
    rend_img_overlay = renderer(
        vert_shifted, cam=cam_for_render, img=mask, do_alpha=False)

    # rend_img_overlay[0<rend_img_overlay] = 1

    black_pixels_mask = np.all(rend_img_overlay == [0, 0, 0], axis=-1)
    non_black_pixels_mask = np.any(rend_img_overlay != [0, 0, 0], axis=-1)
    # or non_black_pixels_mask = ~black_pixels_mask

    image_copy = rend_img_overlay.copy()
    image_copy[black_pixels_mask] = [255, 255, 255]
    image_copy[non_black_pixels_mask] = [0, 0, 0]

    #kernel = np.ones((2,2),np.uint8)
    #image_copy = cv2.erode(image_copy,kernel,iterations = 1)

    image_copy = ~image_copy
    preimg = cv2.bitwise_and(img, image_copy)
    
    toolbox.debug([rend_img_overlay, image_copy, preimg])
    """
    preimg = toolbox.premask(img, cam_for_render, vert_shifted)
    toolbox.debug([img, preimg])


    skel_img = vis_util.draw_skeleton(img, joints_orig)

    x = toolbox.crop(skel_img,
                     toolbox.pad(
                         toolbox.inject(joints_orig, [2, 3, 8, 9]),
                         size=0
                     ),
                     warped=True)
    y = toolbox.crop(skel_img,
                     toolbox.pad(
                         toolbox.inject(joints_orig, [2, 3, 8, 9]),
                         size=-20
                     ),
                     warped=True)
    #toolbox.debug([x, y, skel_img])


img, proc_param, _joints, verts, cams = joints.parse("input/lewy.jpg")
visualize(img, proc_param, _joints, verts, cams)
