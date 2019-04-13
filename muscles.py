
from hmr.src.util import renderer as vis_util

import joints
import toolbox

# def half(a, b):
    # return [(a[0]-b[0])/2, (a[1]-b[1])/2]


pb = [8, 7]  # otoczka
lb = [9, 10]  # otoczka
# pk = [8, half(8, 2), half(half(2, 3), 12), 12]
# lk = [9, half(9, 3), half(half(3, 2), 12), 12]
pr = [7, 6]  # otoczka
lr = [10, 11]  # otoczka
pu = [2, 1]  # otoczka grubsza niz ramie czy biceps
lu = [3, 4]  # otoczka grubsza niz ramie czy biceps
pl = [1, 0]  # otoczka
ll = [4, 5]  # otoczka
# lewy_abs = intersect(pole(9, 3, 2), pole(12, 9, 3, half(2, 3)))
# prawy_abs = intersect(pole(8, 3, 2), pole(12, 8, 3, half(3, 2)))

img, proc_param, joints, verts, cam = joints.parse("data/lewy.jpg")

cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
    proc_param, verts, cam, joints, img_size=img.shape[:2])

skel_img = vis_util.draw_skeleton(img, joints_orig)

prawy_biceps = toolbox.crop(skel_img, toolbox.inject(joints_orig, pb))
lewy_biceps = toolbox.crop(skel_img, toolbox.inject(joints_orig, lb))
#prawa_klatka = toolbox.crop(skel_img, toolbox.inject(joints_orig, pk))
#lewa_klatka = toolbox.crop(skel_img, toolbox.inject(joints_orig, lk))
prawe_ramie = toolbox.crop(skel_img, toolbox.inject(joints_orig, pr))
lewe_ramie = toolbox.crop(skel_img, toolbox.inject(joints_orig, lr))
prawe_udo = toolbox.crop(skel_img, toolbox.pad(toolbox.inject(joints_orig, pu), size=3))
lewe_udo = toolbox.crop(skel_img, toolbox.pad(toolbox.inject(joints_orig, lu), size=3))
prawa_lydka = toolbox.crop(skel_img, toolbox.pad(toolbox.inject(joints_orig, pl), size=3))
lewa_lydka = toolbox.crop(skel_img, toolbox.pad(toolbox.inject(joints_orig, ll), size=3))
#lewa_klatka = toolbox.crop(skel_img, toolbox.inject(joints_orig, lk))

toolbox.debug([prawy_biceps, lewy_biceps, prawe_ramie, lewe_ramie, prawe_udo, lewe_udo, \
prawa_lydka, lewa_lydka])
