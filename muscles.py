
from hmr.src.util import renderer as vis_util

import joints
import toolbox
from toolbox import pt, pts

img, proc_param, joints, verts, cam = joints.parse("data/lewy.jpg")

cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
    proc_param, verts, cam, joints, img_size=img.shape[:2])


def half(x):
    a, b = x[0]
    return [(a[0]+b[0])/2, (a[1]+b[1])/2]


pb = [7, 8]  # otoczka
lb = [9, 10]  # otoczka
pr = [7, 6]  # otoczka
lr = [10, 11]  # otoczka
pu = [2, 1]  # otoczka grubsza niz ramie czy biceps
lu = [3, 4]  # otoczka grubsza niz ramie czy biceps
pl = [1, 0]  # otoczka
ll = [4, 5]  # otoczka
# lewy_abs = intersect(pole(9, 3, 2), pole(12, 9, 3, half(2, 3)))
# prawy_abs = intersect(pole(8, 3, 2), pole(12, 8, 3, half(2, 3)))

In = toolbox.inject

# toolbox.crop( pts(half(In(pr)) + In(pr[0])) )

skel_img = vis_util.draw_skeleton(img, joints_orig)
p23 = half(In(joints_orig, [2, 3]))
p12_23 = half(pts([In(joints_orig, [12])[0], [p23]]))
p28 = half(In(joints_orig, [2, 8]))
p39 = half(In(joints_orig, [3, 9]))
pkla = pts([*In(joints_orig, [12, 8])[0], p12_23, p28])
lkla = pts([*In(joints_orig, [12, 9])[0], p12_23, p39])
labs = pts([*In(joints_orig, [2])[0], p28, p12_23, p23])
pabs = pts([*In(joints_orig, [3])[0], p39, p12_23, p23])

prawy_biceps = toolbox.line_crop(skel_img, *toolbox.inject(joints_orig, pb)[0], size=50)
lewy_biceps = toolbox.line_crop(skel_img, *toolbox.inject(joints_orig, lb)[0], size=50)
prawa_klatka = toolbox.crop(skel_img, pkla)
lewa_klatka = toolbox.crop(skel_img, lkla)
prawe_ramie = toolbox.line_crop(skel_img, *toolbox.inject(joints_orig, pr)[0], size=40)
lewe_ramie = toolbox.line_crop(skel_img, *toolbox.inject(joints_orig, lr)[0], size=40)
prawe_udo = toolbox.line_crop(skel_img, *toolbox.inject(joints_orig, pu)[0], size=60)
lewe_udo = toolbox.line_crop(skel_img, *toolbox.inject(joints_orig, lu)[0], size=60)
prawa_lydka = toolbox.line_crop(skel_img, *toolbox.inject(joints_orig, pl)[0], size=40)
lewa_lydka = toolbox.line_crop(skel_img, *toolbox.inject(joints_orig, ll)[0], size=40)
lewy_abs = toolbox.crop(skel_img, labs)
prawy_abs = toolbox.crop(skel_img, pabs)


toolbox.debug([lewy_abs, prawy_abs, prawa_klatka, lewa_klatka, prawy_biceps, lewy_biceps, \
prawe_ramie, lewe_ramie, prawe_udo, lewe_udo, prawa_lydka, lewa_lydka])
