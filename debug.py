from hmr.src.util import renderer as vis_util

import joints


def visualize(img, proc_param, joints, verts, cam):
    """
    Renders the result in original image coordinate frame.
    """
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
    plt.imshow(skel_img)
    plt.title('joint projection')
    plt.axis('off')
    plt.show()
    # import ipdb
    # ipdb.set_trace()


img, proc_param, joints, verts, cams = joints.parse("data/lewy.jpg")
visualize(img, proc_param, joints, verts, cams)
