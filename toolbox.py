import cv2
import numpy as np
import pyclipper

import math


def debug(imgs, txt="debug"):
    import matplotlib.pyplot as plt
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig = plt.figure(1)
    plt.clf()
    for i, img in enumerate(imgs):
        print(i)
        fig.add_subplot(1, len(imgs), i + 1)
        plt.imshow(img)
        plt.title("[{}] {}".format(i, txt))
        plt.axis('off')
    plt.show()


def normal(points): return [[int(a), int(b)] for a, b in points]


def polysort(points):
    points = normal(points)
    mlat = sum(x[0] for x in points) / len(points)
    mlng = sum(x[1] for x in points) / len(points)

    def __sort(x):  # main math --> found on MIT site
        return (math.atan2(x[0] - mlat, x[1] - mlng) +
                2 * math.pi) % (2 * math.pi)
    points.sort(key=__sort)
    return pts(points)


def pad(points, size=60):
    pco = pyclipper.PyclipperOffset()
    points = polysort(points[0])
    pco.AddPath(points[0],
                pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
    padded_points = pco.Execute(size)[0]
    print("PAD", padded_points)
    return pts(padded_points)


def crop(img, points, warped=True):
    if not warped:
        mask = np.zeros(img.shape, dtype=np.uint8)
        cv2.fillPoly(mask, points, (255))
        print("MASK", mask.shape)
        print("IMG", img.shape)
        res = cv2.bitwise_and(img, img, mask)
        rect = cv2.boundingRect(points)
        return res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    else:
        rect = cv2.minAreaRect(points)
        center, size, angle = rect[0], rect[1], rect[2]
        center, size = tuple(map(int, center)), tuple(map(int, size))
        height, width = img.shape[0], img.shape[1]
        M = cv2.getRotationMatrix2D(center, angle, 1)
        img_rot = cv2.warpAffine(img, M, (width, height))
        return cv2.getRectSubPix(img_rot, size, center)


def line_box(img, pt1, pt2, size=25):
    # FIXME: size -> dynamiczny do dlugosci reki (pt1-pt2)
    mask = np.zeros(img.shape, dtype=np.uint8)
    cv2.line(mask, pt(pt1), pt(pt2), (255, 255, 255), size)
    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    cnt = contours[0]
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    return np.int0(box)

def line_crop(img, pt1, pt2, size=25, warped=True):
    box = line_box(img, pt1, pt2, size=size)
    return crop(img, pts(box), warped=warped)

def inject(points, idx_points):
    return pts([points[idx] for idx in idx_points])


def pt(point):
    return (int(point[0]), int(point[1]))


def pts(points):
    for i in range(0, len(points)):
        points[i] = list(map(int, np.array(points[i]).flatten()))
    return np.array([points], dtype=np.int32)
