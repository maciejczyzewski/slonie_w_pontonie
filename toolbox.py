import cv2
import numpy as np

def debug(img, txt="debug"):
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.clf()
    plt.subplot(231)
    plt.imshow(img)
    plt.title(txt)
    plt.axis('off')
    plt.show()

def crop(img, points):
    mask = np.zeros(img.shape, dtype=np.uint8)
    cv2.fillPoly(mask, points, (255))
    print("MASK", mask.shape)
    print("IMG", img.shape)
    res = cv2.bitwise_and(img,img,mask)
    rect = cv2.boundingRect(points)
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    return cropped

def line_crop(img, pt1, pt2, size=25):
    # FIXME: size -> dynamiczny do dlugosci reki
    mask = np.zeros(img.shape, dtype=np.uint8)
    cv2.line(mask, pt(pt1), pt(pt2), (255,255,255), size)
    ret,thresh = cv2.threshold(mask,127,255,0)
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    contours,hierarchy = cv2.findContours(thresh, 1, 2)
    cnt = contours[0]
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return crop(img, pts(box))

def pt(point):
    return (int(point[0]), int(point[1]))

def pts(points):
    for i in range(0, len(points)):
        points[i] = list(map(int, np.array(points[i]).flatten()))
    return np.array([points], dtype=np.int32)
