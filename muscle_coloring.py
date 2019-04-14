import numpy as np
import cv2


def ReLU(x):
    return x * (x > 0)


ziomek = cv2.imread('webapp/ziomek/ziomek.jpeg')
lewa_klatka_mask = cv2.imread('webapp/ziomek/lewa_klatka.png')
prawa_klatka_mask = cv2.imread('webapp/ziomek/prawa_klatka.png')
prawa_lydka_mask = cv2.imread('webapp/ziomek/prawa_lydka.png')
lewa_lydka_mask = cv2.imread('webapp/ziomek/lewa_lydka.png')
prawe_ramie_mask = cv2.imread('webapp/ziomek/prawe_ramie.png')
lewe_ramie_mask = cv2.imread('webapp/ziomek/lewe_ramie.png')
prawe_udo_mask = cv2.imread('webapp/ziomek/prawe_udo.png')
lewe_udo_mask = cv2.imread('webapp/ziomek/lewe_udo.png')
prawy_abs_mask = cv2.imread('webapp/ziomek/prawy_abs.png')
lewy_abs_mask = cv2.imread('webapp/ziomek/lewy_abs.png')
prawy_biceps_mask = cv2.imread('webapp/ziomek/prawy_biceps.png')
lewy_biceps_mask = cv2.imread('webapp/ziomek/lewy_biceps.png')

cialo = [
lewa_klatka_mask,
prawa_klatka_mask,
prawa_lydka_mask,
lewa_lydka_mask,
prawe_ramie_mask,
lewe_ramie_mask,
prawe_udo_mask,
lewe_udo_mask,
prawy_abs_mask,
lewy_abs_mask,
prawy_biceps_mask,
lewy_biceps_mask]

wagi = [
1.0,
-0.2,
0.3,
-0.1,
1.0,
0.5,
-1.0,
0.3,
0.7,
0.3,
-0.2,
0.2
]

dimensions = ziomek.shape
color = np.zeros((dimensions[0], dimensions[1], 3), np.uint8)

for i in range(12):
    color[:, :] = (0, 255 * (wagi[i] > 0.0), 255 * (wagi[i] < 0.0))  # (B, G, R)
    mask = cv2.bitwise_and(cialo[i], color)
    ziomek = cv2.bitwise_or(ziomek, mask)


cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', ziomek)
cv2.waitKey(0)
cv2.destroyAllWindows()
