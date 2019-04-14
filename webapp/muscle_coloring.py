import numpy as np
import cv2

ziomek = cv2.imread('ziomek/ziomek.jpeg')
lewa_klatka_mask = cv2.imread('ziomek/lewa_klatka.png')
prawa_klatka_mask = cv2.imread('ziomek/prawa_klatka.png')
prawa_lydka_mask = cv2.imread('ziomek/prawa_lydka.png')
lewa_lydka_mask = cv2.imread('ziomek/lewa_lydka.png')
prawe_ramie_mask = cv2.imread('ziomek/prawe_ramie.png')
lewe_ramie_mask = cv2.imread('ziomek/lewe_ramie.png')
prawe_udo_mask = cv2.imread('ziomek/prawe_udo.png')
lewe_udo_mask = cv2.imread('ziomek/lewe_udo.png')
prawy_abs_mask = cv2.imread('ziomek/prawy_abs.png')
lewy_abs_mask = cv2.imread('ziomek/lewy_abs.png')
prawy_biceps_mask = cv2.imread('ziomek/prawy_biceps.png')
lewy_biceps_mask = cv2.imread('ziomek/lewy_biceps.png')

# green = np.uint8(ziomek.shape, [[[0, 255, 0]]])
green = np.array(ziomek.shape, [0, 255, 0])
# mask = np.zeros(img.shape, dtype = "uint8")
ziomek = cv2.bitwise_or(ziomek, green)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', ziomek)
cv2.waitKey(0)
cv2.destroyAllWindows()
