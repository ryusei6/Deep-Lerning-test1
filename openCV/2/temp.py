import cv2
import numpy as np

img = cv2.imread('imgs/dog.jpg')

#画素値の取得
px = img[100,100]
print(px)

# accessing only blue pixel
blue = img[100,100,0]
print(blue)
