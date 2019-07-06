import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../imgs/dog.jpg')

#画素値の取得
# px = img[100,100]
# print(px)


# for i in range(100):
#     for j in range(100):
#         img[i,j] = [100,100,100]

plt.imshow(img,"gray")
plt.show()
