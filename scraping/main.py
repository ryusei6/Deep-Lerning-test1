import numpy as np
import matplotlib.pyplot as plt

file_name = './data/imgs/LarryPage_face/0001_1_cutted.jpg'
img = plt.imread(file_name)
R, G, B = img[...,0], img[...,1], img[...,2]
img_gray = (0.298912 * R + 0.586611 * G + 0.114478 * B) # NTSC加重平均法(OpenCVもこれ)

def img_show():
    plt.figure(figsize=(7,5))
    plt.axis("off")
    plt.imshow(img_gray.reshape(img_gray.shape[0],img_gray.shape[1]),cmap="gray")
    plt.show()

img_show()
