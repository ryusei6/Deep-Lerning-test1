#-*- coding:utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt


def pass_to_laplacian(img):
    kernel = np.array([[0, 1,  0],
                       [1, -4, 1],
                       [0, 1,  0]])

    output = cv2.filter2D(img, -1, kernel)
    return output

def main():
    # 画像読み込み
    gray = cv2.imread("lena.jpg", 0)

    # Laplacianフィルタ
    # output = cv2.Laplacian(gray,cv2.CV_32F)
    # output = cv2.convertScaleAbs(output)
    output = pass_to_laplacian(gray)

    # COLOR_BGR2RGB
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    # 結果を出力
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.imshow(gray)
    ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, left=False, bottom=False)
    ax2 = fig.add_subplot(122)
    ax2.imshow(output)
    ax2.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, left=False, bottom=False)
    plt.show()


if __name__ == "__main__":
    main()
