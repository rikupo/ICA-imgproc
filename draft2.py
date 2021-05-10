import cv2
from PIL import Image
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # 画像の読み込み
    img = cv2.imread("cat9.png")
    print(img)

    print("shape 0 :" + str(img.shape[0]))
    print("shape 1 :" + str(img.shape[1]))
    print("shape 2 :" + str(img.shape[2]))

    cv2.imshow("",img[:, :, :])
    cv2.waitKey(0)

    # 画像の読み込み
    img = cv2.imread("PCAW-non-normalize.png")
    print(img)

    print("shape 0 :" + str(img.shape[0]))
    print("shape 1 :" + str(img.shape[1]))
    print("shape 2 :" + str(img.shape[2]))

    cv2.imshow("",img[:, :, :])
    cv2.waitKey(0)


