import cv2
from PIL import Image
import numpy as np
import pandas as pd
import sys
from sklearn import preprocessing
import matplotlib.pyplot as plt


def phi(x):  # ハイパボリック関数
    tau = np.sqrt(12) / np.pi  # 参考論文では未使用
    return np.tanh(x) * (2 / tau)


def normalizeMinMax(x, axis=0, epsilon=1E-5):  # 正規化を元に戻す
    vmin = np.min(x, axis)
    vmax = np.max(x, axis)
    return (x - vmin) / (vmax - vmin + epsilon)


if __name__ == '__main__':

    # 画像の読み込み グレースケールで
    img1 = cv2.imread("ball-bef.png", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("ball-aft.png", cv2.IMREAD_GRAYSCALE)
    # print(img1.shape)
    img3 = cv2.imread("bef-ver4.png")
    img4 = cv2.imread("aft-ver4.png")

    print(img1.shape)
    print(img3.shape)
    print(img3)

