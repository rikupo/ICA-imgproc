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
    img1 = cv2.imread("bef-ver4.png", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("aft-ver4.png", cv2.IMREAD_GRAYSCALE)

    img3 = cv2.imread("bef-ver4.png")  # 後々用


    # 1列化
    line1 = np.ravel(img1)
    line2 = np.ravel(img2)

    s1 = pd.Series(line1)
    s2 = pd.Series(line2)
    res1 = s1.corr(s2)
    print("処理前 相関係数 :" + str(res1))

    # 正規化
    img1_std = np.zeros(line1.shape)
    img2_std = np.zeros(line2.shape)
    img1_std = preprocessing.scale(line1)
    img2_std = preprocessing.scale(line2)

    ica_in1 = img1_std
    ica_in2 = img2_std

    # ICA　参考論文の式から作成
    I = np.identity(2)  # 2x2単位行列
    x = np.array([np.ravel(img1_std), np.ravel(img2_std)])  # 入力X[[1次元化img1_std],[1次元化img2_std]]の生成
    W = [[-0.00804006,0.00860839],[0.00459728,0.00342655]]  # 事前に算出したW
    D = [[0.555,0],[0,1]]

    W = np.dot(D,W)

    Y= np.dot(W, x)

    A = [[-96.2,131],[125,123]]

    X = np.dot(A,Y)

    sum = normalizeMinMax(X[0])
    sumsq = sum.reshape(img1.shape[0], img1.shape[1])
    cv2.imshow("maybe s0", sumsq)
    cv2.waitKey(0)


    sum = normalizeMinMax(X[1])
    sumsq = sum.reshape(img1.shape[0], img1.shape[1])
    cv2.imshow("maybe s0", sumsq)
    cv2.waitKey(0)


