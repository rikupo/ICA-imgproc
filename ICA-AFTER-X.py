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

    y = np.dot(W, x)

    # 分離信号Yにして相関係数見る
    print("next process")
    s1 = pd.Series(y[0])
    s2 = pd.Series(y[1])
    res2 = s1.corr(s2)

    print("img1 shape :" + str(img1.shape))


    sum = y[0]
    befs0 = pd.Series(img1_std)
    befs1 = pd.Series(img2_std)

    #  W = [[-0.00804006,0.00860839],[0.00459728,0.00342655]] ならy0移動物体 y1背景
    loop = range(100000)
    delta = 0.001
    b = -2
    flag0 = 0
    flag1 = 0
    for i in loop:
        if flag0 == 1 and flag1 ==1:
            break
        sum = b * y[0] + y[1]  # 右ボールを生かすならy[1]*1.8で or y[0]*0.6 y0移動物体 y1背景
        afts = pd.Series(sum)
        sokan1 = befs1.corr(afts)
        sokan0 = befs0.corr(afts)
        print("\rLoop: " + str(i) + " B :" + str(b) + " SOKAN0: " + str(sokan0) + " SOKAN1: " + str(sokan1), end="")
        b = b + delta
        if np.abs(sokan0) > 0.9999 and flag0 == 0:
            print("\n0 - GOOD SOKAN0 K SU")
            print(b)
            bzero = b
            flag0 = 1
        if np.abs(sokan1) > 0.9999 and flag1 == 0:
            print("\n1 - GOOD SOKAN1 K SU")
            print(b)
            bone = b
            flag1 = 1

    # sum = y[0] + y[1]  # 右ボールを生かすならy[1]*1.8で or y[0]*0.6 y0移動物体 y1背景
    # sum = (-1) * (0.4) * y[0] + y[1]  # 左ボールはy[0]*(-0.4)がよさげ

    sum = bzero *y[0] +y[1]
    sum = normalizeMinMax(sum)
    sumsq = sum.reshape(img1.shape[0], img1.shape[1])
    cv2.imshow("bzero", sumsq)
    cv2.waitKey(0)

    sum = bone * y[0] + y[1]
    sum = normalizeMinMax(sum)
    sumsq = sum.reshape(img1.shape[0], img1.shape[1])
    cv2.imshow("bone", sumsq)
    cv2.waitKey(0)


    y[0] = normalizeMinMax(y[0])
    y[1] = normalizeMinMax(y[1])

    print("img1 shape :" + str(img1.shape))
    y0 = y[0].reshape(img1.shape[0], img1.shape[1])
    y1 = y[1].reshape(img1.shape[0], img1.shape[1])


    HotSpot = np.zeros_like((img3))
    print("y0")
    print(y0)

    y0 = np.clip(y0 * 255, 0, 255).astype(np.uint8)
    y1 = np.clip(y1 * 255, 0, 255).astype(np.uint8)
