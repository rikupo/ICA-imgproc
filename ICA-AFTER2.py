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
    img1 = cv2.imread("TEST111.png", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("TEST111.png", cv2.IMREAD_GRAYSCALE)

    img3 = cv2.imread("s5-1.png")  # 後々用

    #bg_diff_path = './ica-ver2-in-gray-0.png'
    #cv2.imwrite(bg_diff_path, img1)
    #bg_diff_path = './ica-ver2-in-gray-1.png'
    #cv2.imwrite(bg_diff_path, img2)

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

    W = [[-0.00797417 ,0.00852957],[0.00450137,0.00334061]] # W for series 5
    W = [[-0.00576506 ,0.00606348],[0.00252791,0.00169344]] # W for series 5 half eta

    y = np.dot(W, x)



    # 分離信号Yにして相関係数見る
    print("next process")
    s1 = pd.Series(y[0])
    s2 = pd.Series(y[1])
    res2 = s1.corr(s2)

    y[0] = normalizeMinMax(y[0])
    y[1] = normalizeMinMax(y[1])

    print("img1 shape :" + str(img1.shape))
    y0 = y[0].reshape(img1.shape[0], img1.shape[1])
    y1 = y[1].reshape(img1.shape[0], img1.shape[1])
    print(y.shape)
    print("THATS Y")
    print(y0)
    print(y1)

    HotSpot = np.zeros_like((img3))

    y0 = np.clip(y0 * 255, 0, 255).astype(np.uint8)
    y1 = np.clip(y1 * 255, 0, 255).astype(np.uint8)

    loop = range(img1.shape[0])
    for i in loop:
        for j in loop:
            if y0[i,j] > 220:
                HotSpot[i,j] = [0,0,255]
                # print("RED")
            if y0[i,j] < 70:
                HotSpot[i,j] = [255,0,0]
                # print("BLUE")
            if 220 > y0[i,j] > 70:
                HotSpot[i, j] = [255, 255, 255] # 普通のところは真っ白
                # print("ELSE")

    i = 7777  # 縁起の良い数字
    path = "L-S5-" + str(i) + "-" + str('{:.5g}'.format(res2)) + ".txt"
    print(HotSpot)
    print("HS shape :" + str(HotSpot.shape))
    cv2.imshow("", HotSpot)
    cv2.waitKey(0)

    """
    bg_diff_path = "./ica-ver4-done-" + str('{:.5g}'.format(res2)) + "-0.png"
    cv2.imwrite(bg_diff_path, y0)
    bg_diff_path = "./ica-ver4-done-" + str('{:.5g}'.format(res2)) + "-1.png"
    cv2.imwrite(bg_diff_path, y1)
    cv2.destroyAllWindows()
    """
