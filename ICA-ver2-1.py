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

    img1 = cv2.imread("WB-aft.PNG", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("WB-bef.PNG" , cv2.IMREAD_GRAYSCALE)

    img1 = img1 // 100

    cv2.imshow("0", img1)
    cv2.waitKey(0)
    cv2.imshow("1", img2)
    cv2.waitKey(0)

    bg_diff_path = './WBL-GRAY0.png'
    cv2.imwrite(bg_diff_path, img1)
    bg_diff_path = './WBL-GRAY1.png'
    cv2.imwrite(bg_diff_path, img2)

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
    W = np.random.rand(2, 2)  # 初期W 2x2のランダム決定
    W = [[0.00804006, -0.00860839],[0.00459728, 0.00342655]] # 事前に算出したW series 4 Q

    # W = [[-0.00797417 ,0.00852957],[0.00450137,0.00334061]] # W for series 5 eta 0.000001
    # W = [[-0.00576506 ,0.00606348],[0.00252791,0.00169344]] # W for series 5 doubled eta

    dW = np.zeros((2, 2))  # 更新式 dW の初期化
    eta = 0.00001  # 学習係数 0.000001ながい　桁1つかえると解を飛び越える
    # print("mean 1" + str(np.mean(img1_std, axis=1)))
    flag = -10
    i = 0
    loop = 10000
    dWth = 1.0e-10
    looplist = range(loop)
    y = np.dot(W, x)
    for i in looplist:
        dW = eta * (I - phi(y) @ y.T) @ W  # Wの更新式dW
        W = W + dW
        detdW = np.linalg.det(dW)  # dWの行列式
        detW = np.linalg.det(W)
        KIKAKU = detdW / detW

        # 分離信号Yにして相関係数見る
        y = np.dot(W, x)
        s1 = pd.Series(y[0])
        s2 = pd.Series(y[1])
        res2 = s1.corr(s2)

        print("\rLoop: " + str(i) + " Res: " + str(res2) + " KIKAU: " + str('{:.5e}'.format(KIKAKU)))
        if np.abs(KIKAKU) < dWth and np.abs(res2) < 0.01:
            flag = 5
            break
    flag=5;
    # 分離信号Yにして相関係数見る
    #y = np.dot(W, x)
    #s1 = pd.Series(y[0])
    #s2 = pd.Series(y[1])
    #res2 = s1.corr(s2)
    print("next process")
    print(flag)
    s1 = pd.Series(y[0])
    s2 = pd.Series(y[1])
    res2 = s1.corr(s2)
    # flag = -100 # safe mode
    if flag > -2:
        print(W)
        path = "L-S5-1-" + str(i) + "-" + str('{:.5g}'.format(res2)) + ".txt"
        f = open(path,"w")
        print("学習係数 :" + str(eta))
        print("ループ数制限 :" + str(loop))
        print("処理前 相関係数 :" + str(res1))
        print("処理後 相関係数 :" + str(res2))
        f.write("学習係数 :" + str(eta) + " \n ")
        f.write("dW変化しきい値 :" + str(dWth) + " \n ")
        f.write("処理前 相関係数 :" + str(res1) + " \n ")
        f.write("処理後 相関係数 :" + str(res2) + " \n ")
        f.write("W:" + str(W) + " \n ")

        # 正規化を元にもどす
        y[0] = normalizeMinMax(y[0])
        y[1] = normalizeMinMax(y[1])

        print("img1 shape :" + str(img1.shape))
        y0 = y[0].reshape(img1.shape[0], img1.shape[1])

        print("y0 shape :" + str(y0.shape))
        y1 = y[1].reshape(img1.shape[0], img1.shape[1])

        #  保存
        cv2.imshow("", y0)
        cv2.waitKey(0)
        cv2.imshow("", y1)
        cv2.waitKey(0)

        y0 = np.clip(y0 * 255, 0, 255).astype(np.uint8)
        y1 = np.clip(y1 * 255, 0, 255).astype(np.uint8)
        bg_diff_path = "./ica-ver5-1-Q2-" + str('{:.5g}'.format(res2)) + "-0.png"
        cv2.imwrite(bg_diff_path, y0)
        bg_diff_path = "./ica-ver5-1-Q2-" + str('{:.5g}'.format(res2)) + "-1.png"
        cv2.imwrite(bg_diff_path, y1)
        cv2.destroyAllWindows()

        f.close()

