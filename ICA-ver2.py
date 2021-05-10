import cv2
from PIL import Image
import numpy as np
import pandas as pd
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

    # 入力画像の正規化を元にもどす
    """
    img1_stda = normalizeMinMax(img1_std)
    img2_stda = normalizeMinMax(img2_std)

    img1_stda = img1_stda.reshape(img1.shape[0], img1.shape[1])
    img2_stda = img2_stda.reshape(img1.shape[0], img1.shape[1])
    cv2.imshow("", img1_stda)
    cv2.waitKey(0)
    cv2.imshow("", img2_stda)
    cv2.waitKey(0)
    bg_diff_path = './ica-ver2-in-0.png'
    cv2.imwrite(bg_diff_path, img1_stda)
    bg_diff_path = './ica-ver2-in-1.png'
    cv2.imwrite(bg_diff_path, img2_stda)
    """

    ica_in1 = img1_std
    ica_in2 = img2_std

    # ICA　参考論文の式から作成
    eta = 0.000001  # 学習係数
    I = np.identity(2)  # 2x2単位行列
    W = np.random.rand(2, 2)  # 初期W 2x2のランダム決定
    dW = np.zeros((2, 2))  # 更新式 dW の初期化
    x = np.array([np.ravel(img1_std), np.ravel(img2_std)])  # 入力X[[1次元化img1_std],[1次元化img2_std]]の生成

    # print("mean 1" + str(np.mean(img1_std, axis=1)))
    flag = 1
    i = 0
    loop = 4000
    looplist = range(loop)
    for i in looplist:
        y = np.dot(W, x)
        dW = eta * (I - phi(y) @ y.T) @ W  # Wの更新式dW
        W = W + dW
        detW = np.linalg.det(dW)  # dWの行列式
        # 分離信号Yにして相関係数見る
        y = np.dot(W, x)
        s1 = pd.Series(y[0])
        s2 = pd.Series(y[1])
        res2 = s1.corr(s2)
        print("Loop: " + str(i) + " Res: " + str(res2))
        if pd.isnull(res2):
            break
        if np.abs(res2) < 0.01:
            break
        if i > loop - 1:
            flag = -1
            break

        print("")

    # 分離信号Yにして相関係数見る
    #y = np.dot(W, x)
    #s1 = pd.Series(y[0])
    #s2 = pd.Series(y[1])
    #res2 = s1.corr(s2)
    if flag < 0:
        path = "L-" + str(i) + "-" + str('{:.5g}'.format(res2)) + ".txt"
        f = open(path,"w")
        print("学習係数 :" + str(eta))
        print("ループ数制限 :" + str(loop))
        print("処理前 相関係数 :" + str(res1))
        print("処理後 相関係数 :" + str(res2))
        f.write("学習係数 :" + str(eta) + "  ")
        f.write("処理前 相関係数 :" + str(res1) + "  ")
        f.write("処理前 相関係数 :" + str(res1) + "  ")

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
        bg_diff_path = "./ica-ver4-" + str('{:.5g}'.format(res2)) + "-0.png"
        cv2.imwrite(bg_diff_path, y0)
        bg_diff_path = "./ica-ver4-" + str('{:.5g}'.format(res2)) + "-1.png"
        cv2.imwrite(bg_diff_path, y1)
        cv2.destroyAllWindows()

        f.close()

