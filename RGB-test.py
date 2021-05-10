import cv2
from PIL import Image
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt


class PCAWhitening:
    def __init__(self, epsilon=1E-6):
        self.epsilon = epsilon
        # self.mean = None
        self.eigenvalue = None
        self.eigenvector = None
        self.pca = None

    def fit(self, x):
        print("Fit IN")
        print("Fit self : " + str(self))
        print("Fit x : " + str(x))
        """"
        self.mean = np.mean(x, axis=0) #列に沿って平均化
        # CIFA-10の場合は画像を複数取り込む
        各列の平均を元から引く
        print("Fit self.mean " + str(self.mean))
        x_ = x - self.mean  # 平均値で引く
        """
        x_ = x
        print("Fit OVR 2" + "shape:¥n" + str(x_.shape) + str(x_))

        cov = np.dot(x_.T, x_)  # 転置と元の積
        print("Fit OVR 3" + "shape:¥n" + str(cov.shape) + str(cov))

        E, D, _ = np.linalg.svd(cov)  # 特異値分解
        print("Fit OVR 4 E: " + str(E))
        print(",D : " + str(D))

        D = np.sqrt(D) + self.epsilon
        self.eigenvalue = D
        self.eigenvector = E
        print("Fit OVR 5")

        self.pca = np.dot(np.diag(1. / D), E.T)
        print("Fit END")
        return self

    def transform(self, x):
        # x_ = x - self.mean
        x_ = x
        return np.dot(x_, self.pca.T)


def normalizeMinMax(x, axis=0, epsilon=1E-5):
    vmin = np.min(x, axis)
    vmax = np.max(x, axis)
    return (x - vmin) / (vmax - vmin + epsilon)


def normalizeImage(x):
    img = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
    img = normalizeMinMax(img, axis=0)
    return img.reshape(x.shape)


def normalizeImage2(x, epsilon=1E-6):
    vmin = np.min(x)
    vmax = np.max(x)
    return (x - vmin) / (vmax - vmin + epsilon)


if __name__ == '__main__':

    # 画像の読み込み
    img = cv2.imread("cat9.png", 1)  # ndarray形式?
    # print(img)

    # 基本表示
    # cv2.imshow('frame',img )
    print("shape 0 :" + str(img.shape[0]))  # たて
    print("shape 1 :" + str(img.shape[1]))  # 横
    print("shape 2 :" + str(img.shape[2]))  # 3がでる　RGBってことか

    r=(img[:,:,0].reshape(1,-1))
    g=(img[:, :, 1].reshape(1, -1))
    b=(img[:, :, 2].reshape(1, -1))
    print(r)
    print(g)
    print(b)

    rgb = np.append(r, g, axis=0)
    rgb = np.append(rgb, b, axis=0)
    print(rgb)
    # print(img_pcaw)