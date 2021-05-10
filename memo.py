import cv2
from PIL import Image
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

def normalizeImage2(x, epsilon=1E-6):
  vmin = np.min(x)
  vmax = np.max(x)
  return (x - vmin) / (vmax - vmin + epsilon)

x_pcaw = x_train.reshape(x_train.shape[0], -1)
# 元はtrain[画像Np,R,G,B]なので1画像のRGBを全部1行にまとめたんを画像数分の行に入れる
print('x_pcaw.shape=' + str(x_pcaw.shape))
# x_pcaw.shape(50000, 3072)
pcaw = PCAWhitening().fit(x_pcaw)
x_pcaw = pcaw.transform(x_pcaw).reshape(x_train.shape)

plt.clf()
for i in range(0, 16):
  plt.subplot(4, 8, i*2+1)
  fig = plt.imshow(x_train[i,:,:,:])
  fig.axes.get_xaxis().set_visible(False)
  fig.axes.get_yaxis().set_visible(False)
  plt.subplot(4, 8, i*2+2)
  fig = plt.imshow(normalizeImage2(x_pcaw[i,:,:,:]))
  fig.axes.get_xaxis().set_visible(False)
  fig.axes.get_yaxis().set_visible(False)

plt.savefig('cifar10_pcaw.png')


import cv2
from PIL import Image
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt


class PCAWhitening:
    def __init__(self, epsilon=1E-6):
        self.epsilon = epsilon
        self.eigenvalue = None
        self.eigenvector = None
        self.pca = None

    def fit(self, x):
        cov = np.dot(x.T, x)/1024      # 転置と元の積
        E, D, _ = np.linalg.svd(cov) # 特異値分解
        D = np.sqrt(D) + self.epsilon
        self.eigenvalue = D
        self.eigenvector = E
        self.pca = np.dot(np.diag(1. / D), E.T)
        return self

    def transform(self, x):
        return np.dot(x, self.pca.T)

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
    print("shape 0 :" + str(img.shape[0])) #たて
    print("shape 1 :" + str(img.shape[1])) #　横
    print("shape 2 :" + str(img.shape[2]))  # 3がでる　RGBってことか

    # cv2.imshow("",img[:, :, :])
    # cv2.waitKey(0)

    # 正規化
    img_std = np.zeros(img.shape)
    for i in range(0,img.shape[2]):
        img_std[:, :, i] = preprocessing.scale(img[:, :, i])
    # sklearnのscaleで各RGBを正規化

    # PCAW
    # img_pcaw = img_std.reshape(1, -1)  # 1行の行列にする

    r=(img[:,:,0].reshape(1,-1))
    g=(img[:, :, 1].reshape(1, -1))
    b=(img[:, :, 2].reshape(1, -1))
    print(r)
    print(g)
    print(b)
    rgb = np.append(r, g, axis=0)
    rgb = np.append(rgb, b, axis=0)

    img_pcaw = rgb

    # print(img_pcaw)
    pcaw = PCAWhitening().fit(img_pcaw)
    print("PCAW 1st END")
    print(pcaw) # オブジェクトが帰ってくる
    img_pcaw = pcaw.transform(img_pcaw).reshape(img.shape)
    print("PCAW END")

    print(img_pcaw)
    cv2.imshow("", img_pcaw)
    cv2.waitKey(0)
    bg_diff_path  = './PCAW-non-normalize.png'

    cv2.imwrite(bg_diff_path,img_pcaw)
    img_pcaw = normalizeImage2(img_pcaw)
    print(img_pcaw)

    cv2.imshow("", img_pcaw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    bg_diff_path  = './PCAW.png'
    cv2.imwrite(bg_diff_path,img_pcaw)

    # print(img_std[0,0])
    # print(img_std[0,0,0])




    cv2.imshow("", img1)
    cv2.waitKey(0)
    print(img1)
    img1 = np.round(0.9*img1)
    print(img1)
    cv2.imshow("", img1)
    cv2.waitKey(0)
    img1 = 0.001*img1
    print(img1)
    cv2.imshow("", img1)
    cv2.waitKey(0)

    for n in range(0, i):
        print("-")
        if n % 9 == 0:
            print("")
    print('\r', end='')


    while True:
        i += i
        W = W + eta*np.dot((I-np.dot(tanh(y),y.T)),W.T)

        if i > 1000:
            break

            print(img1_std)
            print("")
            print(np.ravel(img1_std))
            print("")
