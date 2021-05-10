import cv2
from PIL import Image
import numpy as np
import random
from sklearn import preprocessing
import matplotlib.pyplot as plt
import itertools

def tanh(x):
  return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

if __name__ == '__main__':



    # 画像の読み込み
    print("hello")
    img1 = cv2.imread("img1.jpeg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("img2.jpeg", cv2.IMREAD_GRAYSCALE)

    # 正規化
    img1_std = np.zeros(img1.shape)
    img2_std = np.zeros(img2.shape)
    img1_std = preprocessing.scale(img1)
    img2_std = preprocessing.scale(img1)

    #ICA
    tau = np.sqrt(12)/np.pi
    eta = 0.01
    I = np.identity(2)
    W = np.random.rand(2, 2)
    dW = np.zeros((2, 2))
    x = [[]]
    #print(img1_std.reshape(1,-1))
    #x[0] = (img1_std.reshape(1,-1))
    #x.append(img2_std.reshape(1, -1))
    #print(x)

    x = np.array([np.ravel(img1_std),np.ravel(img2_std)])


    print(x)

    print("shape" + str(len(x)))
    print("shape" + str(x.shape))
    print(tanh(2))
    print(tanh(-2))
    I = np.identity(2)
    print(I)
    print(I[0,0])

    W = np.random.rand(2, 2)
    print(W)
    print(np.linalg.det(W))

