import cv2
from PIL import Image
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt


def phi(x):  # ハイパボリック関数
    tau = np.sqrt(12) / np.pi  # 参考論文では未使用
    return np.tanh(x)*(2/tau)

def normalizeMinMax(x, axis=0, epsilon=1E-5):  #  正規化を元に戻す
  vmin = np.min(x, axis)
  vmax = np.max(x, axis)
  return (x - vmin) / (vmax - vmin + epsilon)

if __name__ == '__main__':

    # 画像の読み込み グレースケールで


    img1 = cv2.imread("img1.jpeg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("img2.jpeg", cv2.IMREAD_GRAYSCALE)


    img1 = cv2.imread("Bef1-1.png", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("Aft1-1.png", cv2.IMREAD_GRAYSCALE)

    cv2.imshow("", img1)
    cv2.waitKey(0)
    cv2.imshow("", img2)
    cv2.waitKey(0)

    # 正規化
    img1_std = np.zeros(img1.shape)
    img2_std = np.zeros(img2.shape)
    img1_std = preprocessing.scale(img1)
    img2_std = preprocessing.scale(img2)

    cv2.imshow("", img1_std)
    cv2.waitKey(0)
    cv2.imshow("", img2_std )
    cv2.waitKey(0)
    ica_in1 = np.clip(img1_std * 255, 0, 255).astype(np.uint8)
    ica_in2 = np.clip(img2_std * 255, 0, 255).astype(np.uint8)
    print(ica_in1.shape)
    bg_diff_path = './ica-in-0-2.png'
    cv2.imwrite(bg_diff_path, ica_in1)
    bg_diff_path = './ica-in-1-2.png'
    cv2.imwrite(bg_diff_path, ica_in2)

    # ICA　参考論文の式から作成

    eta = 0.000001  # 学習係数
    I = np.identity(2)  # 2x2単位行列
    W = np.random.rand(2, 2)  # 初期W 2x2のランダム決定
    dW = np.zeros((2, 2))  # 更新式 dW の初期化
    x = np.array([np.ravel(img1_std), np.ravel(img2_std)])  # 入力X[[1次元化img1_std],[1次元化img2_std]]の生成

    # print("mean 1" + str(np.mean(img1_std, axis=1)))
    i = 0
    looplist = range(10000)
    for i in looplist:
        print("Loop: " + str(i))
        print("Bef W: " + str(W))
        y = np.dot(W, x)
        print("This is Y: " + str(y))
        dW = eta * (I - phi(y) @ y.T) @ W  # Wの更新式dW
        print("New dW: " + str(dW))
        W = W + dW
        detW = np.linalg.det(dW)  # dWの行列式
        print("New W: " + str(W))
        print("Det dW:" + str(detW))
        if i > 2000:
            break
        
        print("")


    # 分離信号を元の画像の形に戻す
    y = np.dot(W, x)
    # print(y[0])
    print("img1 shape :" + str(img1.shape))
    y0 = y[0].reshape(img1.shape[0], img1.shape[1])
    # print(y0)
    print("y0 shape :" + str(y0.shape))
    y1 = y[1].reshape(img1.shape[0],img1.shape[1])

    cv2.imshow("", y0)
    cv2.waitKey(0)
    cv2.imshow("", y1)
    cv2.waitKey(0)

    # 正規化を元にもどす
    y0 = normalizeMinMax(y0)
    y1 = normalizeMinMax(y1)
    
    #  表示と保存
    print(y0)
    print(y1)
    cv2.imshow("", y0)
    cv2.waitKey(0)
    cv2.imshow("", y1)
    cv2.waitKey(0)
    y0 = np.clip(y0 * 255, 0, 255).astype(np.uint8)
    y1 = np.clip(y1 * 255, 0, 255).astype(np.uint8)
    bg_diff_path = './ica0-2.png'
    cv2.imwrite(bg_diff_path, y0)
    bg_diff_path = './ica1-2.png'
    cv2.imwrite(bg_diff_path, y1)
    cv2.destroyAllWindows()
