import cv2
from PIL import Image
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

def plot_rgb(x):
  vmax = np.max(np.abs(x[:,:,:]))
  plt.plot(x[:,:,0].flatten(), x[:,:,1].flatten(), 'gx')
  plt.plot(x[:,:,0].flatten(), x[:,:,2].flatten(), 'bx')
  plt.xlim(-vmax, vmax)
  plt.ylim(-vmax, vmax)

x_train = cv2.imread("cat9.png")
x_pcaw = cv2.imread("PCAW-non-normalize.png")

plt.clf()
plt.subplot(2, 2, 1)
plt.title('Before scaling')
plt.plot(x_train[:,:,0].flatten(), x_train[:,:,1].flatten(), 'gx')
plt.plot(x_train[:,:,0].flatten(), x_train[:,:,2].flatten(), 'bx')

plt.subplot(2, 2, 2)
plt.title('PCA Whitening')
plot_rgb(x_pcaw)
plt.show()
plt.savefig('cifar10_image_rgb2.png')