import cv2
from PIL import Image
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

x = [[1,2,3],[5,6,7],[2,3,4]]
a = np.diag(x)
print(np.diag(x))
print(np.diag(1. / a))

print(np.mean(x,axis=0))

x_ = x - np.mean(x,axis=0)

print(x_)

a = np.array([1,2,3])
c = np.dot(a,a.T)
print(c)

