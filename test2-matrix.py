import numpy as np
import numpy.linalg as LA

n = 2 #データ数
img1 = np.array([[1,2,3], [4,5,6],[7,8,9]])
img2 = np.array([[3,2,1], [6,5,4],[3,2,1]])


img1 = np.array([[1,2,3], [4,5,6]])
img2 = np.array([[3,2,1], [6,5,4]])

print(img1.T)

print(np.dot(img1,img1.T))

print("mean0" + str(np.mean(img1,axis=0)))
print("mean1" + str(np.mean(img1,axis=1)))
print(img1.shape)

print(x.shape)
print("")
print(np.mean(x[0]))