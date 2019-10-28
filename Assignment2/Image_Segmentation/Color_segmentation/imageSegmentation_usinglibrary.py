import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# from skimage.io import imread
# im = imread("original1.jpg") 
# print(im)

img = Image.open('original1.jpg').convert("RGB")
arr = np.array(img)
arr = arr/255

# record the original shape
shape = arr.shape
print(arr.shape)

# make a 1-dimensional view of arr
flat_arr = arr.ravel()
flat_arr = np.asarray(flat_arr).reshape( arr.shape[0]*arr.shape[1] ,3)
# convert it to a matrix
vector = np.matrix(flat_arr)
print(vector.shape)
kmeans = KMeans(n_clusters=15, random_state=0).fit( vector)
print("Kmean : ",kmeans)
pic2show = kmeans.cluster_centers_[kmeans.labels_]

cluster_pic = pic2show.reshape(arr.shape[0], arr.shape[1], arr.shape[2])
print(cluster_pic.shape)
plt.imshow(cluster_pic)
plt.show()