import sys
from PIL import Image
import glob
import cv2

import numpy as np
import pandas as pd
import struct
from numpy import linalg as LA
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')  

image_no = 100
Total_components = 784      # 28*28 pixel images
print("Hey!, We are using MNISt Data. So there are total 784 components.")
number_of_components = 784

X_data = np.zeros(shape=(28,28), dtype=int)
files = glob.glob("./data/processed/images/train/0_*")
data = np.array([np.array(Image.open(fname)) for fname in files])
print("Shape of original data ", data.shape)

# ---- extract the data from IDX(Binary) files into a numpy array --- 
    # with open('./Data_IDX_Format/train-images-idx3-ubyte','rb') as f:
    #     magic, size = struct.unpack(">II", f.read(8))
    #     nrows, ncols = struct.unpack(">II", f.read(8))
    #     data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    #     data = data.reshape((size, nrows, ncols))
    #     print("Shape of original data ", data.shape)
# ---- extract the data from IDX(Binary) files into a numpy array ---

# ---------- Show the image ------------
import matplotlib.pyplot as plt
plt.imshow(data[ image_no,:,:],cmap='gray')
plt.title('Original Image ( ' + str(Total_components) + ' eigenVectors) ')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.savefig("./Results/PCAReconstruct/OriginalImage.jpg")
# plt.show()
# ---------- Show the image ------------

# flatten the 3d array into 2d 
# i.e. convert a 28*28 image into 784 sized vector
flat_arr = data.ravel()
flat_arr = np.asarray(flat_arr).reshape(data.shape[0], data.shape[1]*data.shape[2])
print("Shape of flattened data ,",flat_arr.shape)

flat_arr = flat_arr

# Compute mean vector
def mean(Data):
    sum_vector = np.zeros(shape=(Data.shape[1]),dtype = int)
    mean_vector = np.zeros(shape=(Data.shape[1]),dtype= int)
    for sample in Data:
        sum_vector = sum_vector + sample
    
    mean_vector = (sum_vector)/int(Data.shape[0])
    return mean_vector

# Compute energy value
def Energy(eigenvalues, K):
    EnergyValue = 0
    count = 0
    for i in eigenValues :
        EnergyValue = EnergyValue + pow(i,2)
        count = count + 1
        if count == K:
            break 
    return EnergyValue

# mean = np.mean(flat_arr, axis = 0)
mean = mean(flat_arr)
print("Shape of mean vector ", mean.shape)
X = flat_arr
X = X - mean

print("Shape of X ,", X.shape)

# finding covariance matrix , transpose(X)*X
cov_mat = np.matmul(X.transpose(),X   )
# cov_mat = np.cov(flat_arr.transpose())

print("Shape of covariance matrix :", cov_mat.shape)
eigenValues, eigenVectors = LA.eigh(cov_mat)
print("Shape of eigen vectors ",eigenVectors.shape)

# sort the eigenvectors according to eigen valuess
index = eigenValues.argsort()[::-1]   
eigenValues = eigenValues[index]

eigenVectors = eigenVectors[:,index]


# <<<<<<<<------------case 1-------------------!

print("Number of components in case 1", number_of_components)
eigenVectors = eigenVectors[:,:-(Total_components- (number_of_components-1))]

# Project the input image to eigen space
projected_data = np.matmul(X , eigenVectors)
# reconstruct the original image using eigen vectors
reconstruct_data  =np.matmul(projected_data, eigenVectors.transpose())
reconstruct_data = reconstruct_data +mean
reconstruct_data = reconstruct_data.reshape(reconstruct_data.shape[0],data.shape[1],  data.shape[2])

plt.imshow( reconstruct_data[image_no,:,:], cmap='gray')
plt.title('Reconstructed Image ( ' + str(number_of_components) + ' eigenVectors)')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.savefig("./Results/PCAReconstruct/ReconstructImage_all_EVs.jpg")

images = images = [Image.open(x) for x in ['./Results/PCAReconstruct/OriginalImage.jpg', './Results/PCAReconstruct/ReconstructImage_all_EVs.jpg']]
widths, heights = zip(*(i.size for i in images))

total_width = sum(widths)
max_height = max(heights)
new_im = Image.new('RGB', (total_width, max_height))
x_offset =0
for im in images:
  new_im.paste(im, (x_offset,0))
  x_offset += im.size[0]

new_im.show()
new_im.save('./Results/PCAReconstruct/PCA_all_EVs.jpg')

# -------------case1 ------------------>>>>>>>

# <<<<<<< --------- case2 -----------------!

# Select top K eigen vectorS with Energy ratio
Total_Energy = Energy(eigenValues, Total_components)
Effective_numberOfComponents = 0
x_list = []
y_list = []
for k in range(1,Total_components):
    Energy_for_K = Energy(eigenValues, k)

    y_list.append(Energy_for_K/ Total_Energy)
    x_list.append(k)
    if Energy_for_K/Total_Energy >= 0.99 :
        
        Effective_numberOfComponents = k
        break  
plt.close()
plt.title("Energy Graph")
plt.xlabel('Number of components (K)')
plt.ylabel('Ratio of Energy at K components to All components')
plt.plot(x_list, y_list , label='number of components')
plt.legend()
plt.savefig('./Results/PCAReconstruct/EnergyPlot.jpg') 
plt.show()
print("Number of Top eigen vectors ", Effective_numberOfComponents)
eigenVectors = eigenVectors[:,:-(Total_components- (Effective_numberOfComponents-1))]

# Project the input image to eigen space
projected_data = np.matmul(X , eigenVectors)
# reconstruct the original image using eigen vectors
reconstruct_data  =np.matmul(projected_data, eigenVectors.transpose())
reconstruct_data = reconstruct_data + mean
reconstruct_data = reconstruct_data.reshape(reconstruct_data.shape[0],data.shape[1],  data.shape[2])
# new =  np.concatenate((data, reconstruct_data), axis = 2)

plt.imshow( reconstruct_data[image_no,:,:], cmap='gray')
plt.title('Reconstructed Image ( ' + str(Effective_numberOfComponents) + ' eigenVectors)')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.savefig("./Results/PCAReconstruct/ReconstructImage_Effective_EVs.jpg")
# plt.show()

images = images = [Image.open(x) for x in ['./Results/PCAReconstruct/OriginalImage.jpg', './Results/PCAReconstruct/ReconstructImage_Effective_EVs.jpg']]
widths, heights = zip(*(i.size for i in images))

total_width = sum(widths)
max_height = max(heights)
new_im = Image.new('RGB', (total_width, max_height))
x_offset =0
for im in images:
  new_im.paste(im, (x_offset,0))
  x_offset += im.size[0]

new_im.show()
new_im.save('./Results/PCAReconstruct/PCA_Effective_EVs.jpg')

# ---------------- case 2 ---------------->>>>>>>>>>>>


