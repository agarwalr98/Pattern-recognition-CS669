import numpy as np
from PIL import Image
from matplotlib import colors

import matplotlib.pyplot as plt

# from skimage.io import imread
# im = imread("original1.jpg") 

img = Image.open('original1.jpg').convert("RGB")
arr = np.array(img)

# record the original shape
shape = arr.shape
print("original shape of Image : ", arr.shape)

rows = shape[0]
cols = shape[1]
# make a 1-dimensional view of arr
flat_arr = arr.ravel()
flat_arr = np.asarray(flat_arr).reshape( arr.shape[0]*arr.shape[1] ,3)

vector = np.zeros(shape=(int( flat_arr.shape[0]),5),dtype = int )
j= 0
for i in flat_arr:
    RGBPixel_with_position = np.zeros(shape = (5),dtype = int)
    RGBPixel_with_position[0] = i[0]
    RGBPixel_with_position[1] = i[1]
    RGBPixel_with_position[2] = i[2]
    RGBPixel_with_position[3] = j/rows
    RGBPixel_with_position[4] = j % rows
    vector[j] = RGBPixel_with_position

    j = j+1
# convert it to a matrix
# vector = np.matrix(flat_arr)
# print("2D shape : ", vector.shape)

# vector.resize(vector.shape[0],5)
# print(vector[1])
# print(vector.shape)
# j = 0
# for i in vector:
#     print(i)
#     RGB_vector = i
#     vector[j].resize(5)
#     vector[j][0] = RGB_vector[0]
#     vector[j][1] = RGB_vector[1]
#     vector[j][2] = RGB_vector[2]
#     vector[j][3] = j/rows
#     vector[j][4] = j % rows
#     j = j + 1     

# do something to the vector
# vector[:,::10] = 128

# reform a numpy array of the original shape
# arr2 = np.asarray(vector).reshape(shape)

# make a PIL image
# img2 = Image.fromarray(arr2, 'RGB')
# img2.show() 

dimension = 5  # 5 for R,G,B,X,Y
K = 13         # take general value (number of different clusters)
TotalData = vector.shape[0]
Mean_Clusters = np.random.uniform(low=0 , high =255 , size=(int(K), int(dimension)))
R_nk = np.zeros(shape=(int(TotalData)), dtype=int)
# print(R_nk.shape)

def AssignCluster(Xn, Mean_Clusters):
    Cluster_no = 0
    ans = 0
    MaX_distance = 99999999
    for Mean_vector in Mean_Clusters:
        distance =0
        num = 0
        # Euclidean distance  
        distance  = np.linalg.norm(Xn-Mean_vector)
        
        if distance < MaX_distance:
            MaX_distance =  distance
            ans = Cluster_no
        Cluster_no = Cluster_no + 1
    return ans

def CostCompute(R_nk, Data):
    cost = 0
    j =0
    for i in Data:
        cost += np.linalg.norm(i- Mean_Clusters[ int(R_nk[j] ) ])
    
        j = j+1
    return cost
    
def mean_calculate(R_nk, Data, Cluster_no):
    mean = np.zeros(shape=(int(dimension)),dtype =float)

    sum = np.zeros(shape=(int(dimension)),dtype =float)
    j =  0
    Cluster_no_samples = 0
    for i in Data:
        if R_nk[j] == Cluster_no:
            sum = sum + i
            Cluster_no_samples = Cluster_no_samples + 1
        j= j+1
    
    if Cluster_no_samples != 0:
        mean = np.divide(sum,Cluster_no_samples)
    return mean

l = 0  # Iteration Number
Cost = 0 # Cost/Loss 
Precost = 0 # Cost of previous Iteration.
while True:
    j = 0
    Precost = Cost
    print("\nIteration number: ",l)
    R_nk = np.zeros(shape=(int(TotalData)), dtype=float)
    cluster_pic = np.zeros(shape=(int(TotalData), int(3 )), dtype=float)
    
    for i in vector:
        PredictCluster = AssignCluster(i, Mean_Clusters)
        # print(PredictCluster)
        R_nk[j] = PredictCluster
        
        
        # Divide by 255 to get range within [0, 1]. This is required Only for plotting an RGB image.
        RGBVector = np.zeros(shape = (int(3)), dtype = int)
        RGBVector[0] = Mean_Clusters[PredictCluster][0]
        RGBVector[1] = Mean_Clusters[PredictCluster][1]
        RGBVector[2] = Mean_Clusters[PredictCluster][2]

        RGBVector = RGBVector/255
        cluster_pic[j] = RGBVector
        # cluster_pic[j] = Mean_Clusters[PredictCluster]/255
        # cluster_pic[j] =colors.to_rgb(color[PredictCluster])
        j = j + 1
    print(cluster_pic.shape)
    print(cluster_pic)
    cluster_pic = cluster_pic.reshape(arr.shape[0], arr.shape[1],3)
    plt.imshow(cluster_pic)
    
        # plot_name = 'plt' + str(l) + '.png'
        # plt.title("Iteration number " + str(l))
        # plt.xlabel("Dim 1")
        # plt.ylabel("Dim 2")
        # plt.show()
        # plt.savefig('./plot/Linear/'+ plot_name)

    # Update the mean value(Centroid of Cluster)    
    for i in range(0,K):
        Mean_Clusters[i] = mean_calculate(R_nk, vector, i)

    name = "Iteration" + str(l)
    plt.imsave(name + '.png', cluster_pic)
    Cost = CostCompute(R_nk, vector)
    print("precost: ",Precost)
    print("Cost: ", Cost)

    # If there is no change in pixel values, stop the process
    if l==  50 or Cost == Precost:
        break
    l = l+1
    
