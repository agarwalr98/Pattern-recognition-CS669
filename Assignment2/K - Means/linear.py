import numpy as np
import matplotlib.pyplot as plt
import random

print("========== K MEANS CLUSTERING ===========\n")
colors=['red', 'blue', 'green']

file1 = np.loadtxt("Data/Linear/LS_Group01/Class1.txt",delimiter= " ", skiprows=0)
file2 = np.loadtxt("Data/Linear/LS_Group01/Class2.txt",delimiter= " ", skiprows=0)
file3 = np.loadtxt("Data/Linear/LS_Group01/Class3.txt",delimiter= " ", skiprows=0)

TotalData = 0
for i in file1:
    TotalData = TotalData + 1
for i in file2:
    TotalData = TotalData + 1
for i in file3:
    TotalData = TotalData + 1

dimension = 2
K = 3

index = 0
Data = np.zeros(shape = (int(TotalData), int(2)), dtype = float)  
for i in file1:
    Data[index] = i
    index = index + 1

for i in file2:
    Data[index] = i
    index = index + 1

for i in file3:
    Data[index] = i
    index = index + 1

minimum = 99999999
maximum = -99999999
for i in Data:
    if i[0]>maximum:
        maximum = i[0]
    if i[0]<minimum:
        minimum = i[0] 

random.shuffle(Data)
# //mean vector with some random values
Mean_Clusters = np.random.uniform(low=minimum , high = maximum, size=(int(K), int(dimension)))
# Mean_Clusters[0]= -2.2365, 2.3224
# Mean_Clusters[1]= 2.4445, 4.5807
# Mean_Clusters[2]= 1.872,  1.1315

R_nk = np.zeros(shape=(int(TotalData), int(K)), dtype=float)

def AssignCluster(Xn, Mean_Clusters):
    Cluster_no = 0
    ans = 0
    MaX_distance = 99999999
    for i in Mean_Clusters:
        distance = (Xn[0]-i[0])*(Xn[0]- i[0]) + (Xn[1]-i[1])*(Xn[1]-i[1])
        if distance < MaX_distance:
            MaX_distance =  distance
            ans = Cluster_no
        Cluster_no = Cluster_no + 1
    return ans

def mean_calculate(R_nk, Data, Cluster_no):
    mean = np.zeros(shape=(int(dimension)),dtype =float)
    # mean[0]= 0
    # mean[1] =10

    sum = np.zeros(shape=(int(dimension) ),dtype =float)
    j =  0
    Cluster_no_samples = 0
    for i in Data:
        if R_nk[j][Cluster_no] == 1:
            sum = sum + i
            Cluster_no_samples = Cluster_no_samples + 1
        j= j+1
    
    if Cluster_no_samples != 0:
        mean = np.divide(sum,Cluster_no_samples)
    return mean

def CostCompute(R_nk, Data):
    cost = 0
    j =0
    for i in Data:
        if R_nk[j][0]==1:
            cost += (i[0]-Mean_Clusters[0][0])*(i[0]-Mean_Clusters[0][0]) + (i[1]-Mean_Clusters[0][1])*(i[1]-Mean_Clusters[0][1])
        elif R_nk[j][1]==1:
            cost += (i[0]-Mean_Clusters[1][0])*(i[0]-Mean_Clusters[1][0]) + (i[1]-Mean_Clusters[1][1])*(i[1]-Mean_Clusters[1][1])
        else :
            cost += (i[0]-Mean_Clusters[2][0])*(i[0]-Mean_Clusters[2][0]) + (i[1]-Mean_Clusters[2][1])*(i[1]-Mean_Clusters[2][1])
        j = j+1
    return cost

cost = 0
precost = -10
num =0
j = 0
l =0
while True:
    j = 0
    precost = cost
    print("\nIteration number: ",l)
    R_nk = np.zeros(shape=(int(TotalData), int(K)), dtype=float)
    for i in Data:
        PredictCluster = AssignCluster(i, Mean_Clusters)
        R_nk[j][PredictCluster] = 1
        j = j + 1
        plt.plot(float(i[0]), float(i[1]),"o", color=colors[PredictCluster])

    
	
    plot_name = 'Iteration' + str(l) + '.png'
    plt.title("Iteration number " + str(l))
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.savefig('./plot/Linear/'+ plot_name)
    for i in range(0,K):
        Mean_Clusters[i] = mean_calculate(R_nk, Data, i)
   


    cost = CostCompute(R_nk, Data)
    print("precost: ", precost)
    print("Cost: ", cost)
    if precost == cost:
        break
    l = l+1

