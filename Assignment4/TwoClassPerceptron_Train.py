import numpy as np
import random
import matplotlib.pyplot as plt
dimension = 2

Class1_data = np.loadtxt("LS_Group08/Class1.txt",delimiter= " ", skiprows=0)
print("Size of True Class1 : ",Class1_data.shape[0] )
Class2_data = np.loadtxt("LS_Group08/Class2.txt",delimiter= " ", skiprows=0)
print("Size of True Class2 : ",Class2_data.shape[0] )

# 1 
Class1_label = np.zeros(shape=(Class1_data.shape[0]), dtype = int)
Class1_label.fill(1)
#-1
Class2_label = np.zeros(shape=(Class2_data.shape[0]), dtype = int)
Class2_label.fill(-1)

Data =  np.concatenate( (Class1_data, Class2_data),axis = 0)
print("Size of Data: ", Data.shape[0])
x0 = np.zeros((Data.shape[0],1),dtype = float)
x0.fill(1)
Data = np.hstack((x0,Data))

Total_samples = Data.shape[0]
label = np.concatenate((Class1_label, Class2_label), axis =0)
print("Size of label: ",  label.shape[0])
# print(label)

# Column vector: Transpose([w0, w1, w2])
weights = np.random.rand(dimension+1)

def output(data, weight):
    if np.matmul(np.transpose(weight), data ) >= 0:
        return 1
    else :
        return -1


Predicted_Class1 = []
Predicted_Class2 = []
# print(Predicted_Class1)
# print(Predicted_Class2)
misclassified = True
epoch = 0
while misclassified == True:
    # print(epoch)
    index = 0
    wrong_Classified = 0
    misclassified = False
    Predicted_Class1.clear()
    Predicted_Class2.clear()
    for sample in Data:
        y = output(sample, weights)
        if y != label[index]:
            wrong_Classified += 1
            misclassified = True
            weights = weights +(-y)*(sample)
        
        # print(y)
        if y == 1:
            Predicted_Class1.append(sample)
        else:
            Predicted_Class2.append(sample)        
        index += 1

    Error = (wrong_Classified / float(Total_samples))*100
    print("Epoch : ",epoch," Error :", Error)
    epoch += 1


Predicted_Class1 = np.asarray(Predicted_Class1)
Predicted_Class2 = np.asarray(Predicted_Class2)


print("Size of Predicted Class1 : ",Predicted_Class1.shape[0])
print("Size of Predicted Class2 : ",Predicted_Class2.shape[0])
Predicted_Class1 = Predicted_Class1[:,1:]
Predicted_Class2 = Predicted_Class2[:,1:]

# plt.scatter(Predicted_Class1, Predicted_Class2,"o",color="blue")
plt.plot(Predicted_Class1[:,0], Predicted_Class1[:,1],"o ", color="green", label="class1")
plt.plot(Predicted_Class2[:,0], Predicted_Class2[:,1],"o", color="blue", label="class2")
plt.legend()
x = np.linspace(-10,15 ,100)
y = -(weights[1]*x)/weights[2] - weights[0]/weights[2]
plt.plot(x,y,'-r' )
plt.show()




