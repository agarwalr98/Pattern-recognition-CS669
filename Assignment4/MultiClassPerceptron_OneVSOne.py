import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dimension = 2
num_classes = 3

colors = ['green', 'orange', 'Blue']
Class1_data = np.loadtxt("LS_Group07/Class1.txt" ,delimiter= " ", skiprows=0)
print("Size of True Class1 : ",Class1_data.shape[0])
Class2_data = np.loadtxt("LS_Group07/Class2.txt" ,delimiter= " ", skiprows=0)
print("Size of True Class2 : ",Class2_data.shape[0])
Class3_data = np.loadtxt("LS_Group07/Class3.txt" ,delimiter= " ", skiprows=0)
print("Size of True Class3 : ",Class3_data.shape[0])

plt.plot(Class1_data[:,0],Class1_data[:,1],"o", label="Class1", color = colors[0])
plt.plot(Class2_data[:,0],Class2_data[:,1],"o", label="Class2", color = colors[1] )
plt.plot(Class3_data[:,0],Class3_data[:,1],"o", label="Class3", color = colors[2])
plt.title("Input Data")
plt.legend()

plt.show()

# 0 
Class1_label = np.zeros(shape=(Class1_data.shape[0]), dtype = int)
Class1_label.fill(0)
# 1
Class2_label = np.zeros(shape=(Class2_data.shape[0]), dtype = int)
Class2_label.fill(1)
#2
Class3_label = np.zeros(shape=(Class3_data.shape[0]), dtype = int)
Class3_label.fill(2)

Data =  np.concatenate( (Class1_data, Class2_data),axis = 0)
Data =  np.concatenate( (Data, Class3_data),axis = 0)
print("Size of Data: ", Data.shape[0])

Total_samples = Data.shape[0]
label = np.concatenate((Class1_label, Class2_label), axis =0)
label = np.concatenate((label, Class3_label), axis =0)
print("Size of label: ",  label.shape[0])

Train_Data_Class1, Test_Data_Class1, Train_label_Class1, Test_label_Class1 = train_test_split(Class1_data,Class1_label,test_size=0.25)
Train_Data_Class2, Test_Data_Class2, Train_label_Class2, Test_label_Class2 = train_test_split(Class2_data,Class2_label,test_size=0.25)
Train_Data_Class3, Test_Data_Class3, Train_label_Class3, Test_label_Class3 = train_test_split(Class3_data,Class3_label,test_size=0.25)

Train_Data = np.concatenate((Train_Data_Class1, Train_Data_Class2), axis = 0)
Train_Data = np.concatenate((Train_Data, Train_Data_Class3), axis = 0)
Train_label = np.concatenate((Train_label_Class1, Train_label_Class2), axis = 0)
Train_label = np.concatenate((Train_label, Train_label_Class3), axis = 0)
x0 = np.zeros((Train_Data.shape[0],1),dtype = float)
x0.fill(1)
Train_Data = np.hstack((x0, Train_Data))
print("Total Training Data : ", Train_Data.shape[0])

Test_Data = np.concatenate((Test_Data_Class1, Test_Data_Class2), axis = 0)
Test_Data = np.concatenate((Test_Data, Test_Data_Class3), axis = 0)
Test_label = np.concatenate((Test_label_Class1, Test_label_Class2), axis = 0)
Test_label = np.concatenate((Test_label, Test_label_Class3), axis = 0)
x0 = np.zeros((Test_Data.shape[0],1),dtype = float)
x0.fill(1)
Test_Data = np.hstack((x0, Test_Data))
print("Total Training Data : ",Test_Data.shape[0])

# Column vector: Transpose([w0, w1, w2])
weights = np.random.rand( int ((num_classes*(num_classes - 1))/2), dimension+1)
for i in range(0, len(weights)):
    weights[i] = np.random.rand(dimension+1)
print("Size of weights : ",weights.shape)

def output(data, weight):
    if np.matmul(np.transpose(weight), data ) >= 0:
        return 1
    else :
        return -1

epoch = 0
convergencePlot_number = 1
Epochwise_Loss1 = [[],[]]
Epochwise_Loss2 = [[],[]]
Epochwise_Loss3 = [[],[]]
misclassified = True
weight_index = 0
for j in range(0,num_classes-1 ):
    misclassified = True
    epoch = 0
    
    for cls in range(j+1,num_classes ):
        wrong_Classified = 0
        misclassified = True
        index = 0
        epoch = 0
        while misclassified == True:
            misclassified = False
            wrong_Classified = 0
            index = 0
            for sample in Train_Data:
                if Train_label[index]== j or Train_label[index] == cls:
                    y = output(sample, weights[weight_index])
                    if (y ==-1 and Train_label[index]==j) :
                        wrong_Classified += 1
                        misclassified = True
                        weights[weight_index] = weights[weight_index] +(-y)*(sample)
                    elif y==1 and Train_label[index]==cls:
                        wrong_Classified += 1 
                        misclassified = True
                        weights[weight_index] = weights[weight_index] +(-y)*(sample)
                        # Predicted_Class[j].append(sample)
                index += 1
                     
            Error = (wrong_Classified / float(Total_samples))*100 
            if convergencePlot_number == 1 :
                Epochwise_Loss1[0].append(epoch)
                Epochwise_Loss1[1].append( Error)
            elif convergencePlot_number == 2:
                Epochwise_Loss2[0].append(epoch)
                Epochwise_Loss2[1].append( Error)
            else:
                Epochwise_Loss3[0].append(epoch)
                Epochwise_Loss3[1].append( Error)
            print("Epoch : ",epoch," Error :", "%.2f" %Error)
            epoch += 1
        convergencePlot_number = convergencePlot_number + 1        
        weight_index += 1
    
print(Epochwise_Loss1)
print(Epochwise_Loss2)
print(Epochwise_Loss3)
    # print(len(Predicted_Class[j]))
    

# print(len(Predicted_Class[1]))
# Predicted_Class = np.asarray(Predicted_Class)
# print((Predicted_Class[0]).shape)
# print("Size of Predicted Class1 : ",Predicted_Class[0].shape[0])
# print("Size of Predicted Class2 : ",Predicted_Class[1].shape[0])

# Predicted_Class1 = Predicted_Class1[:,1:]
# Predicted_Class2 = Predicted_Class2[:,1:]
# Predicted_Class = Predicted_Class[:,:,1:]
# print(Predicted_Class.shape)

# plt.scatter(Predicted_Class1, Predicted_Class2,"o",color="blue")
plt.plot(Class1_data[:,0],Class1_data[:,1],"o", label="Class1",color = colors[0])
plt.plot(Class2_data[:,0],Class2_data[:,1],"o", label="Class2",color = colors[1])
plt.plot(Class3_data[:,0],Class3_data[:,1],"o", label="Class3",color = colors[2])

x = np.linspace(3, 22,100)
y = -(weights[0][1]*x)/weights[0][2] - weights[0][0]/weights[0][2]

# w0 + w1*x + w2*y  = 0
slope = -((weights[0][1])/weights[0][2])
intercept = - (weights[0][0]/(weights[0][2]))
plt.plot(x,y,'-r', label='y = %0.2fx + %0.2f'%(slope, intercept), color=colors[0])

x = np.linspace(-5,8, 100) 
y = -(weights[1][1]*x)/weights[1][2] - weights[1][0]/weights[1][2]
slope = -(weights[1][1])/(weights[1][2])
intercept = -(weights[1][0])/(weights[1][2])
plt.plot(x,y,'-r' ,label='y = %02.fx + %0.2f' % (slope, intercept),color = colors[1])

x = np.linspace(-5,8, 100)
y = -(weights[2][1]*x)/weights[2][2] - weights[2][0]/weights[2][2]
slope = -(weights[2][1])/(weights[2][2])
intercept = -(weights[2][0])/(weights[2][2])
plt.plot(x,y,'-r' ,label='y = %02.fx + %0.2f' % (slope, intercept),color = colors[2])
plt.title("Training Samples")
plt.legend()
plt.savefig("Result/MultiClassPerceptron_Training.jpg")
plt.show()

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Convergence Plot")
plt.plot(Epochwise_Loss1[0], Epochwise_Loss1[1],'-o', label="For Class1 and Class2")
plt.plot(Epochwise_Loss2[0], Epochwise_Loss2[1],'-o', label="For Class1 and Class3")
plt.plot(Epochwise_Loss3[0], Epochwise_Loss3[1],'-o', label="For Class2 and Class3 ")
plt.legend()
plt.savefig("Result/TwoClassPerceptron_"+ "Convergence.jpg")

plt.show()


# ------ testing -------------
# Note that testing is implemented keeping in mind that there are only three classes
Predicted_Class1 = []
Predicted_Class2 = []
Predicted_Class3 = []
            # Predicted_Class = []

# weights[0] - for class 0 and 1
# weights[1] - for class 0 and 2
# weights[2] - for class 1 and 2
misclassified = True
epoch = 0
index = 0

for sample in Test_Data:
    if Test_label[index] == 0 or  Test_label[index] ==1 :
        y = output(sample, weights[0])
        if y >=0 :
            Predicted_Class1.append(sample)
        elif y<0:
            Predicted_Class2.append(sample)
    index = index + 1
    
index = 0
for sample in Test_Data:
    if Test_label[index] == 0 or  Test_label[index] == 2:
        y = output(sample, weights[1])

        if y >=0 :
            Predicted_Class1.append(sample)
        else:
            Predicted_Class3.append(sample)      
    index += 1

index =0
for sample in Test_Data:
    if Test_label[index] == 1 or Test_label[index] == 2:
        y = output(sample, weights[2])

        if y >=0 :
            Predicted_Class2.append(sample)
        else:
            Predicted_Class3.append(sample) 
    index += 1
Predicted_Class1 = np.asarray(Predicted_Class1)
Predicted_Class2 = np.asarray(Predicted_Class2)
Predicted_Class3 = np.asarray(Predicted_Class3)
Predicted_Class1 = Predicted_Class1[:,1:]
Predicted_Class2 = Predicted_Class2[:,1:]
Predicted_Class3 = Predicted_Class3[:,1:]

print("Shape of Predicted points : ,", Predicted_Class1.shape)
plt.plot(Predicted_Class1[:,0],Predicted_Class1[:,1],"o", label="Class1",color = colors[0])
plt.plot(Predicted_Class2[:,0],Predicted_Class2[:,1],"o", label="Class2",color = colors[1])
plt.plot(Predicted_Class3[:,0],Predicted_Class3[:,1],"o", label="Class3",color = colors[2])

x = np.linspace(3, 22,100)
y = -(weights[0][1]*x)/weights[0][2] - weights[0][0]/weights[0][2]

# w0 + w1*x + w2*y  = 0
slope = -((weights[0][1])/weights[0][2])
intercept = - (weights[0][0]/(weights[0][2]))
plt.plot(x,y,'-r', label='y = %0.2fx + %0.2f'%(slope, intercept), color=colors[0])

x = np.linspace(-5,8, 100) 
y = -(weights[1][1]*x)/weights[1][2] - weights[1][0]/weights[1][2]
slope = -(weights[1][1])/(weights[1][2])
intercept = -(weights[1][0])/(weights[1][2])
plt.plot(x,y,'-r' ,label='y = %02.fx + %0.2f' % (slope, intercept),color = colors[1])

x = np.linspace(-5,8, 100)
y = -(weights[2][1]*x)/weights[2][2] - weights[2][0]/weights[2][2]
slope = -(weights[2][1])/(weights[2][2])
intercept = -(weights[2][0])/(weights[2][2])
plt.plot(x,y,'-r' ,label='y = %02.fx + %0.2f' % (slope, intercept),color = colors[2])
plt.title("Testing Samples")
plt.legend()
plt.savefig("Result/MultiClassPerceptron_Testing.jpg")
plt.show()