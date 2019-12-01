import numpy as np
import random

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dimension = 2
num_classes = 3

colors = ['Black', 'Purple', 'red']
labels = ['Class1', 'Class2', 'Class3']
Class1_data = np.loadtxt("LS_Group08/Class1.txt" ,delimiter= " ", skiprows=0)
print("Size of True Class1 : ",Class1_data.shape[0])
Class2_data = np.loadtxt("LS_Group08/Class2.txt" ,delimiter= " ", skiprows=0)
print("Size of True Class2 : ",Class2_data.shape[0])
Class3_data = np.loadtxt("LS_Group08/Class3.txt" ,delimiter= " ", skiprows=0)
print("Size of True Class3 : ",Class3_data.shape[0])

plt.plot(Class1_data[:,0],Class1_data[:,1],"o", label="Class1", color = colors[0])
plt.plot(Class2_data[:,0],Class2_data[:,1],"o", label="Class2", color = colors[1] )
plt.plot(Class3_data[:,0],Class3_data[:,1],"o", label="Class3", color = colors[2])
plt.title("Input Data")
plt.legend()
plt.savefig("Result/Input_Data.jpg")
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
x0 = np.zeros((Data.shape[0],1),dtype = float)
x0.fill(1)
Data = np.hstack((x0, Data))
print("Size of Data: ", Data.shape[0])

Total_samples = Data.shape[0]
label = np.concatenate((Class1_label, Class2_label), axis =0)
label = np.concatenate((label, Class3_label), axis =0)
print("Size of label: ",  label.shape[0])
# Train_Data, Test_Data, Train_label, Test_label = train_test_split(Data, label,test_size=0.25)

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
weights = np.random.rand((num_classes), dimension+1)
for i in range(0, len(weights)):
    weights[i] = np.random.rand(dimension+1)
print("Size of weights : ",weights.shape)

def output(data, weight):
    if np.matmul(np.transpose(weight), data ) >= 0:
        return 1
    else :
        return -1

epoch = 0
misclassified = True
Predicted_Class = [[[],[]],[[],[]], [[],[]]]
print(Predicted_Class[0])
Epoch_wise_Loss = [[[],[]],[[],[]], [[],[]]]
wrong_Classified = 1000000
for cls in range(0,num_classes):
    wrong_Classified = 1000000
    loss = 1
    print("For Class   ",  cls+1)
    epoch = 0
    while loss > 0.00   :
        misclassified = False
        wrong_Classified = 0
        index = 0                       
        
        for sample in Train_Data:
            y = output(sample, weights[cls])
            
            if (y ==-1 and Train_label[index]==cls) :
                
                wrong_Classified += 1
                misclassified = True
                weights[cls] = weights[cls] +(-y)*(sample)
            elif y==1 and Train_label[index]!=cls:
            
                wrong_Classified += 1
                misclassified = True
                weights[cls] = weights[cls] +(-y)*(sample)
            index += 1
                    
        Error = (wrong_Classified / float(Total_samples))*100
        loss = Error
        Epoch_wise_Loss[cls][0].append(epoch)
        Epoch_wise_Loss[cls][1].append(loss)
        epoch += 1
    print("Epoch : ", epoch, " Loss : ", loss)

# Predicted_Class = np.asarray(Predicted_Class)
print(Train_Data_Class1.shape)
plt.plot(Train_Data_Class1[:,0], Train_Data_Class1[:,1] ,"o", color= colors[0], label=labels[0])
x = np.linspace(0, 15)
y = -(weights[0][1]*x)/weights[0][2] - (weights[0][0])/weights[0][2]

# # w0 + w1*x + w2*y  = 0
slope = -((weights[0][1])/weights[0][2])
intercept = - (weights[0][0]/(weights[0][2]))
plt.plot(x,y,'-r', label='y = %0.2fx + %0.2f'%(slope, intercept), color=colors[0])

plt.plot(Train_Data_Class2[:,0], Train_Data_Class2[:,1] ,"o", color= colors[1], label=labels[1])
x = np.linspace(-5,26 )
y = -(weights[1][1]*x)/weights[1][2] - (weights[1][0])/weights[1][2]
slope = -((weights[1][1])/weights[1][2])
intercept = - (weights[1][0]/(weights[1][2]))
plt.plot(x,y,'-r', label='y = %0.2fx + %0.2f'%(slope, intercept), color=colors[1])

plt.plot(Train_Data_Class3[:,0], Train_Data_Class3[:,1] ,"o", color= colors[2], label=labels[2])
x = np.linspace(-3, 15 )
y = -(weights[2][1]*x)/weights[2][2] - (weights[2][0])/weights[2][2]
slope = -((weights[2][1])/weights[2][2])
intercept = -(weights[2][0]/(weights[2][2]))
plt.plot(x,y,'-r', label='y = %0.2fx + %0.2f'%(slope, intercept), color=colors[2])

plt.title("Training Samples : One Vs All Perceptron Algo")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.legend()
plt.savefig("Result/MultiClassPerceptron_OneVsAll_"+ "Training.jpg")
plt.show()


### Training Decision Boundaries 
### Decision Boundaries
w1 = weights[0]
w2 = weights[1]
w3 = weights[2]
class1_train_data =[]
class2_train_data =[]
class3_train_data =[]
final=[]
for i in range(len(Train_Data)):
	x=Train_Data[i]
	arg1=x[0]*w1[0]+x[1]*w1[1]+x[2]*w1[2]
	arg2=x[0]*w2[0]+x[1]*w2[1]+x[2]*w2[2]
	arg3=x[0]*w3[0]+x[1]*w3[1]+x[2]*w3[2]
	if arg1>arg2 and arg1>arg3:
		class1_train_data.append(x)
		final.append(1)
	elif arg3>arg1 and arg3>arg2:
		class3_train_data.append(x)
		final.append(3)
	else:
		class2_train_data.append(x)
		final.append(2)
print(np.shape(class1_train_data))
print(np.shape(class2_train_data))
print(np.shape(class3_train_data))



Decision_boundary1 = [] 
Decision_boundary2 = [] 
Decision_boundary3 = [] 

for x in np.arange(-32, 32, 0.1):
	for y in np.arange(-32, 32, 0.1):
		arg1=1*w1[0]+x*w1[1]+y*w1[2]
		arg2=1*w2[0]+x*w2[1]+y*w2[2]
		arg3=1*w3[0]+x*w3[1]+y*w3[2]
		if arg1>arg2 and arg1>arg3:
			Decision_boundary1.append([1, x, y])
		elif arg3>arg1 and arg3>arg2:
			Decision_boundary3.append([1, x, y])
		else:
			Decision_boundary2.append([1, x, y])

print(np.shape(Decision_boundary1))
print(np.shape(Decision_boundary2))
print(np.shape(Decision_boundary3))
A=[]
B=[]
for i in Decision_boundary1:
	A.append(i[1])
	B.append(i[2])
plt.scatter(A, B, c='Green', alpha=0.5 )
A=[]
B=[]
for i in Decision_boundary2:
	A.append(i[1])
	B.append(i[2])
plt.scatter(A, B, c='Orange', alpha=0.5)
A=[]
B=[]
for i in Decision_boundary3:
	A.append(i[1])
	B.append(i[2])
plt.scatter(A, B, c='Pink', alpha=0.5)

A=[]
B=[]
for i in class1_train_data:
	A.append(i[1])
	B.append(i[2])
plt.plot(A, B, 'o', color=colors[0], markersize=3, label=labels[0]);
A=[]
B=[]
for i in class2_train_data:
	A.append(i[1])
	B.append(i[2])
plt.plot(A, B, 'o', color=colors[1], markersize=3, label=labels[1]);
A=[]
B=[]
for i in class3_train_data:
	A.append(i[1])
	B.append(i[2])
plt.plot(A, B, 'o', color=colors[2], markersize=3, label=labels[2]);
# plt.legend(loc='upper left')
plt.title("Training Samples : Decision Boundaries ")
plt.legend()
plt.savefig("Result/MultiClass_OneVsAll_Training_DecisionBoundaries.jpg")
plt.show()




# Convergence Plot - Epoch vs Loss
for i in range(0, num_classes):
    plt.plot(Epoch_wise_Loss[i][0], Epoch_wise_Loss[i][1], "-o" ,color = colors[i], label = labels[i])
    
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("One Vs All Perceptron Algorithm")
plt.legend()
plt.savefig("Result/MultiClass_OneVsAll_convergence.jpg")
plt.show()





# Testing  (One vs All Perceptron ALgorithm)
Predicted_Class = [[[],[]], [[],[]], [[],[]]]
for sample in Test_Data:
    max = -999999999
    max_index = 0
    for weight_index in range(0, len(weights)):
        y = output(sample, weights[weight_index])
        if y > max:
            max = y
            max_index = weight_index
    # if y == 1:
    Predicted_Class[max_index][0].append(sample[1])
    Predicted_Class[max_index][1].append(sample[2])
            
for cls in range(0, num_classes):
    plt.plot(Predicted_Class[cls][0], Predicted_Class[cls][1],"o", color= colors[cls], label=labels[cls])
    x = np.linspace(-5,20)
    y = -(weights[cls][1]*x)/weights[cls][2] - (weights[cls][0])/weights[cls][2]
    slope = -((weights[cls][1])/weights[cls][2])
    intercept = - (weights[cls][0]/(weights[cls][2]))
    plt.plot(x,y,'-r', label='y = %0.2fx + %0.2f' %(slope, intercept), color =colors[cls])

plt.title("Testing Samples")
plt.legend()
plt.savefig("Result/MultiClassPerceptron_OneVsAll_Testing.jpg")
plt.show()


### Testing Decision Boundaries 
### Decision Boundaries
w1 = weights[0]
w2 = weights[1]
w3 = weights[2]
class1_test_data = []
class2_test_data = []
class3_test_data = []
final=[]
for i in range(len(Test_Data)):
	x=Test_Data[i]
	arg1=x[0]*w1[0]+x[1]*w1[1]+x[2]*w1[2]
	arg2=x[0]*w2[0]+x[1]*w2[1]+x[2]*w2[2]
	arg3=x[0]*w3[0]+x[1]*w3[1]+x[2]*w3[2]
	if arg1>arg2 and arg1>arg3:
		class1_test_data.append(x)
		final.append(1)
	elif arg3>arg1 and arg3>arg2:
		class3_test_data.append(x)
		final.append(3)
	else:
		class2_test_data.append(x)
		final.append(2)
print(np.shape(class1_train_data))
print(np.shape(class2_train_data))
print(np.shape(class3_train_data))


Decision_boundary1 = [] 
Decision_boundary2 = [] 
Decision_boundary3 = [] 

for x in np.arange(-32, 32, 0.1):
	for y in np.arange(-32, 32, 0.1):
		arg1=1*w1[0]+x*w1[1]+y*w1[2]
		arg2=1*w2[0]+x*w2[1]+y*w2[2]
		arg3=1*w3[0]+x*w3[1]+y*w3[2]
		if arg1>arg2 and arg1>arg3:
			Decision_boundary1.append([1, x, y])
		elif arg3>arg1 and arg3>arg2:
			Decision_boundary3.append([1, x, y])
		else:
			Decision_boundary2.append([1, x, y])

print(np.shape(Decision_boundary1))
print(np.shape(Decision_boundary2))
print(np.shape(Decision_boundary3))
A=[]
B=[]
for i in Decision_boundary1:
	A.append(i[1])
	B.append(i[2])
plt.scatter(A, B, c='Green', alpha=0.5 )
A=[]
B=[]
for i in Decision_boundary2:
	A.append(i[1])
	B.append(i[2])
plt.scatter(A, B, c='Orange', alpha=0.5)
A=[]
B=[]
for i in Decision_boundary3:
	A.append(i[1])
	B.append(i[2])
plt.scatter(A, B, c='Pink', alpha=0.5)

A=[]
B=[]
for i in class1_test_data:
	A.append(i[1])
	B.append(i[2])
plt.plot(A, B, 'o', color=colors[0], markersize=3, label=labels[0]);
A=[]
B=[]
for i in class2_test_data:
	A.append(i[1])
	B.append(i[2])
plt.plot(A, B, 'o', color=colors[1], markersize=3, label=labels[1]);
A=[]
B=[]
for i in class3_test_data:
	A.append(i[1])
	B.append(i[2])
plt.plot(A, B, 'o', color=colors[2], markersize=3, label=labels[2]);

plt.title("Testing Samples : Decision Boundaries ")
plt.legend()
plt.savefig("Result/MultiClass_OneVsAll_Testing_DecisionBoundaries.jpg")
plt.show()
