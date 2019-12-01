import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dimension = 2
splitRatio = 0.75

labels = ['Class1', 'Class2']
colors = ['Black', 'Purple', 'orange']
Class1_data = np.loadtxt("LS_Group08/Class1.txt",delimiter= " ", skiprows=0)
Class2_data = np.loadtxt("LS_Group08/Class2.txt",delimiter= " ", skiprows=0)

# 1
Class1_label = np.zeros(shape=(Class1_data.shape[0]), dtype = int)
Class1_label.fill(1)
#-1
Class2_label = np.zeros(shape=(Class2_data.shape[0]), dtype = int)
Class2_label.fill(-1)

Data =  np.concatenate( (Class1_data, Class2_data),axis = 0)

x0 = np.zeros((Data.shape[0],1),dtype = float)
x0.fill(1)
Data = np.hstack((x0,Data))

Total_samples = int(Data.shape[0])
print("Total Samples : ", Total_samples)
Train_samples = int (splitRatio*Total_samples)
print("Training Samples : ", Train_samples)
Test_samples = int((1-splitRatio)*Total_samples)
print("Testing Samples : ", Test_samples)

label = np.concatenate((Class1_label, Class2_label), axis =0)
print("Size of Total label: ",  label.shape[0])
Train_Data, Test_Data, Train_label, Test_label = train_test_split(Data,label,test_size=0.25)
print(Train_Data.shape[0])
# Train_Data, Test_Data = Data[:Train_samples,:], Data[Train_samples:,:]

# Train_label, Test_label = label[:Train_samples], label[Train_samples:]
print("Size of Training Labels  : ",Train_label.shape[0])
print("Size of Testing Labels  : ",Test_label.shape[0])

# Column vector: Transpose([w0, w1, w2])
# Random Initialisation of weights
weights = np.random.rand(dimension+1)

def output(data, weight):
    if np.matmul(np.transpose(weight), data ) >= 0:
        return 1
    else :
        return -1

print("-------------")
print(" Training ")
print("-------------")
Predicted_Class1 = []
Predicted_Class2 = []
misclassified = True
epoch_index = 0
EpochWise_Loss = [[],[]]
while misclassified == True:
    # print(epoch)
    index = 0
    wrong_Classified = 0
    misclassified = False
    Predicted_Class1.clear()
    Predicted_Class2.clear()
    for sample in Train_Data:
        y = output(sample, weights)
        if y != Train_label[index]:
            wrong_Classified += 1
            misclassified = True
            weights = weights +(-y)*(sample)
        
        # print(y)
        if y == 1:
            Predicted_Class1.append(sample)
        else:
            Predicted_Class2.append(sample)        
        index += 1

    Loss = (wrong_Classified / float(Train_samples))*100
    EpochWise_Loss[0].append(epoch_index)
    EpochWise_Loss[1].append(Loss)
    print("Epoch : ",epoch_index," Error :",Loss)
    epoch_index += 1


Predicted_Class1 = np.asarray(Predicted_Class1)
Predicted_Class2 = np.asarray(Predicted_Class2)


print("Size of Predicted Class1 : ",Predicted_Class1.shape[0])
print("Size of Predicted Class2 : ",Predicted_Class2.shape[0])
Predicted_Class1 = Predicted_Class1[:,1:]
Predicted_Class2 = Predicted_Class2[:,1:]
# plt.scatter(Predicted_Class1, Predicted_Class2,"o",color="blue")


# --------  Training Plot --------
plt.plot(Predicted_Class1[:,0], Predicted_Class1[:,1],"o ", color=colors[0], label="class1")
plt.plot(Predicted_Class2[:,0], Predicted_Class2[:,1],"o", color= colors[1], label="class2")

x = np.linspace(-10,15 ,100)
y = -(weights[1]*x)/weights[2] - weights[0]/weights[2]
slope = -((weights[1])/weights[2])
intercept = - (weights[0]/(weights[2]))
plt.plot(x,y,'-r',label='y = %.2fx + %.2f' %(slope, intercept))
plt.legend()
plt.title("Training Samples")
plt.savefig("Result/TwoClassPerceptron_Training.jpg")
plt.show()

# ------- Convergence Plot -------
plt.plot(EpochWise_Loss[0], EpochWise_Loss[1], '-o',color= 'blue',label="Loss in each iteration")
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.title("Convergence Plot")
plt.legend()
plt.savefig("Result/TwoClassPerceptron_Converge.jpg")
plt.show()


# ------ Training Decision Boundary ------
class1_train_data =[]
class2_train_data =[]
final=[]
for i in range(len(Train_Data)):
	x=Train_Data[i]
	arg1=x[0]*weights[0]+x[1]*weights[1]+x[2]*weights[2]
	if arg1>0:
		class1_train_data.append(x)
		final.append(1)
	else:
		class2_train_data.append(x)
		final.append(2)
print(np.shape(class1_train_data))
print(np.shape(class2_train_data))

Decision_boundary1 = [] 
Decision_boundary2 = [] 

for x in np.arange(-30, 30, 0.1):
	for y in np.arange(-30, 30, 0.1):
		arg1=1*weights[0]+x*weights[1]+y*weights[2]

		if arg1> 0:
			Decision_boundary1.append([1, x, y])
		else:
			Decision_boundary2.append([1, x, y])

print(np.shape(Decision_boundary1))
print(np.shape(Decision_boundary2))

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
plt.title("Training Samples : One Vs One Perceptron ALGO ")
plt.legend()
plt.savefig("Result/TwoClass_OneVSAll_Train_DecisionBoundary.jpg")
plt.show()


# ------- Testing ------
print("-------------")
print(" Testing ")
print("-------------")

Predicted_Class1 = []
Predicted_Class2 = []
misclassified = True
epoch = 0

for sample in Test_Data:
    y = output(sample, weights)
    if y >=0 :
        Predicted_Class1.append(sample)
    else:
        Predicted_Class2.append(sample)      
    index += 1

Predicted_Class1 = np.asarray(Predicted_Class1)
Predicted_Class2 = np.asarray(Predicted_Class2)

print("Size of Predicted Class1 : ",Predicted_Class1.shape[0])
print("Size of Predicted Class2 : ",Predicted_Class2.shape[0])
if len(Predicted_Class1)!=0:
    Predicted_Class1 = Predicted_Class1[:,1:]

if len(Predicted_Class2)!=0:
    Predicted_Class2 = Predicted_Class2[:,1:]

# plt.scatter(Predicted_Class1, Predicted_Class2,"o",color="blue")
if len(Predicted_Class1)!=0:
    plt.plot(Predicted_Class1[:,0], Predicted_Class1[:,1],"o ", color=colors[0], label="class1")
if len(Predicted_Class2)!=0 :
    plt.plot(Predicted_Class2[:,0], Predicted_Class2[:,1],"o", color=colors[1], label="class2")

x = np.linspace(-10,15 ,100)
y = -(weights[1]*x)/weights[2] - weights[0]/weights[2]
slope = -((weights[1])/weights[2])
intercept = - (weights[0]/(weights[2]))
plt.plot(x,y,'-r',label='y = %.2fx + %.2f' %(slope, intercept))
plt.legend()
plt.title("Testing Samples")
plt.savefig("Result/TwoClassPerceptron_Testing.jpg")
plt.show()


### Testing Decision Boundary
### Decision Boundary
class1_test_data =[]
class2_test_data =[]
final=[]
for i in range(len(Test_Data)):
	x=Test_Data[i]
	arg1=x[0]*weights[0]+x[1]*weights[1]+x[2]*weights[2]
	if arg1>0:
		class1_test_data.append(x)
		final.append(1)
	else:
		class2_test_data.append(x)
		final.append(2)
print(np.shape(class1_test_data))
print(np.shape(class2_test_data))

Decision_boundary1 = [] 
Decision_boundary2 = [] 
for x in np.arange(-30, 30, 0.1):
	for y in np.arange(-30, 30, 0.1):
		arg1=1*weights[0]+x*weights[1]+y*weights[2]

		if arg1> 0:
			Decision_boundary1.append([1, x, y])
		else:
			Decision_boundary2.append([1, x, y])

print(np.shape(Decision_boundary1))
print(np.shape(Decision_boundary2))

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
for i in Predicted_Class1:
	A.append(i[0])
	B.append(i[1])
plt.plot(A, B, 'o', color=colors[0], markersize=3, label=labels[0]);
A=[]
B=[]
for i in Predicted_Class2:
	A.append(i[0])
	B.append(i[1])
plt.plot(A, B, 'o', color=colors[1], markersize=3, label=labels[1]);
plt.title("Testing Samples and Decision ")
plt.legend()
plt.savefig("Result/TwoClass_Test_DecisionBoundary.jpg")
plt.show()






