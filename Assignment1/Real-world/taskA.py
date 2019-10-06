import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
# from sklearn import svm
import random

print("Wait Please!!")
num_classes = 3
# split ratio
splitValue = 0.75
# file2 = np.loadtxt("rd_group2/class2.txt",delimiter= " ", skiprows=0)
# file3 = np.loadtxt("rd_group2/class3.txt",delimiter= " ", skiprows=0)

numData_Class1 = 0
numData_Class2 = 0
numData_Class3 = 0
# Total data of each claSS
with open("rd_group2/class1.txt") as file:
    for i in file:
        numData_Class1 = numData_Class1 + 1

with open("rd_group2/class2.txt") as file:
    for i in file:
        numData_Class2 = numData_Class2 + 1


with open("rd_group2/class3.txt") as file:
    for i in file:
        numData_Class3 = numData_Class3 + 1
h = .02
xx, yy = np.meshgrid(np.arange(-25, 25, h),
                     np.arange(-25, 25, h))
# print(xx)
# for i in head:
#     numData = numData + 1

# size of train dataset 
Train_size_Class1 = int(splitValue*numData_Class1)
Train_size_Class2 = int(splitValue*numData_Class2)
Train_size_Class3 = int(splitValue*numData_Class3)


# size of test dataset
Test_size_Class1 = int((1- splitValue)*numData_Class1)
Test_size_Class2 = int((1- splitValue)*numData_Class2)

Test_size_Class3 = int((1- splitValue)*numData_Class3)


# Train data
Class1_train = np.zeros(shape=(int(Train_size_Class1),2), dtype=float)
Class2_train = np.zeros(shape=(int(Train_size_Class2),2), dtype=float)
Class3_train = np.zeros(shape=(int(Train_size_Class3),2), dtype=float)

# test data
Class1_test = np.zeros(shape=(int(Test_size_Class1)+1,2), dtype=float)
Class2_test = np.zeros(shape=(int(Test_size_Class2)+1,2), dtype=float) 
Class3_test = np.zeros(shape=(int(Test_size_Class3) + 1,2), dtype=float)

train_set = np.zeros(shape=(int(Train_size_Class1 + Train_size_Class2 + Train_size_Class3),2), dtype=float)
label = np.zeros(shape=(int(Train_size_Class1 + Train_size_Class2 + Train_size_Class3),), dtype=int)
k=0

# import training and testing data of class1
j = 0 
with open("rd_group2/class1.txt") as file1:
    for i in file1:
        i = i.split()
        if j < Train_size_Class1:
            Class1_train[j] = i
            train_set[k] = i
            label[k] = 0
            k = k + 1
            j = j + 1
        elif j >= Train_size_Class1:
            Class1_test[j-Train_size_Class1] = i
            j = j + 1 




# import training and testing data of class2
j = 0 
with open("rd_group2/class2.txt") as file2:
    for i in file2:
        i = i.split()
        if j < Train_size_Class2:
            Class2_train[j] = i
            train_set[k] = i
            label[k] = 0
            k = k + 1
            j = j + 1
        elif j >= Train_size_Class2:
            Class2_test[j-Train_size_Class2] = i
            j = j + 1 
    
# import training and testing data of class3
# print(Train_size)
j = 0 
with open("rd_group2/class3.txt") as file3:
    for i in file3:
        i = i.split()
        if j < Train_size_Class3:
            Class3_train[j] = i
            train_set[k] = i
            label[k] = 0
            k = k + 1
            j = j + 1
        elif j >= Train_size_Class3:
            Class3_test[j-Train_size_Class3] = i
            j = j + 1 


def mean(Class_train):
    sumx = 0;sumy = 0;count = 0
    for line in Class_train:
        sumx = sumx + float(line[0])
        sumy = sumy + float(line[1])
        count = count + 1
    sumx /= count
    sumy /= count
    return sumx, sumy

def varience(Class_train, m1, m2):
    varx=0;vary=0;count=0
    for line in Class_train:
        varx += pow(float(line[0])-m1, 2)
        vary += pow(float(line[1])-m2, 2)
        count = count + 1
    varx /= count
    vary /= count
    return varx, vary

def posterior_parameter(mean_vector, class_indx):
    wi = mean_vector/sigma_square
    wio = 0
    for i in range(2):
        wio += mean_vector[i]**2
    wio /= 2*sigma_square
    wio = -wio
    wio += np.log(sample_count[class_indx]/total_sample)
    return wi, wio

def decision_boundary(m1, m2, class1_indx, class2_indx):
    xo = (m1+m2)/2
    mean_diff_mod = 0
    for i in range(2):
        mean_diff_mod += pow(m1[i]-m2[i],2)
    xo -= (sigma_square*(m1-m2)/mean_diff_mod)*np.log(sample_count[class1_indx]/sample_count[class2_indx])
    return xo


# def avg_var(varience_matrix):
#     print(varience_matrix.shape)
#     return np.sum(varience_matrix, axis=0)/varience_matrix.shape[0]

u = np.zeros(shape=(3,2), dtype=float)
var = np.zeros(shape=(3,2), dtype=float)
no_of_classes = 3
sample_count = np.zeros(3, dtype=int)

sample_count[0] = numData_Class1
sample_count[1] = numData_Class2
sample_count[2] = numData_Class3

# Find mean and variance for Class 1
u[0][0],u[0][1] = mean(Class1_train)
# print(u[0][0], u[0][1])
var[0][0], var[0][1] = varience(Class1_train, u[0][0], u[0][1])

                # f=open("./LS_Group01/Class1.txt")
                # sample_count[0]=0
                # for line in f:
                #     x=line.split()
                #     x[0] = float(x[0])
                #     x[1] = float(x[1])
                #     sample_count[0] += 1
                #     plt.plot(x[0], x[1], "o", color='red')
                #     # print(x[0],x[1])
                # f.seek(0)
                # u[0][0],u[0][1] = mean(f)
                # print(u[0][0], u[0][1])
                # f.seek(0)
                # var[0][0], var[0][1] = varience(f, u[0][0], u[0][1])
                # f.close()



                # f=open("./LS_Group01/Class2.txt")
                # sample_count[1]=0
                # for line in f:
                #     x=line.split()
                #     x[0] = float(x[0])
                #     x[1] = float(x[1])
                #     sample_count[1] += 1
                #     plt.plot(x[0], x[1], "o", color='green')
                #     # print(x[0],x[1])
                # f.seek(0)
                # u[1][0], u[1][1] = mean(f)
                # print(u[1][0], u[1][1])
                # f.seek(0)
                # var[1][0], var[1][1] = varience(f, u[1][0], u[1][1])
                # f.close()

# mean and variance for Class 2
u[1][0], u[1][1] = mean(Class2_train)
# print(u[1][0], u[1][1])
var[1][0], var[1][1] = varience(Class2_train, u[1][0], u[1][1])

# mean and variance for Class 3
u[2][0],u[2][1] = mean(Class3_train)
# print(u[2][0], u[2][1])
var[2][0], var[2][1] = varience(Class3_train, u[2][0], u[2][1])

            # f=open("./LS_Group01/Class3.txt")
            # sample_count[2]=0
            # for line in f:
            #     x=line.split()
            #     x[0] = float(x[0])
            #     x[1] = float(x[1])
            #     sample_count[2] += 1
            #     plt.plot(x[0], x[1], "o", color='blue')
            #     # print(x[0],x[1])
            # f.seek(0)
            # u[2][0],u[2][1] = mean(f)
            # print(u[2][0], u[2][1])
            # f.seek(0)
            # var[2][0], var[2][1] = varience(f, u[2][0], u[2][1])
            # # plt.show()
            # f.close()


total_sample = 0
for i in range(3):
    total_sample += sample_count[i]

''' Covar is covarience matrix for the task1 '''
covar = np.zeros(shape=(2, 2))
covar[0][0] = var[0][0] + var[1][0] + var[2][0] + var[0][1] + var[1][1] + var[2][1]
covar[1][1] = covar[0][0]
covar /= 6
# print(covar)

sigma_square = covar[0][0]
 
def class_contour():
    Z = np.zeros(shape=xx.shape, dtype=int)
    wi = np.zeros(shape=(3,2))
    wio = np.zeros(shape=(3,))
    for i in range(3):
        wi[i], wio[i] = posterior_parameter(u[i], i)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            maximum_prob = -999999999
            predicted_cls = 0
            for k in range(3):
                num = wi[k][0]*float(xx[i][j]) + wi[k][1]*float(yy[i][j])
                num += wio[k]
                if maximum_prob < num :
                    maximum_prob = num
                    predicted_cls = k
            Z[i][j] = predicted_cls
    return Z
# C = 1.0  # SVM regularization parameter
# print(xx)
# svc = svm.SVC(kernel='linear', C=C).fit(train_set, label)

# Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
# print(Z)
# print(Z.shape)
Z=class_contour()
Z = Z.reshape(xx.shape)
# print(Z)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(train_set[:,0], train_set[:, 1], c=label)

maximum_prob =-9999999999
predicted_cls = 0
accuracy = 0
PredictedSamples_Class1 = 0
PredictedSamples_Class2 = 0
PredictedSamples_Class3 = 0

correctSamples_Class1  = 0
correctSamples_Class2 = 0
correctSamples_Class3 = 0

C11 = 0
C12 = 0
C13 = 0
C21 = 0
C22 = 0
C23 = 0
C31 = 0
C32 = 0
C33 = 0
for line in Class1_test:
    predicted_cls = 0
    maximum_prob = -9999999999
    for i in range(3):
        w_t, wio = posterior_parameter(u[i], i)
        num = w_t[0]*float(line[0]) + w_t[1]*float(line[1])
        num =num + wio
        if maximum_prob < num :
            maximum_prob = num
            predicted_cls = i
    if predicted_cls == 0:
        correctSamples_Class1 = correctSamples_Class1 + 1
        C11 = C11 +1

    if predicted_cls == 0:
        PredictedSamples_Class1 = PredictedSamples_Class1 + 1
    elif predicted_cls == 1: 
        PredictedSamples_Class2 = PredictedSamples_Class2 + 1
        C12 = C12 + 1
    elif predicted_cls == 2:
        PredictedSamples_Class3 = PredictedSamples_Class3 + 1
        C13 = C13 + 1

for line in Class2_test:
    predicted_cls = 0
    maximum_prob = -9999999999
    for i in range(3):
        w_t, wio = posterior_parameter(u[i], i)
        num = w_t[0]*float(line[0]) + w_t[1]*float(line[1])
        num =num + wio
        if maximum_prob < num :
            maximum_prob = num
            predicted_cls = i
    if predicted_cls == 1:
        correctSamples_Class2 = correctSamples_Class2 + 1
        C22 = C22 + 1
    if predicted_cls == 0:
        PredictedSamples_Class1 = PredictedSamples_Class1 + 1
        C21 = C21 + 1
    elif predicted_cls == 1: 
        PredictedSamples_Class2 = PredictedSamples_Class2 + 1
    elif predicted_cls == 2:
        PredictedSamples_Class3 = PredictedSamples_Class3 + 1
        C23 = C23 + 1


for line in Class3_test:
    predicted_cls = 0
    maximum_prob = -9999999999
    for i in range(3):
        w_t, wio = posterior_parameter(u[i], i)
        num = w_t[0]*float(line[0]) + w_t[1]*float(line[1])
        num =num + wio
        if maximum_prob < num :
            maximum_prob = num
            predicted_cls = i
    if predicted_cls == 2:
        correctSamples_Class3 = correctSamples_Class3 + 1
        C33 = C33 + 1
    if predicted_cls == 0:
        C31 = C31 +1
        PredictedSamples_Class1 = PredictedSamples_Class1 + 1
    elif predicted_cls == 1: 
        PredictedSamples_Class2 = PredictedSamples_Class2 + 1
        C32 = C32 + 1
    elif predicted_cls == 2:
        PredictedSamples_Class3 = PredictedSamples_Class3 + 1

# print(correctSamples_Class1)
# print(correctSamples_Class2)
# print(correctSamples_Class3)

# print(PredictedSamples_Class1)
# print(PredictedSamples_Class2)
# print(PredictedSamples_Class3)


# Accuracy
Total_CorrectSamples = correctSamples_Class1 + correctSamples_Class2 + correctSamples_Class3
accuracy = float( float( Total_CorrectSamples)/ float(Test_size_Class1 + Test_size_Class2 + Test_size_Class3))
accuracy  =accuracy*100
print("Accuracy : ",accuracy)

# Precision
try:
    Precision_class1 = float(correctSamples_Class1)/float(PredictedSamples_Class1)
except:
    Precision_class1 = -1

try:
    Precision_class2 = float(correctSamples_Class2)/float(PredictedSamples_Class2)
except:
    Precision_class2 = -1

try:
    Precision_class3 = float(correctSamples_Class3)/float(PredictedSamples_Class3)
except:
    Precision_class3 = -1
Precision_mean = float(float((Precision_class1 + Precision_class2 + Precision_class3))/3)

# Recall
try:
    Recall_class1 = float(correctSamples_Class1)/ float(Test_size_Class1)
except :
    Recall_class1 = -1

try:
    Recall_class2 = float(correctSamples_Class2)/ float(Test_size_Class2)
except:
    Recall_class2 = -1
try:    
    Recall_class3 = float(correctSamples_Class3)/ float(Test_size_Class3)
except:
    Recall_class3 = -1
Recall_mean = float(float((Recall_class1 + Recall_class2 + Recall_class3))/3)

print("Precision for Class1 : ", Precision_class1)
print("Precision for Class2 : ", Precision_class2)
print("Precision for Class3 : ", Precision_class3)
print("Mean Precision : ", Precision_mean)

# F score
F_Class1 = 2*(float(Precision_class1))*(float(Recall_class1))
F_Class1 = float(F_Class1) / (Precision_class1 + Recall_class1)
F_Class2 = 2*(float(Precision_class2))*(float(Recall_class2))
F_Class2 = float(F_Class2) / (Precision_class2 + Recall_class2)
F_Class3 = 2*(float(Precision_class3))*(float(Recall_class3))
F_Class3 = float(F_Class3) / (Precision_class3 + Recall_class3)
F_mean  = float(F_Class1 + F_Class2 + F_Class3)/3

print("F Score for Class1 :",F_Class1)
print("F Score for Class2 :",F_Class2)
print("F Score for Class3 :",F_Class3) 
print("Mean F Value : ",F_mean)

Confusion_matrix = np.zeros(shape=(num_classes, num_classes), dtype = int)
Confusion_matrix[0][0] = C11
Confusion_matrix[0][1] = C21
Confusion_matrix[0][2] = C31
Confusion_matrix[1][0] = C12
Confusion_matrix[1][1] = C22
Confusion_matrix[1][2] = C32
Confusion_matrix[2][0] = C13
Confusion_matrix[2][1] = C23
Confusion_matrix[2][2] = C33
print("Confusion Matrix \n")
print (Confusion_matrix,"\n")
plt.show()
