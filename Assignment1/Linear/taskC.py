import numpy as np
import matplotlib.pyplot as plt
import math

import random
h = .02
xx, yy = np.meshgrid(np.arange(-25, 25, h),
                     np.arange(-25, 25, h))

splitValue = 0.75
num_classes = 3
# Load the data files
file1 = np.loadtxt("LS_Group02/Class1.txt",delimiter= " ", skiprows=0)
file2 = np.loadtxt("LS_Group02/Class2.txt",delimiter= " ", skiprows=0)
file3 = np.loadtxt("LS_Group02/Class3.txt",delimiter= " ", skiprows=0)

numData = 0
for i in file1:
    numData = numData + 1

# size of train dataset 
Train_size = int(splitValue*numData)

# size of test dataset
Test_size = int((1- splitValue)*numData)

# Train data
Class1_train = np.zeros(shape=(int(Train_size),2), dtype=float)
Class2_train = np.zeros(shape=(int(Train_size),2), dtype=float) 
Class3_train = np.zeros(shape=(int(Train_size),2), dtype=float)

# test data
Class1_test = np.zeros(shape=(int(Test_size),2), dtype=float)
Class2_test = np.zeros(shape=(int(Test_size),2), dtype=float)
Class3_test = np.zeros(shape=(int(Test_size),2), dtype=float)

train_set = np.zeros(shape=(int(Train_size*3),2), dtype=float)
label = np.zeros(shape=(int(Train_size*3),), dtype=int)
k=0

# random.shuffle(file1)
# random.shuffle(file2)
# random.shuffle(file3)

j = 0 
for i in file1:
    if j < Train_size:
        Class1_train[j] = i
        train_set[k] = i
        label[k] = 0
        k = k + 1
        j = j + 1
    elif j >= Train_size:
        Class1_test[j-Train_size] = i
        j = j + 1 

# import training and testing data of class2
j = 0 
for i in file2:
    if j < Train_size:
        Class2_train[j] = i
        train_set[k] = i
        label[k] = 1
        k = k + 1
        j = j + 1
    elif j >= Train_size:
        Class2_test[j-Train_size] = i
        j = j + 1 
    
# import training and testing data of class3
print("Size of training data for each Class:  ", Train_size)
j = 0
for i in file3:
    if j < Train_size:
        Class3_train[j] = i
        train_set[k] = i
        label[k] = 2
        k = k + 1
        j = j + 1
    elif j >= Train_size:
        Class3_test[j-Train_size] = i
        j = j + 1

# print(Class1_test)
# print(Class2_test)
# print(Class3_test)
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

u = np.zeros(shape=(3,2), dtype=float)
var = np.zeros(shape=(3,2), dtype=float)
no_of_classes = 3
input_dimension = 2
sample_count = np.zeros(3, dtype=int)

sample_count[0] = numData
sample_count[1] = numData
sample_count[2] = numData


u[0][0],u[0][1] = mean(Class1_train)
# print(u[0][0], u[0][1])
var[0][0], var[0][1] = varience(Class1_train, u[0][0], u[0][1])

u[1][0], u[1][1] = mean(Class2_train)
# print(u[1][0], u[1][1])
var[1][0], var[1][1] = varience(Class2_train, u[1][0], u[1][1])

# mean and variance for Class 3
u[2][0],u[2][1] = mean(Class3_train)
# print(u[2][0], u[2][1])
var[2][0], var[2][1] = varience(Class3_train, u[2][0], u[2][1])

total_sample = 0
for i in range(3):
    total_sample += sample_count[i]
print("Total available samples :    ",total_sample)

''' Covar is covarience matrix for the task3 '''
covar = np.zeros(shape=(no_of_classes,2, 2))
for i in range(no_of_classes):
    for j in range(input_dimension):
        covar[i][j][j] = var[i][j]
# print(covar)  
# covar /= no_of_classes
# print(covar[1])

def decision_boundary(m1, m2, class1_indx, class2_indx):
    sigma_inverse = np.linalg.inv(covar)
    xo = (m1+m2)/2
    mean_diff = m1-m2
    denominator = np.matmul(mean_diff.transpose(), sigma_inverse)
    denominator = np.matmul(denominator, mean_diff)
    xo -= (mean_diff/denominator)*np.log(sample_count[class1_indx]/sample_count[class2_indx])
    return xo

def likelihood(mean_vector, class_indx, sample, ln_covar_mod, ln_pci):
    # //sum 
    sigma_inverse = np.linalg.inv(covar[class_indx])    
    wit = 0
    wit += np.matmul(sample.transpose(), np.matmul(sigma_inverse, sample))
    wit -= 2*np.matmul(mean_vector.transpose(), np.matmul(sigma_inverse, sample))
    wit += np.matmul(mean_vector.transpose(), np.matmul(sigma_inverse, mean_vector))
    wit = -0.5*wit
    wio = 0
    wio -= 0.5*ln_covar_mod
    wio += ln_pci
    return wit+wio


Ln_covar_mod=np.zeros(shape=(3,))
Ln_pci=np.zeros(shape=(3,))
for i in range(3):
    Ln_covar_mod[i] = np.log(np.linalg.det(covar[i]))
    Ln_pci[i] = np.log(1/3)

maximum_prob = 0
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
    maximum_prob = -999999999
    for i in range(3):
        num = likelihood(u[i], i, line, math.log(np.linalg.det(covar[i])), Ln_pci[i])
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
    maximum_prob = -999999999
    for i in range(3):
        num = likelihood(u[i], i, line, np.linalg.det(covar[i]), Ln_pci[i])
        # num = w_t[0]*float(line[0]) + w_t[1]*float(line[1])
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
    maximum_prob = -999999999
    for i in range(3):
        
        num = likelihood(u[i], i, line, np.linalg.det(covar[i]), Ln_pci[i])
        if maximum_prob < num:
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
accuracy = float( float( Total_CorrectSamples)/ float(num_classes*Test_size))
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

print("Precision for Class1 : ", Precision_class1)
print("Precision for Class2 : ", Precision_class2)
print("Precision for Class3 : ", Precision_class3)
print("Mean Precision : ", Precision_mean)

# Recall
try:
    Recall_class1 = float(correctSamples_Class1)/ float(Test_size)
except :
    Recall_class1 = -1

try:
    Recall_class2 = float(correctSamples_Class2)/ float(Test_size)
except:
    Recall_class2 = -1
try:    
    Recall_class3 = float(correctSamples_Class3)/ float(Test_size)
except:
    Recall_class3 = -1
Recall_mean = float(float((Recall_class1 + Recall_class2 + Recall_class3))/3)

print("Recall_class1 ", Recall_class1)
print("Recall_class2 ", Recall_class2)
print("Recall_class3 ",Recall_class3)
print("Mean recall : ",Recall_mean)

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


def class_contour():
    Z = np.zeros(shape=xx.shape, dtype=int)
    for i in range(Z.shape[0]):
        print(i)
        for j in range(Z.shape[1]):
            maximum_prob = 0
            predicted_cls = 0
            for k in range(3):
                sample = np.zeros(shape=(2,))
                sample[0] = xx[i][j]
                sample[1] = yy[i][j]
                num = likelihood(u[k], k, sample, Ln_covar_mod[k], Ln_pci[k])
                if maximum_prob > num :
                    maximum_prob = num
                    predicted_cls = k
            Z[i][j] = predicted_cls
    return Z

Z=class_contour()
Z = Z.reshape(xx.shape)
print(Z)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(train_set[:,0], train_set[:, 1], c=label)
plt.show()

