import matplotlib.pyplot as plt

f=open("./LS_Group01/Class1.txt")

for line in f:
    x=line.split()
    x[0] = float(x[0])
    x[1] = float(x[1])
    plt.plot(x[0], x[1], "o", color='red')
    print(x[0],x[1])
f.close()



f=open("./LS_Group01/Class2.txt")

for line in f:
    x=line.split()
    x[0] = float(x[0])
    x[1] = float(x[1])
    plt.plot(x[0], x[1], "o", color='green')
    print(x[0],x[1])
f.close()


f=open("./LS_Group01/Class3.txt")

for line in f:
    x=line.split()
    x[0] = float(x[0])
    x[1] = float(x[1])
    plt.plot(x[0], x[1], "o", color='blue')
    print(x[0],x[1])
plt.show()
f.close()
