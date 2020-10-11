#Rena Repenning
#HW2 Problem3
# "Problem 2.23 part (a) and (c) of the textbook. (See p.15 of Lecture 7.)"
import numpy as np
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2)
numTrials = 1000

def randomSampleD():
    Xd = np.random.uniform(low=-1.0, high=1.0, size=2)
    Y = []
    pi = 3.1415926535
    for i in range(len(Xd)):
        Y.append( np.sin(pi*Xd[i]) )
    return Xd[0], Y[0], Xd[1], Y[1]

def aFxn(X, Xt, Yt):
    return ((Yt[0]-Yt[1])/(Xt[0]-Xt[1])) * X + (((Xt[0]*Yt[1])-(Xt[1]*Yt[0]))/(Xt[0]-Xt[1]))

def cFxn(X, Xt, Yt):
    return (Yt[0]+Yt[1])/2 + X*0

X = np.arange(-1., 1., .01)
resA = []
resC = []
for i in range(numTrials):
    D = randomSampleD()
    Xt = [D[0], D[2]]
    Yt = [D[1], D[3]]

    #A
    Ga = aFxn(X, Xt, Yt)
    axs[0].plot(X, Ga, color="silver")
    #C
    Gc = cFxn(X, Xt, Yt)
    axs[1].plot(X, Gc, color="silver")
    #SAVE RESULTS
    resA.append(Ga)
    resC.append(Gc)

# FIND EACH MEAN
meanA = np.mean(resA)
meanC = np.mean(resC)

# G Bar
g_barA = np.mean(resA, axis=0)
g_barC = np.mean(resC, axis=0)

axs[0].plot(X, g_barA, color="black", label="A gBar")
axs[1].plot(X, g_barC, color="black", label="C gBar")

#Find variance
var_A = np.subtract( np.mean(np.power(resA, 2), axis=0), np.power(g_barA, 2))
var_C = np.subtract( np.mean(np.power(resC, 2), axis=0), np.power(g_barC, 2))
axs[0].plot(X, var_A, color="pink", label="var_A")
axs[1].plot(X, var_C, color="blue", label="var_C")

F = np.sin(3.1415926535*X)
#BIAS
bias_A = np.power( np.subtract(g_barA, F), 2 )
bias_C = np.power( np.subtract(g_barC, F), 2 )
axs[0].plot(X, bias_A, color="blue", label="bias A")
axs[1].plot(X, bias_C, color="blue", label="bias C")

#e out
eOut_A = np.add(var_A, bias_A)
eOut_C = np.add(var_C, bias_C)
axs[0].plot(X, eOut_A, color="purple", label="Eout A")
axs[1].plot(X, eOut_C, color="purple", label="Eout C")

#plot training fxn
axs[0].plot(X, F, color="black", label="F")
axs[1].plot(X, F, color="black", label="F")

print("\nA")
print(f"Avg gBar = {np.mean(g_barA)}")
print(f"Avg var = {np.mean(var_A)}")
print(f"Avg bias = {np.mean(bias_A)}")
print(f"Avg eOut = {np.mean(eOut_A)}")
print("\nC")
print(f"Avg gBar = {np.mean(g_barC)}")
print(f"Avg var = {np.mean(var_C)}")
print(f"Avg bias = {np.mean(bias_C)}")
print(f"Avg eOut = {np.mean(eOut_C)}")

axs[0].legend(loc='best')
axs[1].legend(loc='best')
plt.show()
