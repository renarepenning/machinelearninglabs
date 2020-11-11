#Rena Repenning -- ML HW 5

## COMMENT runOne() AND runTwo() IN/OUT TO RUN ##

import numpy as np
import matplotlib.pyplot as plt
import sys
import collections
fig, axs = plt.subplots(5)
#These are two-dimensional features (average-intensity and symmetry) of handwritten digits from 0 to 9.
train_data = np.loadtxt("features.small.train")
Y_train = train_data[:, 0]
X_train = train_data[:, 1:]
x_test = np.loadtxt("features.small.test") #typo ==> validation


#1 - K-NEAREST NEIGHBOR CLASSIFIER
def knn(x_test, X_train, Y_train, k):
    Djs = {}
    for i, xj in enumerate(X_train):
        #euclid dist between train and validate
        dj = np.sqrt((x_test[1] - xj[0])**2 + (x_test[2] - xj[1])**2)
        Djs[dj] = i
    # Find k-nn by sorting xj's according to key, d(j)'s
    Djs_items = Djs.items()
    sortedDj = sorted(Djs_items) #RETURNS TUPLES
    k_nns = sortedDj[:k]
    # Count the number of votes for label 0 through 9
    votes = [0]*10
    for i in range(len(k_nns)): #get first k elements
        idx = k_nns[i][1]
        v = int(Y_train[idx])
        votes[v] += 1
    #return mode of K labels
    max = -1
    I = 0
    for i in range(10):
        if votes[i] > max:
            max = votes[i]
            I = i
    return I #return the y_test which is the most highly voted label

def runOne():
    K = [1,11,21,31]
    for k in range(len(K)):
        best = []
        for v in x_test:
            knnRes = knn(v, X_train, Y_train, K[k])
            best.append(knnRes)
        err = 0
        for i in range(len(best)):
            if best[i] == x_test[i][0]:
                err += 1
        err = err/len(best)
        print("prediction err for k = ", K[k], ": ", err)
        axs[k].scatter(x_test[:, 1], x_test[:, 2], c=best)
    #plot actual
    sct = axs[4].scatter(x_test[:, 1], x_test[:, 2], c=x_test[:, 0])
    plt.legend(*sct.legend_elements(), loc="right")
    plt.show()


#2 - RBF classifier
def RBF_classifier(x_test,X_train,Y_train, r):
    alphas = []
    for i, xj in enumerate(X_train):
        euc = np.sqrt((x_test[1] - xj[0])**2 + (x_test[2] - xj[1])**2)
        z = euc / r
        Aj = np.exp(-0.5*(z**2))
        alphas.append( (i, Aj) )
    #Count the number of votes for label 0 through 9 weighted by alpha(j)'s
    votes = [0]*10
    for i in range(len(X_train)):
        pred_Y = int(Y_train[int(alphas[i][0])]) #get index to "vote" for
        votes[pred_Y] += alphas[i][1]
    #return mode of K labels
    max = -1
    I = 0
    for i in range(10):
        if votes[i] > max:
            max = votes[i]
            I = i
    return I # Return y_test which is the most highly voted label

def runTwo():
    R =  [0.01, 0.05, 0.1, 0.5, 1]
    for r in range(len(R)):
        best = []
        for v in x_test:
            rbf = RBF_classifier(v, X_train, Y_train, R[r])
            best.append(rbf)
        err = 0
        for i in range(len(best)):
            if best[i] == x_test[i][0]:
                err += 1
        err = err/len(best)
        print("prediction err on r = ", R[r], ": ", err)
        axs[r].scatter(x_test[:, 1], x_test[:, 2], c=best)
    #plot actual
    sct = axs[4].scatter(x_test[:, 1], x_test[:, 2], c=x_test[:, 0])
    plt.legend(*sct.legend_elements(), loc="right")
    plt.show()



## RUN PROBLEM 1##
# runOne()
## RUN PROBLEM 2 ##
# runTwo()
