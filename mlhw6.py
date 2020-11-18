import numpy as np
import matplotlib.pyplot as plt9

# A - randomly generate N=100 pairs of two dimensional data {x1, ... x100 } by uniformly sampling them on the domain [-1,1] x [-1, 1]
x1 = np.random.uniform(-1, 1, 100)
x2 = np.random.uniform(-1, 1, 100)
X_pts = np.dstack((np.ones(x1.shape), x1, x2))[0]
ws1 = np.array([0, 1, -1])
ws2 = np.array([0, 1, 1])
h1 = np.sign(np.matmul(X_pts, np.transpose(ws1)))
h2 = np.sign(np.matmul(X_pts, np.transpose(ws2)))

# B - compute the true labels y1,y2,...., yN to those data by XOR
def xor(a, b):
    S = []
    for x in range(len(a)):
        if (h1[x]>0 and h2[x]<0) or (h1[x]<0 and h2[x]>0):
            S.append(1)
        else:
            S.append(-1)
    return S
Y = xor(h1, h2)


# C - Copy the weight matrices W^(1), W^(2), and W^(3) from L21 p5
w1 = np.array([ws1, ws2])
w2 = np.array([[-1.5, 1, -1], [-1.5, -1, 1]])
w3 = np.array([1.5, 1, 1])
# ---- implement forward propogation algorithm using W^ns
def forwardProp(X, weights, theta):
    #iterate through forward propogation process
    for w in weights:
        s = np.dot(X, np.transpose(w))
        X = np.c_[(np.ones(X.shape[0]), theta(s))]
    return X[:,1] #return L


# D - Predict the labels of x1, ... , x100 using forward propagation. # What is the E_in using the squared error?
def Ein(Hx, Y):
    return (1/len(Y)) * np.sum(np.power(np.subtract(Hx, Y), 2))

labels_Sign = forwardProp(X_pts, [w1, w2, w3], np.sign) #theta(t)=sign(t)
print("\ntheta=sign()\nEin = ", Ein(labels_Sign, Y),"\nLABELS:\n", labels_Sign)


# E - Repeat part d using theta(t)=tanh(t) for all nodes. What is the E_in using the squared error?
labels_t = forwardProp(X_pts, [w1, w2, w3], np.tanh)
print("\n\ntheta=tanh()\nEin = ", Ein(labels_t, Y),"\nLABELS:\n", labels_t)
