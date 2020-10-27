# ML Homework 4 -- Rena Repenning
import numpy as np
import matplotlib.pyplot as plt
import sys

def mapLegendrePoly(X):
    Z = np.dstack((np.ones(X.shape), X, 1/2*(3*X**2 - 1), 1/2*(5*X**3-3*X), 1/8*(35*X**4-30*X**2+3)) )[0]
    return Z

N = 5
# RANDOMLY SAMPLE 5 points D={(xi,yi)} using independent Gaussian noise s with sigma = 0.1.
Sigmas = np.random.normal(0, .1, (N))
X = np.random.uniform(-1, 1, size=N)
Y = X**2+Sigmas
Z = mapLegendrePoly(X)

plt.xlim(-1, 1)
plt.ylim(-2, 2)
#Plot all Xi, Yi
plt.plot(X, Y, 'ro')
#target curve
Xvals = np.arange(-1,1,0.01)
f = np.power(Xvals, 2)
plt.plot(Xvals, f, label="f", color="black")

L =[0, 1E-5, 1E-2,  1E0]
for i in range(len(L)):
    lam = L[i]
    l = str(lam)
    ZtZ = np.matmul(np.transpose(Z),Z)
    Z1 = np.linalg.pinv( np.add(ZtZ,  lam*np.identity(Z.shape[0]) ))
    Wreg = np.matmul(Z1, np.matmul(np.transpose(Z),Y) )
    print("Lam: ", lam, "W reg: ", Wreg)

    #curve from hypothesis h(x) = w^Tz
    Zmap = mapLegendrePoly(Xvals)
    h = np.dot(Zmap,  np.transpose(Wreg))
    plt.plot(Xvals, h, label='h for lam='+ l)

    #OUT OF SAMPLE ERROR
    print("Eout of ", lam, ": ", np.mean(np.power(np.subtract(np.dot(Zmap, Wreg), f), 2)))

plt.legend(loc="best")
plt.show()
