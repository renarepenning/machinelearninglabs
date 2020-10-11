# ML Homework 3 -- Rena Repenning
import numpy as np
import matplotlib.pyplot as plt
import sys

fig, axs = plt.subplots(4)
N = 100
d = 2
X = np.random.uniform(-1, 1, size=(N, d+1))
X[:, 0] = 1
w = np.random.uniform(-1, 1, size=(d+1))
Y = np.sign(np.dot(X, w))
# ADD NOISE
noise = 5
n = np.random.randint(0, 100, noise)
for i in range(noise):
    Y[n[i]] = Y[n[i]] * -1
#
ind_pos = np.where(Y == 1)[0]
ind_neg = np.where(Y == -1)[0]
axs[0].plot(X[ind_pos, 1], X[ind_pos, 2], 'ro')
axs[2].plot(X[ind_pos, 1], X[ind_pos, 2], 'ro')
axs[0].plot(X[ind_neg, 1], X[ind_neg, 2], 'bx')
axs[2].plot(X[ind_neg, 1], X[ind_neg, 2], 'bx')
X2 = (-(w[1]/w[2]) * np.linspace(-1, 1, 50)) - w[0]/w[2]
axs[0].plot(np.linspace(-1, 1, 50), X2, label='f: targetFxn')
axs[2].plot(np.linspace(-1, 1, 50), X2, label='f: targetFxn')


def pla(X, Y, pocket):
    training_vector = np.random.uniform(-1, 1, size=(d+1))
    e_ins = []
    total_ein = 0
    bestEin = 1000000
    bestTV = training_vector

    for i in range(1000):
        dot_prod = np.dot(X, training_vector)
        rez = np.sign(dot_prod)
        #chose random point to check
        r = np.random.randint(0, 100, 1)
        if rez[r] != Y[r]:
            training_vector += X[r][0]*Y[r]
        thisEin = np.count_nonzero((rez-Y)/len(X))

        if pocket:
            if bestEin > thisEin:
                bestEin = thisEin
                bestTV = training_vector
        e_ins.append( bestEin if pocket else thisEin )

    return bestTV if pocket else training_vector, e_ins


x = np.linspace(-1, 1, 100)
itr = np.linspace(0, 1000, 1000)
# run PLA
H = pla(X, Y, False)
pla_trained = H[0]
pla_eIn = H[1]
print("PLA e in: ",np.sum(H[1])/1000)
pla_Y = (-(pla_trained[1] * x)/pla_trained[2]) - (pla_trained[0]/pla_trained[2])
axs[0].plot(x, pla_Y, label='PLA H')
axs[1].plot(itr, pla_eIn, label='PLA eIns')

# run pocket version of PLA
G = pla(X, Y, True)
pocket_trained = G[0]
pocket_eIn = G[1]
print("Pocket e in: ",np.sum(G[1])/1000)
pocket_Y = (-(pocket_trained[1] * x)/pocket_trained[2]) - (pocket_trained[0]/pocket_trained[2])
axs[2].plot(x, pocket_Y, label='Pocket H')
axs[3].plot(itr, pocket_eIn, label='Pocket eIn')

#Linear Regression Version of PLA
Xdag = np.matmul( np.linalg.pinv(np.matmul(X.transpose(), X)), X.transpose() )
w = np.matmul(Xdag, Y)
wTx = np.matmul(w, X.transpose())
xw = np.matmul(X, w)
term1 = np.matmul(wTx,xw)
t2a = [w[0]*2, w[1]*2, w[2]*2]
t2b = np.matmul(X.transpose(), Y)
term2 = np.matmul(t2a, t2b)
term3 = np.matmul(Y.transpose(), Y)
A = np.subtract(term1, term2)
B = np.add(A, term3)
C = B*(1/100)
print("Regression E in: ", C)
rY = (-(w[1] * X)/w[2]) - (w[0]/w[2])
axs[0].plot(X, rY, label='Linear Regression')
axs[2].plot(X, rY, label='Linear Regression')

axs[0].legend(loc="best")
axs[1].legend(loc="best")
axs[2].legend(loc="best")
axs[3].legend(loc="best")
plt.show()
