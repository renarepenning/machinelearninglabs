import numpy as np
import matplotlib.pyplot as plt

#A

N = 1000 #data set size
d = 10 #2 classes of data

# Generate random training data
X = np.random.uniform(-1, 1, size=(N, d+1))
X[:, 0] = 1

# Calculate weights vector
w = np.random.uniform(-1, 1, size=(d+1))

# Compute true labels for the training data
Y = np.sign(np.dot(X, w))
ind_pos = np.where(Y == 1)[0]#positive examples
ind_neg = np.where(Y == -1)[0]#negative examples
# If there are two few positive or negative examples, then repeat
# w = np.random... until you find good values for w

# Plot points
plt.clf()
plt.plot(X[ind_pos, 1], X[ind_pos, 2], 'ro')#red dot points
plt.plot(X[ind_neg, 1], X[ind_neg, 2], 'bx')#blue 'x' points

## Generate random target function: f(x) = w^Tx
# 	The equation for decision boundary in two dimension is  h(x) = w^T x = w0 + w1*x1 + w2*x2 = 0.
# 	Equivalently, x2 = -(w1/w2)*x1 - w0/w2.
X2 = (-(w[1]/w[2]) * np.linspace(-1, 1, 50)) - w[0]/w[2]

#plot targetFxn, label axis, title
plt.plot(np.linspace(-1, 1, 50), X2, label='f: targetFxn')
plt.title('Target fxn + linearly separated data')
plt.xlabel('X')
plt.ylabel('Y')

## B.
## run PLA on data set generated above
def pla(X, Y):
	training_vector = np.random.uniform(-1, 1, size=(d + 1))
	elements = len(X)
	ctr = 0
	incorrect_v = True
	while incorrect_v:
		dot_prod = np.dot(X, training_vector)
		rez = np.sign(dot_prod)

		#find issues
		for i in range(elements):
			if rez[i] != Y[i]:
				break

		#make sure loop didnt end at a break
		if i != elements - 1:
			training_vector += np.dot(X[i,:], Y[i])
			ctr += 1
		#if loop ran its course
		else:
			incorrect_v = False

	print(f"{ctr} updates")
	return training_vector

#RUN PLA
H = pla(X, Y)

new_slope = H[1]/H[2]
b = H[0]/H[2]

new_Y = -(new_slope * (np.linspace(-1, 1, 100))) - b
plt.plot(np.linspace(-1, 1, 100), new_Y, label='H')
plt.legend(loc="best")
plt.show()
