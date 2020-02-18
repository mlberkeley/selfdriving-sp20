import matplotlib.pyplot as plt
plt.ion()
import numpy as np
from sklearn.linear_model import LinearRegression

NUM_HOUSES = 500

np.random.seed(774)

'''
Data generation.
We generate house lot sizes with mean 6k and std dev 1.5k
We suppose prices are typically 300k + 300 * lot.  In addition
we add normally distributed noise with std dev 200k.
'''
lots = 6000 + 1500 * np.random.normal(size = [NUM_HOUSES])
prices = 300 * lots + 2e5 * np.random.normal(size = [NUM_HOUSES]) + 3e5

'''
We use sklearn to find optimal least squares regression line.
Then we scatter plot the data and the learned trend line.
'''
lr = LinearRegression()
lr_lots = np.reshape(lots, [-1, 1])
lr.fit(lr_lots, prices)
preds = lr.predict(lr_lots)
print(f"sklearn: price = lot * {lr.coef_[0]:.2f} + {lr.intercept_:.0f}")

plt.xlabel("Lot Size (sqft)")
plt.ylabel("House Prices ($)")

plt.scatter(lots, prices)
plt.plot(lots, preds, color = 'g', lw = 2.0)


'''
Now we will demonstrate a gradient descent based solution to
the same problem.
'''

'''
c1 corresponds to the slope, c2 the intercept.
Both initialized to random incorrect values.
'''
c_1 = 150
c_2 = 5e5

'''
Predict price from lot sizes using current weights
'''
def eval_preds():
	return c_1 * lots + c_2

preds = eval_preds()

'''
Derivative of predictions with respect to c1.
p = c1 * l + c2 => dp/dc1 = l
'''
def d_preds_c1():
	return lots

'''
Derivative of predictions with respect to c2.
p = c1 * l + c2 => dp/dc2 = 1
'''
def d_preds_c2():
	return np.ones([NUM_HOUSES])

'''
Mean squared error on current predictions.
This is the cost function we would like to minimize.
'''
def eval_MSE():
	return np.mean((preds - prices) ** 2)

'''
Derivative of loss with respect to c1.
loss = (preds - prices)^2 => dloss/dc1 = 2(preds - prices) * dpreds/dc1
Similarily for c2.
'''
def d_MSE_c1():
	return np.mean(2 * d_preds_c1() * (preds - prices))

def d_MSE_c2():
	return np.mean(2 * d_preds_c2() * (preds - prices))

#Learning Rates for c1 and c2
LR_c1 = 7e-9
LR_c2 = 0.7


'''
Apply gradient descent update rule 500 times.
'''
for i in range(500):
	print(f"ITER {i + 1} MSE: {eval_MSE():.0f} c1: {c_1:.2f} c2: {c_2:.0f}  ", end = "\r")

	#To decrease loss move against the gradient.
	c_1 -= LR_c1 * d_MSE_c1()
	c_2 -= LR_c2 * d_MSE_c2()

	preds = eval_preds()

	a, = plt.gca().plot(lots, preds, color = 'r', lw = 2.0)

	plt.gcf().subplots_adjust(left=0.2)

	plt.show()
	plt.pause(0.01)

	a.remove()

'''
Note how Gradient Descent converges to the optimum!
'''
print(f"Gradient Descent: price = lot * {c_1:.2f} + {c_2:.0f}  ")
