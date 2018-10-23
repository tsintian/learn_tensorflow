

#%%
from builtins import range

import tensorflow as tf


if not tf.executing_eagerly:
    tf.enable_eager_execution()

# Tensorflow provides us with the powerful Automatic Differentiation mechanism
# for differentiation.
# The following codes show how to utilize tf.GradientTape() to get 
# the slop of g(x) = x^2 at x = 3.
x = tf.get_variable('x', shape=[1], initializer=tf.constant_initializer(3.0))
with tf.GradientTape() as tape: 
    # All steps in the context of tf.GradientType() as recorded for differentiation.
    y = tf.square(x)
y_grad = tape.gradient(y,x) # Differentiate y with respect to x.
print([y.numpy(), y_grad.numpy()])


#%%
# We can also utilize tf.GradientTape() to calculate the derivatives of a multivariable function.
# , a vector or a matrix.
X = tf.constant([[1., 2.], [3., 4.]])
y = tf.constant([[1.], [2.]])

w = tf.get_variable('w', shape=[2, 1], initializer=tf.constant_initializer([[1.], [2.]]))
b = tf.get_variable('b', shape=[1], initializer=tf.constant_initializer([1.]))

with tf.GradientTape() as tape:
    L = 0.5 * tf.reduce_sum(tf.square(tf.matmul(X, w) + b -y))
w_grad, b_grad = tape.gradient(L, [w, b])
print([L.numpy(), w_grad.numpy(), b_grad.numpy()])


# Linear Regression

# Given the following data:

# Year  2013,   2014,  2015,  2016, 2017
# Price 12000, 14000, 15000, 16500, 17500

# Now we want to do linear regression on the data above, using linear regression model y= ax + b
# to fit the data, where a and b are unkown parameters.
#%%

# First, we difine and normalize the data.
import numpy as np 

X_raw = np.array([2013, 2014, 2015, 2016, 2017],dtype=np.float32)
y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)

X = (X_raw - X_raw.min())/(X_raw.max() - X_raw.min())
y = (y_raw -y_raw.min())/(y_raw.max() - y_raw.min())

## Using numpy to solve the linear regression min_a,b L(a, b) = \sigma_i (ax_i + b - y_i)^2
#%%
a, b = 0, 0

num_epoch = 10000
learnint_rate = 1e-3
for e in range(num_epoch):
    y_pred = a*X + b
    grad_a, grad_b = (y_pred - y).dot(X), (y_pred - y).sum()

# Update parameters.
a, b = a - learnint_rate * grad_a, b - learnint_rate * grad_b

print(a, b)

#%%
# Utilize Tensorflow to do linear regression
X = tf.constant(X)
y = tf.constant(y)

a = tf.get_variable('a', dtype=tf.float32, shape=[], initializer=tf.zeros_initializer())
b = tf.get_variable('b', dtype=tf.float32, shape=[], initializer=tf.zeros_initializer())

variables = [a, b]

num_epoch = 10000
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learnint_rate)
for e in range(num_epoch):
    with tf.GradientTape() as tape:
        y_pred = a * X + b 
        loss = 0.5 * tf.reduce_sum(tf.square(y_pred - y))
    grads = tape.gradient(loss, variables) 
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))

print(a.numpy(), b.numpy())


