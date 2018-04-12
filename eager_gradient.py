
import tensorflow as tf 
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()


def square(x):
    return tf.multiply(x, x)

grad = tfe.gradients_function(square)

print('square of 3 : {}'.format(square(3.)))
print("derivative of x^2: {}".format(grad(3.)))
