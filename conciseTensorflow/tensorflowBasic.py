#%%
import tensorflow as tf 
tf.enable_eager_execution()


a = tf.constant(1)
b = tf.constant(1)
c = tf.add(a ,b)

print(c)
