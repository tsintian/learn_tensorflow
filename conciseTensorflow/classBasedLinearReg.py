
# Implement the linear model through model class.
#%%
import tensorflow as tf 

if not tf.executing_eagerly():
    tf.enable_eager_execution() 

X = tf.constant([[1., 2., 3.], [4., 5., 6.]])
y = tf.constant([[10.], [20]])

#%%
class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(units = 1, kernel_initializer = tf.zeros_initializer(),
        bias_initializer = tf.zeros_initializer())
    
    def __call__(self, input):
        output = self.dense(input)
        return output

#%%
model = Linear()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

for i in range(100):
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.reduce_mean(tf.square(y_pred - y))
    
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

print(model.variables)