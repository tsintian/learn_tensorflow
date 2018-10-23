
import tensorflow as tf
import numpy as np 

class DataLoader():
    def __init__(self):
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        self.train_data= mnist.train.images 
        self.train_labels = np.asarray(mnist.train.labels, dtype=np.int32) 
        self.eval_data = mnist.test.images 
        self.eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    def get_batch(self, batch_size):
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)
        return self.train_data[index,:], self.train_labels[index]