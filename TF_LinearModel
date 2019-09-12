%matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

import sys
sys.path.append('./utils')

from mnist import MNIST
data = MNIST(data_dir="data/MNIST/")

print("Size of:")
print("- Training-set:\t\t{}".format(data.num_train))
print("- Validation-set:\t{}".format(data.num_val))
print("- Test-set:\t\t{}".format(data.num_test))

# The images are stored in one-dimensional arrays of this length.
img_size_flat = data.img_size_flat

# Tuple with height and width of images used to reshape arrays.
img_shape = data.img_shape

# Number of classes, one class for each of 10 digits.
num_classes = data.num_classes

import functools

def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

class Model:
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.logits
        self.prediction
        self.optimize
        self.error
        
        self.y_pred
        self.correct_prediction
        
    @lazy_property
    def logits(self):
        data_size = int(self.data.get_shape()[1])
        target_size = int(self.target.get_shape()[1])
        weight = tf.Variable(tf.truncated_normal([data_size, target_size]))
        bias = tf.Variable(tf.constant(0.1, shape=[target_size]))
        return tf.matmul(self.data, weight) + bias
        
    @lazy_property
    def prediction(self):
        self.y_pred = tf.nn.softmax(self.logits)
        y_pred_cls = tf.argmax(self.y_pred, axis = 1)
        return y_pred_cls
        #return tf.argmax(y_pred, axis = 1)
    
    @lazy_property
    def optimize(self):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.target)
        cost = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        return optimizer.minimize(cost)
    
    @lazy_property
    def error(self):
        self.correct_prediction = tf.equal(self.prediction, tf.argmax(self.target, 1))
        accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        incorrect_prediction = tf.not_equal(self.prediction, tf.argmax(self.target, 1))
        mistakes = tf.reduce_mean(tf.cast(incorrect_prediction, tf.float32))
        return mistakes

def plot_images2(images, cls_true, cls_pred, logits, pred):
    assert len(images) == len(cls_true) == 9
    
    plt.rcParams["figure.figsize"] = (16,16)
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        xlabel = "True: {0:1d}, Pred: {1:1d}, logits: {2:.2f}, y_pred: {3:.2f}".format(cls_true[i], cls_pred[i], 
                                                                               np.max(logits[i]), np.max(pred[i]))

        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
        
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

def plot_example_errors2():
    # Use TensorFlow to get a list of boolean values
    # whether each test-image has been correctly classified,
    # and a list for the predicted class of each image.
    
    correct, cls_pred, logits, pred = session.run([model.correct_prediction, model.prediction, model.logits, model.y_pred],
                                    feed_dict=feed_dict_test)
    
    # Negate the boolean array.
    incorrect = (correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.x_test[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.y_test_cls[incorrect]
    
    logits = logits[incorrect]
    pred = pred[incorrect]
    
    # Plot the first 9 images.
    plot_images2(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9],
                logits = logits[0:9],
                pred = pred[0:9])

batch_size = 1000
num_steps = 100
    
tf.reset_default_graph()

# TODO : Model object construction
x = tf.placeholder(tf.float32, [None, img_size_flat])
y_true = tf.placeholder(tf.float32, [None, num_classes])
y_true_cls = tf.placeholder(tf.int64, [None])
model = Model(x, y_true)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    
    for step in range(num_steps):
        # TODO : Model Optimization
        x_batch, y_true_batch, _ = data.random_batch(batch_size=batch_size)
        feed_dict_train = {x: x_batch, y_true: y_true_batch}
        session.run(model.optimize, feed_dict_train)
        error = session.run(model.error, feed_dict_train)
        if ((step+1) % 100 == 0):
            print("Error rate @ iter %d : %f" % (step+1, error))
    
    feed_dict_test = {x: data.x_test, y_true: data.y_test, y_true_cls: data.y_test_cls}
    plot_example_errors2()    
    
// Model2 uses sparse_cross_entropy 
class Model2:
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.logits
        self.prediction
        self.optimize
        self.error
        
        self.y_pred
        self.correct_prediction
        
    @lazy_property
    def logits(self):
        data_size = int(self.data.get_shape()[1])
        target_size = int(self.target.get_shape()[1])
        weight = tf.Variable(tf.truncated_normal([data_size, target_size]))
        bias = tf.Variable(tf.constant(0.1, shape=[target_size]))
        return tf.matmul(self.data, weight) + bias
        
    @lazy_property
    def prediction(self):
        self.y_pred = tf.nn.softmax(self.logits)
        y_pred_cls = tf.argmax(self.y_pred, axis = 1)
        return y_pred_cls
        #return tf.argmax(y_pred, axis = 1)
    
    @lazy_property
    def optimize(self):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.argmax(self.target, 1))
        cost = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        return optimizer.minimize(cost)
    
    @lazy_property
    def error(self):
        self.correct_prediction = tf.equal(self.prediction, tf.argmax(self.target, 1))
        accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        incorrect_prediction = tf.not_equal(self.prediction, tf.argmax(self.target, 1))
        mistakes = tf.reduce_mean(tf.cast(incorrect_prediction, tf.float32))
        return mistakes

batch_size = 1000
num_steps = 500
    
tf.reset_default_graph()

# TODO : Model object construction
x = tf.placeholder(tf.float32, [None, img_size_flat])
y_true = tf.placeholder(tf.float32, [None, num_classes])
y_true_cls = tf.placeholder(tf.int64, [None])
model = Model2(x, y_true)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    
    for step in range(num_steps):
        # TODO : Model Optimization
        x_batch, y_true_batch, _ = data.random_batch(batch_size=batch_size)
        feed_dict_train = {x: x_batch, y_true: y_true_batch}
        session.run(model.optimize, feed_dict_train)
        error = session.run(model.error, feed_dict_train)
        if ((step+1) % 100 == 0):
            print("Error rate @ iter %d : %f" % (step+1, error))
    
    feed_dict_test = {x: data.x_test, y_true: data.y_test, y_true_cls: data.y_test_cls}
    plot_example_errors2()    
