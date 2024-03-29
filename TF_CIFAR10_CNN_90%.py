import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import time
from tensorflow.python.layers import base
import tensorflow.contrib.slim as slim
from utils.data_utils import load_CIFAR10, plot_images
from tensorflow.contrib.layers import l2_regularizer
%matplotlib inline
plt.rcParams['figure.figsize'] = (12.0, 6.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

conf = tf.ConfigProto()
conf.gpu_options.per_process_gpu_memory_fraction = 0.2
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

X_train, Y_train, X_val, Y_val, X_test, Y_test, Class_names = load_CIFAR10()
print('Train data shape ' + str(X_train.shape))
print('Train labels shape ' + str(Y_train.shape))
print('Validation data shape ' + str(X_val.shape))
print('Validataion labels shape ' + str(Y_val.shape))
print('Test data shape ' + str(X_test.shape))
print('Test labels shape ' + str(Y_test.shape))
plot_images(X_train, Y_train, Class_names, Each_Category=True)

X_train = np.concatenate((X_train, X_val))
Y_train = np.concatenate((Y_train, Y_val))
mean = np.mean(X_train,axis=(0,1,2,3))
std = np.std(X_train,axis=(0,1,2,3))
X_train = (X_train-mean)/(std+1e-7)
X_val = (X_val-mean)/(std+1e-7)
X_test = (X_test-mean)/(std+1e-7)
print(X_train.shape, Y_train.shape)

def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features
        
def image_augmentation(x, type):
    if type == 0:
        output = x
        return output
    elif type == 1: 
        output = np.zeros_like(x, dtype = np.float32)
        output[2:30, 2:30] = x[:-4, :-4]
        return output
    elif type == 2:
        output = np.zeros_like(x, dtype = np.float32)
        output[2:30, 2:30] = x[4:, :-4]
        return output
    elif type == 3:
        output = np.zeros_like(x, dtype = np.float32)
        output[2:30, 2:30] = x[4:, 4:]
        return output
    elif type == 4:
        output = np.zeros_like(x, dtype = np.float32)
        output[2:30, 2:30] = x[:-4, 4:]
        return output
    elif type == 5:
        output = np.zeros_like(x, dtype = np.float32)
        output = np.fliplr(x)
        return output
    elif type == 6:
        output = np.zeros_like(x, dtype = np.float32)
        output[2:30, 2:30] = np.fliplr(x[:-4, :-4])
        return output
    elif type == 7:
        output = np.zeros_like(x, dtype = np.float32)
        output[2:30, 2:30] = np.fliplr(x[4:, :-4])
        return output
    elif type == 8:
        output = np.zeros_like(x, dtype = np.float32)
        output[2:30, 2:30] = np.fliplr(x[4:, 4:])
        return output
    elif type == 9:
        output = np.zeros_like(x, dtype = np.float32)
        output[2:30, 2:30] = np.fliplr(x[:-4, 4:])
        return output
        
def my_model_2(X, training):
    weight_decay = 1e-4
    regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)
    
    c = tf.reshape(X, shape = [-1, 32, 32, 3])
    
    c = tf.layers.conv2d(c, filters = 32, kernel_size = 3, strides = 1, padding = 'SAME', 
                         kernel_regularizer = regularizer, activation = tf.nn.relu)
    c = tf.layers.batch_normalization(c, training = training, momentum = 0.9, epsilon=0.001)
    c = tf.layers.conv2d(c, filters = 32, kernel_size = 3, strides = 1, padding = 'SAME', 
                         kernel_regularizer = regularizer, activation = tf.nn.relu)
    c = tf.layers.batch_normalization(c, training = training, momentum = 0.9, epsilon=0.001)
    c = tf.nn.max_pool(c, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    c = tf.layers.dropout(c, 0.2, training = training)
    
    c = tf.layers.conv2d(c, filters = 64, kernel_size = 3, strides = 1, padding = 'SAME', 
                         kernel_regularizer = regularizer, activation = tf.nn.relu)
    c = tf.layers.batch_normalization(c, training = training, momentum = 0.9, epsilon=0.001)
    c = tf.layers.conv2d(c, filters = 64, kernel_size = 3, strides = 1, padding = 'SAME', 
                         kernel_regularizer = regularizer, activation = tf.nn.relu)
    c = tf.layers.batch_normalization(c, training = training, momentum = 0.9, epsilon=0.001)
    c = tf.nn.max_pool(c, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    c = tf.layers.dropout(c, 0.3, training = training)
    
    c = tf.layers.conv2d(c, filters = 128, kernel_size = 3, strides = 1, padding = 'SAME', 
                         kernel_regularizer = regularizer, activation = tf.nn.relu)
    c = tf.layers.batch_normalization(c, training = training, momentum = 0.9, epsilon=0.001)
    c = tf.layers.conv2d(c, filters = 128, kernel_size = 3, strides = 1, padding = 'SAME', 
                         kernel_regularizer = regularizer, activation = tf.nn.relu)
    c = tf.layers.batch_normalization(c, training = training, momentum = 0.9, epsilon=0.001)
    c = tf.nn.max_pool(c, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    c = tf.layers.dropout(c, 0.4, training = training)
    
    c = tf.layers.conv2d(c, filters = 256, kernel_size = 3, strides = 1, padding = 'SAME', 
                         kernel_regularizer = regularizer, activation = tf.nn.relu)
    c = tf.layers.batch_normalization(c, training = training, momentum = 0.9, epsilon=0.001)
    c = tf.layers.conv2d(c, filters = 256, kernel_size = 3, strides = 1, padding = 'SAME', 
                         kernel_regularizer = regularizer, activation = tf.nn.relu)
    c = tf.layers.batch_normalization(c, training = training, momentum = 0.9, epsilon=0.001)
    c = tf.nn.max_pool(c, ksize = [1, 2, 2, 1], strides = [1, 1, 1, 1], padding = 'SAME')
    c = tf.layers.dropout(c, 0.5, training = training)
    
    c, num_features = flatten_layer(c)
    c = tf.reshape(c, shape = [-1, num_features])
    output = tf.layers.dense(c, 10, activation = 'softmax')
    return output
    
    def Inception_module(Input, C1, C3_R, C3, C5_R, C5, P3_R):
    '''
    C1, C3, C5: number of filters for the main convolutions
    C3_R, C5_R, P3_R: number of filters for the dimensionality reduction convolutions
    '''

    conv1 = tf.layers.conv2d(Input, filters = C1, kernel_size = 1, strides = 1, padding = 'SAME', activation = tf.nn.relu)
    
    conv2_1 = tf.layers.conv2d(Input, filters = C3_R, kernel_size = 1, strides = 1, padding = 'SAME', activation = tf.nn.relu)
    conv2 = tf.layers.conv2d(conv2_1, filters = C3, kernel_size = 3, strides = 1, padding = 'SAME', activation = tf.nn.relu)
    
    conv3_1 = tf.layers.conv2d(Input, filters = C5_R, kernel_size = 1, strides = 1, padding = 'SAME', activation = tf.nn.relu)
    conv3 = tf.layers.conv2d(conv3_1, filters = C5, kernel_size = 5, strides = 1, padding = 'SAME', activation = tf.nn.relu)
    
    conv4_1 = tf.nn.max_pool(Input, ksize = [1, 3, 3, 1], strides = [1, 1, 1, 1], padding = 'SAME', name = 'conv4_1')
    conv4 = tf.layers.conv2d(conv4_1, filters = P3_R, kernel_size = 1, strides = 1, padding = 'SAME', activation = tf.nn.relu)
    
    Inception = tf.concat([conv1, conv2, conv3, conv4], axis=3, name='Inception')

    return Inception
    
class my_model(object):
    def __init__(self):

        self.X_raw = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name="X")
        self.y = tf.placeholder(tf.int32, shape=[None], name="y")
        self.labels = tf.one_hot(self.y, 10, dtype=tf.int32)
        
        self.training = tf.placeholder(tf.bool, shape=[], name='training')
        self.global_step = tf.Variable(0, trainable = False)
        self.learning_rate = tf.train.exponential_decay(1e-2, self.global_step, int(50*80*10), 0.5, staircase = True)
        
        self.prediction = my_model_2(self.X_raw, self.training)
        self.pred = tf.argmax(self.prediction, axis=1, output_type=tf.int32)
        
        self.cross_entropy = tf.losses.softmax_cross_entropy(logits = self.prediction, onehot_labels = self.labels)
        self.cost = tf.reduce_mean(self.cross_entropy)
        
        self.opt = tf.train.MomentumOptimizer(learning_rate = self.learning_rate, momentum=0.9)
        
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.train_op = self.opt.minimize(self.cost, global_step = self.global_step)
        
        self.mistakes = tf.reduce_mean(tf.cast(tf.not_equal(self.y, self.pred), tf.float32))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y, self.pred), tf.float32))

        
 def train_model(session, model, X, Y, epochs=1, batch_size=500):
    print('[*] Training Start! [*]')
    session.run(tf.global_variables_initializer())
    for step in range(epochs):
        index = np.random.permutation(len(X))
        X, Y = X[index], Y[index]
        cvs_ = 5
        large_num = 10000
        for cv in range(cvs_):
            current_time = time.time()
            X_val = X[cv*large_num:(cv+1)*large_num]
            Y_val = Y[cv*large_num:(cv+1)*large_num]
            X_train = np.concatenate((X[:cv*large_num], X[(cv+1)*large_num:]), axis = 0)
            Y_train = np.concatenate((Y[:cv*large_num], Y[(cv+1)*large_num:]), axis = 0)
            n_batches = len(X_train) // batch_size
            num_aug = 10
            random_list = np.random.permutation(len(X_train))
            for aug in range(num_aug):
                for iteration in range(n_batches):
                    idx = random_list[iteration*batch_size:(iteration+1)*batch_size]
                    X_batch = np.asarray([image_augmentation(X_train[idx[j]], np.random.randint(0,num_aug)) for j in range(batch_size)])
                    y_batch = Y_train[idx]
                    session.run([model.train_op], feed_dict = {model.X_raw: X_batch, model.y: y_batch, model.training: True})
            cross_entropy = 0
            train_accuracy = 0
            random_list = np.random.permutation(len(X_train))
            for iteration in range(n_batches):
                idx = random_list[iteration*batch_size:(iteration+1)*batch_size]
                X_batch = X_train[idx]
                y_batch = Y_train[idx]
                _, ce, accuracy = session.run([model.pred, model.cross_entropy, model.accuracy],
                                                             feed_dict = {model.X_raw: X_batch, model.y: y_batch, model.training: False})
                cross_entropy += ce
                train_accuracy += accuracy
            n_batches = len(X_val) // batch_size
            val_cross_entropy = 0
            val_accuracy = 0
            random_list = np.random.permutation(len(X_val))
            for iteration in range(n_batches):
                idx = random_list[iteration*batch_size:(iteration+1)*batch_size]
                X_val_batch = X_val[idx]
                y_val_batch = Y_val[idx]
                _, ce, accuracy = session.run([model.pred, model.cross_entropy, model.accuracy], 
                                          feed_dict = {model.X_raw: X_val_batch, model.y: y_val_batch, model.training: False})
                val_cross_entropy += ce
                val_accuracy += accuracy
            dur = time.time() - current_time
            print('[*] Epoch:[%3d] CV Process[%1d/5] Time: %.2fsec, Train Loss: %.2f, Val Loss: %.2f, Train Accuracy: %.2f%%, Val Accuracy: %.2f%%'
                  % (step+1, cv+1, dur, cross_entropy/4, val_cross_entropy, 
                     train_accuracy * 100 / X_train.shape[0] * batch_size, val_accuracy * 100 / X_val.shape[0] * batch_size))   
    print("[*] Training done! [*]")
    
def validate_model(session, model, X, Y, batch_size=500):
    n_batches = len(X) // batch_size
    cross_entropy = 0
    val_accuracy = 0
    random_list = np.random.permutation(len(X))
    for iteration in range(n_batches):
        idx = random_list[iteration*batch_size:(iteration+1)*batch_size]
        X_batch = X[idx]
        y_batch = Y[idx]
        _, ce, accuracy = session.run([model.pred, model.cross_entropy, model.accuracy], 
                                          feed_dict = {model.X_raw: X_batch, model.y: y_batch, model.training: False})
        cross_entropy += ce
        val_accuracy += accuracy
    return (cross_entropy / X.shape[0] * batch_size, val_accuracy / X.shape[0] * batch_size)
    
    tf.reset_default_graph()    

model = my_model() 
#model_summary()

with tf.Session(config=conf) as sess:
    train_model(sess, model, X_train, Y_train, epochs = 1)
    print("(Loss, Accuracy) on Training Dataset (%.4f, %.4f)" % validate_model(sess, model, X_train, Y_train))
    print("(Loss, Accuracy) on Validataion Dataset (%.4f, %.4f)" % validate_model(sess, model, X_val, Y_val))
    
    #Save your final model
    saver = tf.train.Saver()
    saver.save(sess, "./model_checkpoints/my_model_final")
    
    tf.reset_default_graph()  

with tf.Session(config=conf) as sess:
    model = my_model()
    saver = tf.train.Saver()
    saver.restore(sess, "./model_checkpoints/my_model_final")
    print("(Loss, Accuracy) on Test Dataset (%.4f, %.4f)" % validate_model(sess, model, X_test, Y_test))
