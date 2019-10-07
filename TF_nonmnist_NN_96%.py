from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import os

#configuration for gpu usage
conf = tf.ConfigProto()
# you can modify below as you want
#conf.gpu_options.per_process_gpu_memory_fraction = 0.4
#conf.gpu_options.allow_growth = True
#os.environ['CUDA_VISIBLE_DEVICES']='0'
print(tf.__version__)

#mean = np.mean(train_dataset)
#std = np.std(train_dataset)
X_train = train_dataset
Y_train = train_labels
X_val = valid_dataset
Y_val = valid_labels
X_test = test_dataset
Y_test = test_labels

import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import time
from tensorflow.python.layers import base
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import l2_regularizer
%matplotlib inline
plt.rcParams['figure.figsize'] = (12.0, 6.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features

def get_model_params(sess):
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {gvar.op.name: value for gvar, value in zip(gvars, sess.run(gvars))}

def restore_model_params(sess, model_params):
    gvar_names = list(model_params.keys())
    assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign")
                  for gvar_name in gvar_names}
    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
    feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
    sess.run(assign_ops, feed_dict=feed_dict)
    
class my_model(object):
    def __init__(self):
        self.X = tf.placeholder(tf.float32, shape=[None, 28*28], name="X")
        self.y = tf.placeholder(tf.int32, shape=[None, 10], name="y")
        self.training = tf.placeholder(tf.bool, shape=[], name="training")
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate")
        #self.learning_rate = tf.train.exponential_decay(5e-3, self.global_step, int((20000/256)*50), 0.8, staircase = True)
        
        weight_decay = 1e-4
        regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)
        c = tf.layers.dense(self.X, 256,kernel_regularizer=regularizer, activation='elu')
        c = tf.layers.dense(c, 256, kernel_regularizer=regularizer, activation='elu')
        c = tf.layers.dense(c, 256, kernel_regularizer=regularizer, activation='elu')
        c = tf.layers.dense(c, 256, kernel_regularizer=regularizer, activation='elu')
        c = tf.layers.dense(c, 256, kernel_regularizer=regularizer, activation='elu')
        c = tf.layers.dense(c, 256, kernel_regularizer=regularizer, activation='elu')
        c = tf.layers.dropout(c, 0.3, training=self.training)
        c = tf.layers.dense(c, 128, kernel_regularizer=regularizer, activation='elu')
        c = tf.layers.dense(c, 128, kernel_regularizer=regularizer, activation='elu')
        c = tf.layers.dense(c, 128, kernel_regularizer=regularizer, activation='elu')
        c = tf.layers.dense(c, 128, kernel_regularizer=regularizer, activation='elu')
        c = tf.layers.dense(c, 128, kernel_regularizer=regularizer, activation='elu')
        c = tf.layers.dense(c, 128, kernel_regularizer=regularizer, activation='elu')
        c = tf.layers.dropout(c, 0.3, training=self.training)
        self.prediction = tf.layers.dense(c, 10, kernel_regularizer=regularizer, activation='softmax')
        self.pred = tf.argmax(self.prediction, axis=1, output_type=tf.int32)
        #c = tf.layers.dropout(c, 0.5, training=self.training)
        #c = tf.layers.batch_normalization(c, training=self.training, momentum=0.9, epsilon=0.001)
        
        self.cross_entropy = tf.losses.softmax_cross_entropy(logits=self.prediction, onehot_labels=self.y)
        base_loss = tf.reduce_mean(self.cross_entropy)
        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.cost = tf.add_n([base_loss] + reg_loss, name="loss")
        
        #self.opt = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.99, use_nesterov=True)
        self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        #self.opt = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate)
        
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.train_op = self.opt.minimize(self.cost, global_step=self.global_step)
        
        self.mistakes = tf.reduce_mean(tf.cast(tf.not_equal(self.pred, tf.cast(tf.argmax(self.y, axis=1), tf.int32)), tf.float32))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, tf.cast(tf.argmax(self.y, axis=1), tf.int32)), tf.float32))
        
total_epochs = 0
best_loss_val = np.infty
checks_since_last_progress = 0
max_checks_without_progress = 50
best_model_params = None

def train_model(session, model, X_train, Y_train, X_val, Y_val, early_stop=False, lr=1e-3, epochs=1, batch_size=128):
    print('[*] Training Start! [*]')
    global total_epochs
    global best_loss_val
    global checks_since_last_progress
    global max_checks_without_progress
    global best_model_params
    for step in range(total_epochs, total_epochs + epochs):
        current_time = time.time()
        n_batches = len(X_train) // batch_size
        random_list = np.random.permutation(len(X_train))
        for iteration in range(n_batches):
            idx = random_list[iteration*batch_size:(iteration+1)*batch_size]
            X_batch = X_train[idx]
            y_batch = Y_train[idx]
            session.run([model.train_op], feed_dict = {model.X: X_batch, model.y: y_batch,
                                                       model.training: True, model.learning_rate: lr})            
        cross_entropy = 0
        train_accuracy = 0
        for iteration in range(n_batches):
            X_batch = X_train[iteration*batch_size:(iteration+1)*batch_size]
            y_batch = Y_train[iteration*batch_size:(iteration+1)*batch_size]
            _, ce, accuracy = session.run([model.pred, model.cross_entropy, model.accuracy], 
                                          feed_dict = {model.X: X_batch, model.y: y_batch, model.training: False})
            cross_entropy += ce
            train_accuracy += accuracy
        n_batches = len(X_val) // batch_size
        val_cross_entropy = 0
        val_accuracy = 0
        for iteration in range(n_batches):
            X_val_batch = X_val[iteration*batch_size:(iteration+1)*batch_size]
            y_val_batch = Y_val[iteration*batch_size:(iteration+1)*batch_size]
            _, ce, accuracy = session.run([model.pred, model.cross_entropy, model.accuracy], 
                                          feed_dict = {model.X: X_val_batch, model.y: y_val_batch, model.training: False})
            val_cross_entropy += ce
            val_accuracy += accuracy
        dur = time.time() - current_time
        print('[*] Epoch:[%3d] Time: %.2fsec, Train Loss: %.2f, Val Loss: %.2f, Train Accuracy: %.2f%%, Val Accuracy: %.2f%%'
                    % (step + 1, dur, cross_entropy/(X_train.shape[0]/X_val.shape[0]), val_cross_entropy, 
                       train_accuracy * 100 / X_train.shape[0] * batch_size, val_accuracy * 100 / X_val.shape[0] * batch_size))
        if early_stop:
            if val_cross_entropy < best_loss_val:
                best_loss_val = val_cross_entropy
                checks_since_last_progress = 0
                best_model_params = get_model_params(session)
            else: 
                checks_since_last_progress += 1
            if checks_since_last_progress > max_checks_without_progress:
                print('[*] Early Stopping')
                break
    total_epochs += epochs
    print("[*] Training done! [*]")
    
def validate_model(session, model, X, Y, batch_size=500):
    n_batches = len(X) // batch_size
    cross_entropy = 0
    val_accuracy = 0
    for iteration in range(n_batches):
        X_batch = X[iteration*batch_size:(iteration+1)*batch_size]
        y_batch = Y[iteration*batch_size:(iteration+1)*batch_size]
        _, ce, accuracy = session.run([model.pred, model.cross_entropy, model.accuracy], 
                                          feed_dict = {model.X: X_batch, model.y: y_batch, model.training: False})
        cross_entropy += ce
        val_accuracy += accuracy
    return (cross_entropy / X.shape[0] * batch_size, val_accuracy / X.shape[0] * batch_size)
    
tf.reset_default_graph()    

model = my_model() 

sess = tf.Session()
sess.run(tf.global_variables_initializer())
train_model(sess, model, X_train, Y_train, X_val, Y_val, lr=1e-4, batch_size = 128, epochs = 200)

train_model(sess, model, X_train, Y_train, X_val, Y_val, lr=2e-5, batch_size = 128, epochs = 100)

print("(Loss, Accuracy) on Test Dataset (%.4f, %.4f)" % validate_model(sess, model, X_test, Y_test))

saver = tf.train.Saver()
saver.save(sess, "./model_checkpoints/my_model_final")
