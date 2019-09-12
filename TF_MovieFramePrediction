import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
from data_utils import get_train_batch, get_val_batch, get_test_batch
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, BatchNormalization, Flatten, Activation
from keras.models import Sequential, Model, load_model
from keras import optimizers, metrics
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import LSTM, Lambda, TimeDistributed, GRU, RepeatVector, CuDNNGRU

%matplotlib inline
plt.rcParams['figure.figsize'] = (12.0, 6.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def plot_image(image, shape=[64, 64, 3]):
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")

def plot_images(images, num_images):
    fig = plt.figure(figsize=(24, 8))
    for i in range(0, num_images):
        fig.add_subplot(num_images//5, 5, i+1)
        plt.imshow(images[i])
        plt.axis("off")
    plt.show()
    
encoder = load_model('./models/Encoder_0613_128.hdf5')
decoder = load_model('./models/Decoder_0613_128.hdf5')
encoder.trainable = False
decoder.trainable = False
encoder.summary()
decoder.summary()

dim = 4*4*8

time_steps = 10;
num_outputs = 10;
n_train = 20000
n_val = 500

val_input_batch = np.empty(shape = (n_val*2, time_steps, dim), dtype = np.float32)
val_output_batch = np.empty(shape = (n_val*2, num_outputs, dim), dtype = np.float32)
for i in range(n_val):
    inputs, outputs = get_val_batch(i, i+1)
    inputs = np.reshape(inputs, (-1, 64, 64, 3))
    temp = encoder.predict_on_batch(inputs)
    temp = temp[None, :, :]
    val_input_batch[i] = temp
    outputs = np.reshape(outputs, (-1, 64, 64, 3))
    temp = encoder.predict_on_batch(outputs)
    temp = temp[None, :, :]
    val_output_batch[i] = temp
for i in range(n_val):
    outputs, inputs = get_val_batch(i, i+1)
    inputs = np.reshape(inputs, (-1, 64, 64, 3))
    inputs = np.flip(inputs, 0)
    temp = encoder.predict_on_batch(inputs)
    temp = temp[None, :, :]
    val_input_batch[n_val+i] = temp
    outputs = np.reshape(outputs, (-1, 64, 64, 3))
    outputs = np.flip(outputs, 0)
    temp = encoder.predict_on_batch(outputs)
    temp = temp[None, :, :]
    val_output_batch[n_val+i] = temp
    
initializer = keras.initializers.glorot_uniform(seed = None)
regularizer = keras.regularizers.l2(1e-6)

inputs = Input(shape = (10, 128))
X = LSTM(1200, activation = 'relu', recurrent_activation = 'hard_sigmoid', kernel_initializer = 'glorot_uniform',
         recurrent_initializer = 'orthogonal', kernel_regularizer = None, dropout = 0.0, recurrent_dropout = 0.0,
         return_sequences = False, go_backwards = False, use_bias = True, bias_initializer = 'zeros',activity_regularizer = None)(inputs)
X = RepeatVector(10)(X)
X = LSTM(1200, activation = 'relu', recurrent_activation = 'hard_sigmoid', kernel_initializer = 'glorot_uniform',
         recurrent_initializer = 'orthogonal', kernel_regularizer = None, dropout = 0.0, recurrent_dropout = 0.0,
         return_sequences = True, go_backwards = False, use_bias = True, bias_initializer = 'zeros',activity_regularizer = None)(X)
outputs = TimeDistributed(Dense(units = 128, activation = 'tanh', kernel_initializer = initializer,
                                bias_initializer = 'zeros', kernel_regularizer = regularizer,), input_shape = (10, 1200))(X)

model = Model(inputs, outputs)

opt = optimizers.Adam(lr = 2e-4, decay = 0.0, amsgrad = False)
model.compile(optimizer = opt, loss = 'mse', metrics = ['mse'])

model.summary()
dictionary = {v.name: i for i, v in enumerate(model.layers)}

def image_augmentation(inputs, outputs, opt, time_steps = 10):
    inputs_ = np.zeros_like(inputs, dtype = np.float32)
    outputs_ = np.zeros_like(outputs, dtype = np.float32)
    if opt == 0:
        inputs_ = inputs
        outputs_ = outputs
    elif opt == 1:
        for i in range(time_steps):
            inputs_[i][3:61, 3:61] = inputs[i][:-6, :-6]
            outputs_[i][3:61, 3:61] = outputs[i][:-6, :-6]
    elif opt == 2:
        for i in range(time_steps):
            inputs_[i][3:61, 3:61] = inputs[i][6:, :-6]
            outputs_[i][3:61, 3:61] = outputs[i][6:, :-6]
    elif opt == 3:
        for i in range(time_steps):
            inputs_[i][3:61, 3:61] = inputs[i][6:, 6:]
            outputs_[i][3:61, 3:61] = outputs[i][6:, 6:]
    elif opt == 4:
        for i in range(time_steps):
            inputs_[i][3:61, 3:61] = inputs[i][:-6, 6:]
            outputs_[i][3:61, 3:61] = outputs[i][:-6, 6:]
    elif opt == 5:
        for i in range(time_steps):
            inputs_[i][:-6, :-6] = inputs[i][3:61, 3:61]
            outputs_[i][:-6, :-6] = outputs[i][3:61, 3:61]
    elif opt == 6:
        for i in range(time_steps):
            inputs_[i][6:, :-6] = inputs[i][3:61, 3:61]
            outputs_[i][6:, :-6] = outputs[i][3:61, 3:61]
    elif opt == 7:
        for i in range(time_steps):
            inputs_[i][6:, 6:] = inputs[i][3:61, 3:61]
            outputs_[i][6:, 6:] = outputs[i][3:61, 3:61]
    elif opt == 8:
        for i in range(time_steps):
            inputs_[i][:-6, 6:] = inputs[i][3:61, 3:61]
            outputs_[i][:-6, 6:] = outputs[i][3:61, 3:61]
    return inputs_, outputs_

def batch_data_generator_for_LSTM(batch_size):
    while(True):
        input_batch = np.empty(shape = (batch_size, time_steps, dim), dtype = np.float32)
        output_batch = np.empty(shape = (batch_size, num_outputs, dim), dtype = np.float32)
        for i in range(batch_size):
            if np.random.randint(0,2) == 0:
                inputs, outputs = get_train_batch(1)
                inputs = np.reshape(inputs, (-1, 64, 64, 3))
                outputs = np.reshape(outputs, (-1, 64, 64, 3))
                opt = np.random.randint(0,9)
                inputs, outputs = image_augmentation(inputs, outputs, opt)
                inputs = np.asarray(inputs)
                outputs = np.asarray(outputs)
                temp = encoder.predict_on_batch(inputs)
                temp = temp[None, :, :]
                input_batch[i] = temp
                temp = encoder.predict_on_batch(outputs)
                temp = temp[None, :, :]
                output_batch[i] = temp
            else:
                outputs, inputs = get_train_batch(1)
                inputs = np.reshape(inputs, (-1, 64, 64, 3))
                inputs = np.flip(inputs, 0)
                outputs = np.reshape(outputs, (-1, 64, 64, 3))
                outputs = np.flip(outputs, 0)
                opt = np.random.randint(0,9)
                inputs, outputs = image_augmentation(inputs, outputs, opt)
                inputs = np.asarray(inputs)
                outputs = np.asarray(outputs)
                temp = encoder.predict_on_batch(inputs)
                temp = temp[None, :, :]
                input_batch[i] = temp
                temp = encoder.predict_on_batch(outputs)
                temp = temp[None, :, :]
                output_batch[i] = temp
        yield input_batch, output_batch

def schedule(epoch, lr):
    decay_rate = 0.5
    decay_step = 100
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr
        
def show_outputs(data_type = 'train', index = 0, is_plotting = True):
    if data_type == 'train':
        inputs, answers = get_train_batch(1)
        inputs = np.reshape(inputs, (-1, 64, 64, 3))
        answers = np.reshape(answers, (-1, 64, 64, 3))
        encoded_inputs = encoder.predict_on_batch(inputs)
        encoded_inputs = encoded_inputs[None, :, :]
        features = model.predict_on_batch(encoded_inputs)
        features = np.reshape(features, (-1, dim))
        outputs = decoder.predict_on_batch(features)
    elif data_type == 'validation':
        inputs, answers = get_val_batch(index, index+1)
        answers = np.reshape(answers, (-1, 64, 64, 3))
        encoded_inputs = encoder.predict_on_batch(inputs)
        encoded_inputs = encoded_inputs[None, :, :]
        features = model.predict_on_batch(encoded_inputs)
        features = np.reshape(features, (-1, dim))
        outputs = decoder.predict_on_batch(features)
    if is_plotting:
        plot_images(answers, num_outputs)
        plot_images(outputs, num_outputs)
    return inputs, outputs
    
epochs = 50
batch_size = 200

initial_epoch = 0

CP = ModelCheckpoint(filepath = './models/model0616_U.hdf5', monitor = 'val_loss', verbose = 1, save_best_only = False, mode = 'auto')
ES = EarlyStopping(monitor = 'val_loss', patience = 30, verbose = 1, mode = 'auto', restore_best_weights = True)
LS = LearningRateScheduler(schedule, verbose = 1)

history = model.fit_generator(batch_data_generator_for_LSTM(batch_size), steps_per_epoch = n_train // batch_size, epochs = epochs,
                              verbose = 1, callbacks = [CP], validation_data = (val_input_batch, val_output_batch),
                              shuffle = True, initial_epoch = initial_epoch)

initial_epoch = epochs
_, _ = show_outputs()


