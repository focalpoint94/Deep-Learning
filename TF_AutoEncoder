import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
from data_utils import get_train_batch, get_val_batch, get_test_batch
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, BatchNormalization, Flatten, Activation, Dropout
from keras.models import Sequential, Model, load_model
from keras import optimizers, metrics
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

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
    
n_train = 10000
n_val = 500
batch1, batch2 = get_val_batch(0, n_val)
X_val_batch = np.concatenate([batch1, batch2])
X_val_batch = np.reshape(X_val_batch, (-1, 64, 64, 3))
Y_val_batch = X_val_batch
X_val_batch = np.reshape(X_val_batch, (-1, 64 * 64 * 3))

initializer = keras.initializers.glorot_uniform(seed = None)
regularizer = keras.regularizers.l2(1e-7)

inputs = Input(shape=(64 * 64 * 3,))
encoded = Dense(units = 256, activation = 'tanh', kernel_initializer = initializer, kernel_regularizer = regularizer)(inputs)

L1 = Reshape((8 , 8 , 4))
L2 = Conv2D(32, kernel_size = (3,3), strides = 1, activation = 'relu', padding = 'same', 
                 kernel_initializer = initializer, kernel_regularizer = regularizer)
L3 = BatchNormalization()
L4 = Conv2D(32, kernel_size = (3,3), strides = 1, activation = 'relu', padding = 'same', 
                 kernel_initializer = initializer, kernel_regularizer = regularizer)
L5 = BatchNormalization()
L6 = UpSampling2D(size = (2,2), interpolation = 'nearest')
L7 = Conv2D(16, kernel_size = (3,3), strides = 1, activation = 'relu', padding = 'same', 
                 kernel_initializer = initializer, kernel_regularizer = regularizer)
L8 = BatchNormalization()
L9 = Conv2D(16, kernel_size = (3,3), strides = 1, activation = 'relu', padding = 'same', 
                 kernel_initializer = initializer, kernel_regularizer = regularizer)
L10 = BatchNormalization()
L11 = UpSampling2D(size = (2,2), interpolation = 'nearest')
L12 = Conv2D(8, kernel_size = (3,3), strides = 1, activation = 'relu', padding = 'same', 
                 kernel_initializer = initializer, kernel_regularizer = regularizer)
L13 = BatchNormalization()
L14 = Conv2D(8, kernel_size = (3,3), strides = 1, activation = 'relu', padding = 'same', 
                 kernel_initializer = initializer, kernel_regularizer = regularizer)
L15 = BatchNormalization()
L16 = UpSampling2D(size = (2,2), interpolation = 'nearest')
L17 = Conv2D(3, kernel_size = (3,3), strides = 1, activation = 'relu', padding = 'same',
                 kernel_initializer = initializer, kernel_regularizer = regularizer)

X = L1(encoded)
X = L2(X)
X = L3(X)
X = L4(X)
X = L5(X)
X = L6(X)
X = L7(X)
X = L8(X)
X = L9(X)
X = L10(X)
X = L11(X)
X = L12(X)
X = L13(X)
X = L14(X)
X = L15(X)
X = L16(X)
outputs = L17(X)

# Auto-Encoder
model = Model(inputs, outputs)

# Encoder & Decoder
encoder = Model(inputs, encoded)
decoder_input = Input(shape = (8 * 8 * 4,))
D = L1(decoder_input)
D = L2(D)
D = L3(D)
D = L4(D)
D = L5(D)
D = L6(D)
D = L7(D)
D = L8(D)
D = L9(D)
D = L10(D)
D = L11(D)
D = L12(D)
D = L13(D)
D = L14(D)
D = L15(D)
D = L16(D)
decoded = L17(D)
decoder = Model(decoder_input, decoded)

opt = optimizers.Adam(lr = 5e-3, decay = 0.0, epsilon = 1e-8, amsgrad = False)
model.compile(optimizer = opt, loss = 'mse', metrics = ['mse', 'mae'])

model.summary()
dictionary = {v.name: i for i, v in enumerate(model.layers)}

def batch_data_generator(batch_size):
    while(True):
        batch1, batch2 = get_train_batch(batch_size)
        X_batch = np.concatenate([batch1, batch2])
        X_batch = np.reshape(X_batch, (-1, 64, 64, 3))
        Y_batch = X_batch
        X_batch = np.reshape(X_batch, (-1, 64 * 64 * 3))
        yield X_batch, Y_batch
        
def schedule(epoch, lr):
    decay_rate = 0.5
    decay_step = 250
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr
    
initial_epoch = 0

epochs = 15
batch_size = 40

CP = ModelCheckpoint(filepath = './models/AutoEncoder_0618_Dense256.hdf5', monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'auto')
ES = EarlyStopping(monitor = 'val_loss', patience = 30, verbose = 1, mode = 'auto', restore_best_weights = True)
LS = LearningRateScheduler(schedule, verbose = 1)

history = model.fit_generator(batch_data_generator(batch_size), steps_per_epoch = n_train // batch_size, epochs = epochs, verbose = 1,
                             callbacks = [ES, CP], validation_data = (X_val_batch, Y_val_batch),
                             shuffle = True, initial_epoch = initial_epoch)
initial_epoch = epochs
encoder.save('./models/Encoder_0618_Dense256.hdf5')
decoder.save('./models/Decoder_0618_Dense256.hdf5')

opt = optimizers.Adam(lr = 2e-3, decay = 0.0, epsilon = 1e-8, amsgrad = False)
model.compile(optimizer = opt, loss = 'mse', metrics = ['mse', 'mae'])

epochs = 25
batch_size = 40

CP = ModelCheckpoint(filepath = './models/AutoEncoder_0618_Dense256.hdf5', monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'auto')
ES = EarlyStopping(monitor = 'val_loss', patience = 30, verbose = 1, mode = 'auto', restore_best_weights = True)
LS = LearningRateScheduler(schedule, verbose = 1)

history = model.fit_generator(batch_data_generator(batch_size), steps_per_epoch = n_train // batch_size, epochs = epochs, verbose = 1,
                             callbacks = [ES, CP], validation_data = (X_val_batch, Y_val_batch),
                             shuffle = True, initial_epoch = initial_epoch)
initial_epoch = epochs
encoder.save('./models/Encoder_0618_Dense256.hdf5')
decoder.save('./models/Decoder_0618_Dense256.hdf5')





losses = model.evaluate(X_val_batch, X_val_batch, batch_size = 32, verbose = 1)

// Check with Train Data
batch1, batch2 = get_train_batch(1)
X_batch = np.concatenate([batch1, batch2])
X_batch = np.reshape(X_batch, (-1, 64, 64, 3))
X_batch = np.reshape(X_batch, (-1, 64 * 64 * 3))
outputs = model.predict_on_batch(X_batch)
X_batch = np.reshape(X_batch, (-1, 64, 64, 3))
plot_image(X_batch[10])
plot_image(outputs[10])

// Check with Validation Data
batch1, batch2 = get_val_batch(12, 13)
X_batch = np.concatenate([batch1, batch2])
X_batch = np.reshape(X_batch, (-1, 64, 64, 3))
X_batch = np.reshape(X_batch, (-1, 64 * 64 * 3))
outputs = model.predict_on_batch(X_batch)
X_batch = np.reshape(X_batch, (-1, 64, 64, 3))
plot_images(X_batch, 20)
plot_images(outputs, 20)
plot_image(X_batch[5])
plot_image(outputs[5])

// Check if Encoder & Decoder works well
batch1, batch2 = get_train_batch(1)
X_batch = np.concatenate([batch1, batch2])
X_batch = np.reshape(X_batch, (-1, 64, 64, 3))
X_batch = np.reshape(X_batch, (-1, 64 * 64 * 3))
encoded_imgs = encoder.predict_on_batch(X_batch)
decoded_imgs = decoder.predict_on_batch(encoded_imgs)
X_batch = np.reshape(X_batch, (-1, 64, 64, 3))
plot_image(X_batch[6])
plot_image(decoded_imgs[6])





