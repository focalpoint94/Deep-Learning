def batch_data_generator_for_LSTM(batch_size):
    while(True):
        input_batch = np.empty(shape = (batch_size, time_steps, dim), dtype = np.float32)
        output_batch = np.empty(shape = (batch_size, num_outputs, dim), dtype = np.float32)
        for i in range(batch_size):
            if np.random.randint(0,2) == 0:
                inputs, outputs = get_train_batch(1)
                inputs = np.reshape(inputs, (-1, 64, 64, 3))
                temp = encoder.predict_on_batch(inputs)
                temp = temp[None, :, :]
                input_batch[i] = temp
                outputs = np.reshape(outputs, (-1, 64, 64, 3))
                temp = encoder.predict_on_batch(outputs)
                temp = temp[None, :, :]
                output_batch[i] = temp
            else:
                outputs, inputs = get_train_batch(1)
                inputs = np.flip(inputs, 0)
                inputs = np.reshape(inputs, (-1, 64, 64, 3))
                temp = encoder.predict_on_batch(inputs)
                temp = temp[None, :, :]
                input_batch[i] = temp
                outputs = np.flip(outputs, 0)
                outputs = np.reshape(outputs, (-1, 64, 64, 3))
                temp = encoder.predict_on_batch(outputs)
                temp = temp[None, :, :]
                output_batch[i] = temp
        yield input_batch, output_batch
