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
    
total_epochs = 0
best_loss_val = np.infty
checks_since_last_progress = 0
max_checks_without_progress = 1
best_model_params = None

def train_model2(session, model, X_train, Y_train, X_val, Y_val, epochs=1, batch_size=500):
    print('[*] Training Start! [*]')
    global total_epochs
    global best_loss_val
    global checks_since_last_progress
    global max_checks_without_progress
    global best_model_params
    for step in range(total_epochs, total_epochs + epochs):
        current_time = time.time()
        n_batches = len(X_train) // batch_size
        num_aug = 8
        random_list = np.random.permutation(len(X_train))
        r_list = np.random.permutation(num_aug)
        for aug in range(num_aug):
            for iteration in range(n_batches):
                idx = random_list[iteration*batch_size:(iteration+1)*batch_size]
                X_batch = np.asarray([image_augmentation(X_train[idx[j]], r_list[aug]) for j in range(batch_size)])
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
        if val_cross_entropy < best_loss_val:
            best_loss_val = val_cross_entropy
            checks_since_last_progress = 0
            best_model_params = get_model_params(session)
        else: 
            checks_since_last_progress += 1
        dur = time.time() - current_time
        print('[*] Epoch:[%3d] Time: %.2fsec, Train Loss: %.2f, Val Loss: %.2f, Train Accuracy: %.2f%%, Val Accuracy: %.2f%%'
                    % (step + 1, dur, cross_entropy/4, val_cross_entropy, 
                       train_accuracy * 100 / X_train.shape[0] * batch_size, val_accuracy * 100 / X_val.shape[0] * batch_size))
        if checks_since_last_progress > max_checks_without_progress:
            print('[*] Early Stopping')
            break
    total_epochs += epochs
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
    
    
    
def train_model_cv(session, model, X, Y, epochs=1, batch_size=500):
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
            num_aug = 30
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
