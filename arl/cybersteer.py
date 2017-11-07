#!/usr/bin/env python
""" cybersteer.py:
Testing ideas for CyberSteer.
"""

__author__ = "Vinicius Guimaraes Goecks"
__version__ = "0.0.0"
__status__ = "Prototype"
__date__ = "August 22, 2017"

# import
import numpy as np
import cv2
import tensorflow as tf

from keras.layers import Input, Dense, Conv2D, Flatten, LSTM, MaxPooling2D, concatenate
from keras.models import Model

from neural import plot_train_hist, save_neural

tf.logging.set_verbosity(tf.logging.INFO)

def next_batch(num, data, labels, extra):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
    extra_shuffle = [extra[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle), np.asarray(extra_shuffle)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def restore_and_predict():
    """
    Restore previously trained model and execute predictions.
    """
    with tf.Session() as sess:
        model_saver = tf.train.import_meta_graph(model_save_folder + '/my-model.meta')
        model_saver.restore(sess, model_save_folder + '/my-model')
        x = tf.get_collection("placeholder")[0]
        output = tf.get_collection("output")[0] #output will be the tensor for model's last layer
        print("Model restored.")
        print('Initialized')
        #print(sess.run(tf.get_default_graph().get_tensor_by_name('w_conv1:0')))

        #collect list of preprocessed data on submission set
        inputData = []
        with open('stage1_sample_submission.csv') as f:
            reader = csv.reader(f)
            num = 0

            for row in reader:
                if num > 0:
                    patient = row[0]
                    #print(patient)
                    inputData.append(process_data(patient, img_px_size=IMG_SIZE_PX, hm_slices=SLICE_COUNT))
                num += 1

        #prediction!
        prediction = sess.run(output, feed_dict={x: inputData})
        print(prediction)


def load_and_stack(root_file, n_items, n_act, n_frames):
    '''
    Load each experience file and stack them together.
    '''
    def stack_and_split(data, n_act, n_frames):
        '''
        Grab a sequence of frames and stack them together.
        Only uses the last frame's action,
        '''
        # split images and actions
        images = data[:,:-n_act]
        actions = data[n_frames-1:,-n_act:] # also removed appended empty actions
        samples = data.shape[0] - n_frames + 1 # remove index for empty frames

        images_stacked = np.zeros((samples,image_height,image_width,n_frames)) # samples, height, width, channels

        # loop for every image and stack them
        for i in range(samples):
            for j in range(n_frames):
                images_stacked[i,:,:,j] = images[i+j,:].reshape(image_height,image_width)

        return images_stacked, actions

    # load first file
    image_width = 64
    image_height = 36
    data_name = root_file+'0.csv'
    print('Loading '+ data_name)
    data = np.genfromtxt(data_name, delimiter=',')

    # add empty frames so initial image is the first when stack
    n_empty = n_frames - 1 # initial image
    empty_frames = np.zeros((n_empty,data.shape[1]))

    # add empty frames
    data = np.vstack((empty_frames,data))

    # stack frames and split actions
    images_stacked, actions = stack_and_split(data, n_act, n_frames)

    # do the same for the rest of the files
    for i in range (1,n_items):
        # load each other file
        data_name = root_file + str(i) +'.csv'
        print('Loading '+ data_name)
        temp_data = np.genfromtxt(data_name, delimiter=',')

        # add empty frames
        temp_data = np.vstack((empty_frames,temp_data))

        # stack frames and split actions
        temp_images_stacked, temp_actions = stack_and_split(temp_data, n_act, n_frames)

        # apprend temp_data to original dataset
        images_stacked = np.vstack((images_stacked,temp_images_stacked))
        actions = np.vstack((actions,temp_actions))

    cv2.imwrite('test0.png', 255*images_stacked[5,:,:,0])
    cv2.imwrite('test1.png', 255*images_stacked[5,:,:,1])
    cv2.imwrite('test2.png', 255*images_stacked[5,:,:,2])

    return images_stacked, actions


def idea1():
    """
    Testing implementation of idea1, explained below. Small case.

    Idea 1: classify actions between human and not human
    Steps:
    - get human data
    - get random data
    - join dataset
    - train network (output: confidence of the action being taken by human)
    - compute estimated reward

    """
    # load human dataset
    print('Loading files...')
    root_file_human = '/media/vinicius/vinicius_arl/data/test_human5_imit_'
    root_file_random = '/media/vinicius/vinicius_arl/data/test_random5_imit_'
    n_items = 100 # use 100, 5 for test only
    n_act = 2
    n_frames = 3

    # load data stacking frames of images
    human_images, human_actions = load_and_stack(root_file_human, n_items, n_act, n_frames)
    print('Human images: ', human_images.shape)
    print('Human actions: ', human_actions.shape)

    random_images, random_actions = load_and_stack(root_file_random, n_items, n_act, n_frames)
    print('Random images: ', random_images.shape)
    print('Random actions: ', random_actions.shape)

    # create labels
    label_human = np.ones((human_images.shape[0],1))
    label_random = np.zeros((random_images.shape[0],1))

    # join datasets
    total_images = np.vstack((human_images,random_images))
    total_actions = np.vstack((human_actions, random_actions))
    label_total = np.vstack((label_human, label_random))

    # split into train and test dataset
    n = .2 # percent to test
    total_images_train = total_images[0:int(total_images.shape[0]*(1-n)),:]
    total_actions_train = total_actions[0:int(total_actions.shape[0]*(1-n)),:]
    label_total_train = label_total[0:int(label_total.shape[0]*(1-n)),:]

    total_images_test = total_images[int(total_images.shape[0]*(1-n)):-1,:]
    total_actions_test = total_actions[int(total_actions.shape[0]*(1-n)):-1,:]
    label_total_test = label_total[int(label_total.shape[0]*(1-n)):-1,:]

    # CREATE GRAPH
    # input image and output label
    x = tf.placeholder(tf.float32, shape=[None, 36, 64, 3], name='x')
    act = tf.placeholder(tf.float32, shape=[None, 2], name='act')
    y_ = tf.placeholder(tf.float32, shape=[None, 1], name='y_')

    # first conv layer
    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # second layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # fc layer
    W_fc1 = weight_variable([9*16*64, 256])
    b_fc1 = bias_variable([256])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 9*16*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # connect actions here (use temp layers)
    # temp2: from the action input
    W_fc_temp = weight_variable([n_act, 256])

    # connect to temp layers in one fc layer (fc_temp)
    b_fc_temp = bias_variable([256])
    h_fc_temp = tf.nn.relu(h_fc1 + tf.matmul(act, W_fc_temp) + b_fc_temp)

    # dropout
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    h_fc1_drop = tf.nn.dropout(h_fc_temp, keep_prob)

    # output
    W_fc2 = weight_variable([256, 1])
    b_fc2 = bias_variable([1])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    prediction = tf.sigmoid(y_conv, name='prediction')

    # train and evaluate
    cross_entropy = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y_conv))

    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

    # correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    predicted_class = tf.greater(prediction,0.5)
    correct = tf.equal(predicted_class, tf.equal(y_,1.0))
    accuracy = tf.reduce_mean( tf.cast(correct, 'float') )

    # same model
    saver = tf.train.Saver()

    # fix memory issues
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    # SETUP SUMMARY
    logs_path = '/tmp/tensorflow_logs/example/'
    tf.summary.scalar("cross_entropy", cross_entropy)
    tf.summary.scalar("accuracy", accuracy)
    merged_summary_op = tf.summary.merge_all()

    # RUN GRAPH
    n_epochs = 50000
    with tf.Session(config=config) as sess:

        # initialize variables
        sess.run(tf.global_variables_initializer())

        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        for i in range(n_epochs):
            batch = next_batch(5, total_images_train, label_total_train, total_actions_train)
            # print('images: \n',batch[0])
            # print('act: \n',batch[2])
            # print('label: \n',batch[1])
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0],
                                                          act: batch[2],
                                                          y_: batch[1],
                                                          keep_prob: 1.0})
                print('step %d/%d, training accuracy %g ' % (i, n_epochs, train_accuracy))

            if i % 2000 == 0:
                saver.save(sess, './test_tf/model_partial.ckpt')

            _, summary = sess.run([train_step, merged_summary_op], feed_dict={x: batch[0],
                                                                              act: batch[2],
                                                                              y_: batch[1],
                                                                              keep_prob: 0.2})

            # Write logs at every iteration
            summary_writer.add_summary(summary, i)

        print('test accuracy %g' % accuracy.eval(feed_dict={
          x: total_images_test, act: total_actions_test, y_: label_total_test, keep_prob: 0.2}))

        saver.save(sess, './test_tf/model.ckpt')
        print('saved final model')

        print("Run the command line:\n" \
            "--> tensorboard --logdir="+logs_path +
            "\nThen open http://0.0.0.0:6006/ into your web browser")

        return batch

def load_idea1(batch):
    """
    Load trained model and test a few predictions.
    """
    saver = tf.train.import_meta_graph("./test_tf/model.ckpt.meta")

    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, "./test_tf/model.ckpt")
        print("Model restored.")

        # load previous input placeholders
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        act = graph.get_tensor_by_name("act:0")
        y_ = graph.get_tensor_by_name("y_:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")

        # restore operation
        prediction = graph.get_tensor_by_name("prediction:0")

        # do a prediction
        print(sess.run(prediction,feed_dict={x: batch[0],
                                             act: batch[2],
                                             y_: batch[1],
                                             keep_prob: 1.0}))


def idea2():
    """
    Testing implementation of idea2, explained below. Small case.

    Idea 2: create imitation learning to suggest actions
    Steps:
    - get human data
    - train network (output: action)
    - combine with deep rl to compute both actions in a separated thread
    - compute estimated reward

    """
    # load data
    print('Loading files...')
    root_file = '/media/vinicius/vinicius_arl/data/test_human5_imit_'
    n_items = 100 # use 100, 5 for test only
    n_act = 2
    n_frames = 3

    # load data stacking frames of images
    images, actions = load_and_stack(root_file, n_items, n_act, n_frames)
    print(images.shape)
    print(actions.shape)

    # define inputs (depth sensor image)
    inputs = Input(shape=(36,64,n_frames))

    # a layer instance is callable on a tensor, and returns a tensor
    x = Conv2D(32, (3,3), padding='same', activation='relu')(inputs)
    x = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)

    x = Conv2D(16, (3,3), padding='same', activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)

    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    # x = LSTM(8)(x)
    predictions = Dense(n_act, activation='tanh')(x)

    # This creates a model that includes
    # the Input layer and three layers
    model = Model(inputs=inputs, outputs=predictions)

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])

    hist = model.fit(images, actions, epochs=50, batch_size=32, verbose=2, validation_split=.2)

    pred = model.predict(images[-10:,:,:,:].reshape(10,36,64,3))
    real = actions[-10:,:]
    print(pred-real)

    save_neural(model,'cybersteer_2')
    plot_items = ['loss','mean_absolute_error','val_loss','val_mean_absolute_error']
    plot_train_hist('cybersteer_2', hist, plot_items)

def idea1_keras():
    """
    Testing implementation of idea1, explained below. Small case.
    Using Keras instead of Tensorflow.

    Idea 1: classify actions between human and not human
    Steps:
    - get human data
    - get random data
    - join dataset
    - train network (output: confidence of the action being taken by human)
    - compute estimated reward

    """
    # load human dataset
    print('Loading files...')
    root_file_human = '/media/vinicius/vinicius_arl/data/test_human5_imit_'
    root_file_random = '/media/vinicius/vinicius_arl/data/test_random5_imit_'
    n_items = 100 # use 100, 5 for test only
    n_act = 2
    n_frames = 3

    # load data stacking frames of images
    human_images, human_actions = load_and_stack(root_file_human, n_items, n_act, n_frames)
    print('Human images: ', human_images.shape)
    print('Human actions: ', human_actions.shape)

    random_images, random_actions = load_and_stack(root_file_random, n_items, n_act, n_frames)
    print('Random images: ', random_images.shape)
    print('Random actions: ', random_actions.shape)

    # create labels
    label_human = np.ones((human_images.shape[0],1))
    label_random = np.zeros((random_images.shape[0],1))

    # join datasets
    total_images = np.vstack((human_images,random_images))
    total_actions = np.vstack((human_actions, random_actions))
    label_total = np.vstack((label_human, label_random))

    # split into train and test dataset
    n = .2 # percent to test
    total_images_train = total_images[0:int(total_images.shape[0]*(1-n)),:]
    total_actions_train = total_actions[0:int(total_actions.shape[0]*(1-n)),:]
    label_total_train = label_total[0:int(label_total.shape[0]*(1-n)),:]

    total_images_test = total_images[int(total_images.shape[0]*(1-n)):-1,:]
    total_actions_test = total_actions[int(total_actions.shape[0]*(1-n)):-1,:]
    label_total_test = label_total[int(label_total.shape[0]*(1-n)):-1,:]

    # main_input = images
    # auxiliary_input = actions
    # main_output = label (human or not human)
    main_input = Input(shape=(36,64,n_frames), dtype='float32', name='main_input')

    x = Conv2D(32, (3,3), padding='same', activation='relu')(main_input)
    x = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)

    x = Conv2D(16, (3,3), padding='same', activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)

    conv_out = Flatten()(x)

    auxiliary_input = Input(shape=(2,), name='aux_input')
    x = concatenate([conv_out, auxiliary_input])

    # We stack a deep densely-connected network on top
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    # And finally we add the main logistic regression layer
    main_output = Dense(1, activation='sigmoid', name='main_output')(x)

    # define model with 2 inputs
    model = Model(inputs=[main_input, auxiliary_input], outputs=main_output)

    model.compile(optimizer='adam',
              loss={'main_output': 'binary_crossentropy'},
              metrics=['accuracy'])

    # And trained it via:
    hist = model.fit({'main_input': total_images_train, 'aux_input': total_actions_train},
              {'main_output': label_total_train},
              epochs=50, batch_size=32, verbose=2,
              validation_data=({'main_input': total_images_test,
                                'aux_input': total_actions_test},
                                {'main_output': label_total_test}))

    save_neural(model,'cybersteer_1')

    plot_items = ['loss','acc','val_loss','val_acc']
    plot_train_hist('cybersteer_1', hist, plot_items)


if __name__ == '__main__':
    # batch = idea1()
    # load_idea1(batch)
    idea1_keras()
    # idea2()
