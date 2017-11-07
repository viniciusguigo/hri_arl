#!/usr/bin/env python
""" neural.py:
Neural models to be used in RL.
"""

__author__ = "Vinicius Guimaraes Goecks"
__version__ = "0.0.0"
__status__ = "Prototype"
__date__ = "June 13, 2017"

# import
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # remove TF warning about CPU
import tensorflow as tf

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop, SGD, Adam
from keras.layers import Conv2D, MaxPooling2D, LSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping

import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

def plot_train_hist(run_id, history, plot_items):
    """
    Plot training history.
    """
    for item in plot_items:
        plt.figure()
        plt.xlabel('Epoch [unit]', fontsize='medium')
        if item == 'loss':
            plt.suptitle('Training Loss', fontsize='medium')
            plt.ylabel('Mean Squared Error [unit]', fontsize='medium')
            cut_factor = .0 # in %
        elif item == 'mean_absolute_error':
            plt.suptitle('Training Mean Absolute Error', fontsize='medium')
            plt.ylabel('Mean Absolute Error [unit]', fontsize='medium')
            cut_factor = .0 # in %
        elif item == 'acc':
            plt.suptitle('Training Accuracy', fontsize='medium')
            plt.ylabel('Accuracy [unit]', fontsize='medium')
            cut_factor = .0 # in %
        elif item == 'val_loss':
            plt.suptitle('Validation Loss', fontsize='medium')
            plt.ylabel('Mean Squared Error [unit]', fontsize='medium')
            cut_factor = .0 # in %
        elif item == 'val_acc':
            plt.suptitle('Validation Accuracy', fontsize='medium')
            plt.ylabel('Accuracy [unit]', fontsize='medium')
            cut_factor = .0 # in %
        elif item == 'val_mean_absolute_error':
            plt.suptitle('Validation Mean Absolute Error', fontsize='medium')
            plt.ylabel('Mean Absolute Error [unit]', fontsize='medium')
            cut_factor = .0 # in %
        plt.grid()

        # convert to numpy array and reshape as a column to save data
        hist_list = np.asarray(history.history[item])
        hist_list = hist_list.reshape(len(hist_list),1)
        steps = np.arange(len(hist_list)).reshape(len(hist_list),1)

        # save history
        save_plot = np.hstack((steps,hist_list))
        data_folder = '/media/vinicius/vinicius_arl/data/'
        np.savetxt(data_folder+run_id+'_plot_hist_'+item+'.csv', save_plot, delimiter=',')

        # remove beginning of loss (too high) and fix axis
        hist_init = int(cut_factor * len(hist_list))  # cut first x% of the history (for scaling purposes)
        plt.plot(steps[hist_init:], hist_list[hist_init:])
        plt.xlim([hist_init, len(hist_list)])

        plt.show()

def load_neural(name, loss, opt, metrics=['accuracy']):
    """
    Load pre-trained neural network.
    """
    folder_loc = '/media/vinicius/vinicius_arl/neural_models/'
    # load json and create model
    json_file = open(folder_loc + name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(folder_loc + name + ".h5")
    # loaded_model.compile(loss=loss, optimizer=opt)

    # load with metric, if have a validation_split
    loaded_model.compile(loss=loss, optimizer=opt, metrics=metrics)
    print("Loaded model from disk. Name = ", name)

    return loaded_model


def save_neural(model, name):
    """
    Save neural network (model) with a given name using JSON templates.
    """
    folder_loc = '/media/vinicius/vinicius_arl/neural_models/'
    # serialize model to JSON
    model_json = model.to_json()
    with open(folder_loc + name + ".json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(folder_loc + name + ".h5")

    print("Saved model to disk. Name = ", name)

class DeepNeuralNet(object):
    """
    Class to create and manage deep neural networks.

    Inputs
    ----------
    net_type: 'dense' for fully connected layers
    hidden_layers: number of hidden layers
    neurons: list of number of neurons [n_input, n_hidden1, ... , n_output]
    """
    def __init__(self, hidden_layers, neurons, loss='mse', optimizer='adam', dropout=0.):
        # get parameters
        self.hidden_layers = hidden_layers
        self.neurons = neurons
        self.loss = loss
        self.optimizer = optimizer
        self.dropout = dropout

        # create model
        self.model = self.create_model()

    def create_model(self):
        model = Sequential()

        # define input layer and first hidden
        input_dim = self.neurons[0]
        output_dim = self.neurons[-1]
        hidden1 = self.neurons[1]
        model.add(Dense(hidden1, input_dim=input_dim, init='lecun_uniform', activation='relu'))
        if self.dropout != 0:
            model.add(Dropout(self.dropout))

        # hidden layers
        for i in range(2,self.hidden_layers+1):
            model.add(Dense(self.neurons[i], init='lecun_uniform', activation='relu'))
            if self.dropout != 0:
                model.add(Dropout(self.dropout))

        # output layer and compilation
        model.add(Dense(output_dim, init='lecun_uniform', activation='linear'))
        opt = Adam()
        model.compile(loss=self.loss, optimizer=opt)

        return model


    def copy_weights(self, source, dest):
        dest.set_weights(source.get_weights())
        return dest


class ImitationNetwork(object):
    """
    Conv net written and taylored to do imitation learning based on AirSim
    human expert data.
    It process camera images and outputs actions to the drone.
    Updated to work with multiple actions.
    """
    def __init__(self, n_act=1, mode=None):
        # depth data resolution
        self.width = 64
        self.height = 36

        # for lstm only
        self.n_timesteps = 1
        self.lahead = 1

        # create model
        self.batch_size = 32
        self.n_act = n_act

        if mode == 'dqn':
            self.create_dqn()
        if mode == 'cs_classify':
            # cybersteer version 1: classify actions between human and not human
            self.create_cs_classify()

        if mode == 'cs_imitation':
            # cybersteer version 2: create imitation learning to suggest actions
            self.create_cs_imitation()

        else:
            self.create_model()
            # self.create_model_conv_lstm()

    def create_cs_classify(self):
        """
        Create neural network model for CyberSteer (1): classify actions between
        humans and nonhumans.
        """
        self.name = 'cs_classify'

        model = Sequential()

        model.add(Conv2D(32, (3, 3), input_shape=(self.height, self.width, 1)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))


        # the model so far outputs 3D feature maps (height, width, features)

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

        model.add(Dense(64))
        model.add(Activation('relu'))

        model.add(Dense(self.n_act))
        model.add(Activation('linear'))

        model.compile(loss='mse',
                      optimizer='adam',
                      metrics=['accuracy'])

        # save to class
        print(model.summary())
        plot_model(model, to_file='model.png')
        self.model = model

    def create_cs_imitation(self):
        """
        Create neural network model for CyberSteer (2): imitate human actions and
        compare to learning algorithm.
        """
        self.name = 'cs_imitation'

        model = Sequential()

        model.add(Conv2D(32, (3, 3), input_shape=(self.height, self.width, 1)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))


        # the model so far outputs 3D feature maps (height, width, features)

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

        model.add(Dense(64))
        model.add(Activation('relu'))

        model.add(Dense(self.n_act))
        model.add(Activation('linear'))

        model.compile(loss='mse',
                      optimizer='adam',
                      metrics=['accuracy'])

        # save to class
        print(model.summary())
        plot_model(model, to_file='model.png')
        self.model = model

    def create_dqn(self):
        """
        Create neural network model for DQN.
        """
        self.name = 'dqn_conv'

        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(self.height, self.width, 1)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # the model so far outputs 3D feature maps (height, width, features)

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(3)) # right now hardcoded to -1, 0, 1
        model.add(Activation('linear'))

        model.compile(loss='mse',
                      optimizer='adam',
                      metrics=['accuracy'])

        # save to class
        print(model.summary())
        plot_model(model, to_file='model.png')
        self.model = model

    def create_model(self):
        """
        Create neural network model.
        """
        self.name = 'conv'

        model = Sequential()

        model.add(Conv2D(32, (3, 3), input_shape=(self.height, self.width, 1)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))


        # the model so far outputs 3D feature maps (height, width, features)

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

        model.add(Dense(64))
        model.add(Activation('relu'))

        model.add(Dense(self.n_act))
        model.add(Activation('linear'))

        model.compile(loss='mse',
                      optimizer='adam',
                      metrics=['accuracy'])

        # save to class
        print(model.summary())
        plot_model(model, to_file='model.png')
        self.model = model

    def create_model_conv_lstm(self):
        """
        Create neural network model.
        TESTING: Convolutional + LSTM Layers
        """
        self.name = 'conv_lstm'

        model = Sequential()
        model.add(Conv2D(32, (5, 5), input_shape=(self.height, self.width, 1)))
        model.add(Activation('elu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('elu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # the model so far outputs 3D feature maps (height, width, features)

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dropout(.2))

        model.add(LSTM(128,
                       batch_input_shape=(self.batch_size, self.n_timesteps, self.height*self.width, 1),
                       return_sequences=False,
                       stateful=True))

        model.add(Dense(128))
        model.add(Activation('elu'))
        model.add(Dropout(0.25))
        model.add(Dense(self.n_act))
        model.add(Activation('linear'))

        model.compile(loss='mse',
                      optimizer='adam',
                      metrics=['accuracy'])

        # save to class
        print(model.summary())
        plot_model(model, to_file='model_conv_lstm.png')
        self.model = model

    def create_model_conv_lstm2(self):
        """
        Create neural network model.
        TESTING: Convolutional + LSTM Layers (many to one)
        Ref.: https://github.com/fchollet/keras/issues/5338
        """
        self.name = 'conv_lstm'

        model = Sequential()
        model.add(TimeDistributed(
            Conv2D(32, (3, 3)), input_shape=(self.n_timesteps, self.height, self.width, 1) ))
        model.add(TimeDistributed(Activation('relu')))
        model.add(BatchNormalization())
        model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

        model.add(TimeDistributed(Conv2D(64, (3, 3))))
        model.add(TimeDistributed(Activation('relu')))
        model.add(BatchNormalization())
        model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

        model.add(TimeDistributed(Conv2D(128, (3, 3))))
        model.add(TimeDistributed(Activation('relu')))
        model.add(BatchNormalization())
        model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

        # the model so far outputs 3D feature maps (height, width, features)

        model.add(TimeDistributed(Flatten()))  # this converts our 3D feature maps to 1D feature vectors

        model.add( LSTM (5, return_sequences = True, stateful = False ))
        model.add(Dense(self.n_act))
        model.add(Activation('linear'))

        model.compile(loss='mse',
                      optimizer='adam',
                      metrics=['accuracy'])

        # save to class
        print(model.summary())
        plot_model(model, to_file='model_conv_lstm2.png')
        self.model = model

    def create_model_conv_lstm2(self):
        """
        Create neural network model.
        TESTING: Convolutional + LSTM Layers (many to one)
        Ref.: https://github.com/fchollet/keras/issues/5338
        """
        self.name = 'conv_lstm'

        model = Sequential()
        model.add(TimeDistributed(
            Conv2D(32, (3, 3)), input_shape=(self.n_timesteps, self.height, self.width, 1) ))
        model.add(TimeDistributed(Activation('relu')))
        model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

        model.add(TimeDistributed(Conv2D(64, (3, 3))))
        model.add(TimeDistributed(Activation('relu')))
        model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

        model.add(TimeDistributed(Conv2D(128, (3, 3))))
        model.add(TimeDistributed(Activation('relu')))
        model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

        # the model so far outputs 3D feature maps (height, width, features)

        model.add(TimeDistributed(Flatten()))  # this converts our 3D feature maps to 1D feature vectors

        model.add( LSTM (5, return_sequences = True, stateful = False ))
        model.add(Dense(self.n_act))
        model.add(Activation('linear'))

        model.compile(loss='mse',
                      optimizer='adam',
                      metrics=['accuracy'])

        # save to class
        print(model.summary())
        plot_model(model, to_file='model_conv_lstm2.png')
        self.model = model

    def reshape_lstm(self, x, y):
        """
        Convert data into sequence on inputs for LSTM.
        """
        # old shape
        print('Old shapes for x and y:')
        print(x.shape)
        print(y.shape)

        # reshape in sequences
        new_x = []
        new_y = []

        for i in range(x.shape[0]-self.n_timesteps-1):
            x_before = x[i:(i+self.n_timesteps), :].flatten()
            new_x.append(x_before)

            y_before = y[i:(i+self.n_timesteps), :].flatten()
            new_y.append(y_before)

        # convert to numpy
        x = np.asarray(new_x)
        y = np.asarray(new_y)

        print('New shapes for x and y:')
        print(x.shape)
        print(y.shape)

        return x, y


    def train_model(self, run_id, n_items, n_act = 1, show_example=False):
        """
        Train created model.
        """
        # load model
        model = self.model

        # load files
        print('Loading files ...')
        root_file = '../data/'+run_id+'_imit_'
        data = np.genfromtxt(root_file+'0.csv', delimiter=',')

        for i in range (1,n_items):
            # load each other file
            temp_data = np.genfromtxt(root_file + str(i) +'.csv', delimiter=',')

            # apprend temp_data to original dataset
            data = np.vstack((data,temp_data))

        # split states and actions
        x = data[:,:-n_act]
        y = data[:,-n_act:]

        # capture oroiginal size of features and samples
        samples = x.shape[0]
        features = x.shape[1]
        print('Number of samples: ', samples)
        print('x features: ', features)
        print('y features: ', y.shape[1])

        # reshape to sequence, if lstm
        if self.name == 'conv_lstm':
            x, y = self.reshape_lstm(x,y)

            # reshape arrays to 2d images
            # samples, height, width, channels
            new_samples = samples-(self.n_timesteps+1)
            print('New sample size: ', new_samples)

            x = x.reshape((new_samples, self.n_timesteps, self.height, self.width, 1))
            y = y.reshape((new_samples, self.n_timesteps, n_act))

        else:
            # FOR NON-LSTM CASES
            # reshape arrays to 2d images
            # samples, height, width, channels
            x = x.reshape((samples, self.height, self.width, 1))
            y = y.reshape((samples, n_act))

        # store processed data on class
        self.x = x
        self.y = y

        if show_example:
            # example image
            n = np.random.randint(samples)
            # added interpolation for smoothness
            imgplot = plt.imshow(x[n,:,:,0], interpolation="bicubic")
            plt.title('Example Image')
            plt.tight_layout()
            plt.show()
            print('Sample y data: ', y[n,:])

        # define any desired callbacks (EarlyStopping, ModelCheckpoint, etc)
        # early_stop = EarlyStopping(monitor='val_loss', min_delta=0.01, verbose=1, mode='auto')

        # train
        hist_train = model.fit(x, y, batch_size=self.batch_size,
                         epochs=200,
                         shuffle=False,
                         validation_split=.2,
                         verbose=2)#,
                        #  callbacks=[early_stop])

        # plot train history
        save_neural(self.model, run_id)
        self.plot_train_hist(run_id, hist_train)


    def model_predict(self, img_input):
        """
        Predict action based on image input. Returns action.
        """
        # get number of samples
        samples = img_input.shape[0]

        # reshape input to fit on keras
        if self.name == 'conv_lstm':
            print(img_input[0,:,:,:].shape)
            img_input = img_input[0,:,:,:].reshape((1, self.n_timesteps, self.height, self.width, 1))
        else:
            img_input = img_input.reshape((samples, self.height, self.width, 1))
        act = self.model.predict(img_input, batch_size=1, verbose=2)
        print('Action shape: ', act.shape)
        print('Last action: ', act[-1,:])

        return act


    def plot_train_hist(self, run_id, history):
        """
        Plot training history.
        """
        data_folder = '../data/'
        plot_items = ['loss','acc','val_loss','val_acc']

        for item in plot_items:
            plt.figure()
            plt.xlabel('Epoch [unit]', fontsize='medium')
            if item == 'loss':
                plt.suptitle('Training Loss', fontsize='medium')
                plt.ylabel('Mean Squared Error [unit]', fontsize='medium')
                cut_factor = .1 # in %
            elif item == 'acc':
                plt.suptitle('Training Accuracy', fontsize='medium')
                plt.ylabel('Accuracy [unit]', fontsize='medium')
                cut_factor = .0 # in %
            elif item == 'val_loss':
                plt.suptitle('Validation Loss', fontsize='medium')
                plt.ylabel('Mean Squared Error [unit]', fontsize='medium')
                cut_factor = .1 # in %
            elif item == 'val_acc':
                plt.suptitle('Validation Accuracy', fontsize='medium')
                plt.ylabel('Accuracy [unit]', fontsize='medium')
                cut_factor = .0 # in %
            plt.grid()

            # convert to numpy array and reshape as a column to save data
            hist_list = np.asarray(history.history[item])
            hist_list = hist_list.reshape(len(hist_list),1)
            steps = np.arange(len(hist_list)).reshape(len(hist_list),1)

            # save plot data
            save_plot = np.hstack((steps,hist_list))
            np.savetxt(data_folder+run_id+'_plot_hist_'+item+'.csv', save_plot, delimiter=',')

            # remove beginning of loss (too high) and fix axis
            hist_init = int(cut_factor * len(hist_list))  # cut first x% of the history (for scaling purposes)
            plt.plot(steps[hist_init:], hist_list[hist_init:])
            plt.xlim([hist_init, len(hist_list)])

            plt.show()

if __name__ == '__main__':
    # test training
    run_id = 'test_human5'
    n_items = 100
    n_act = 2

    net = ImitationNetwork(n_act=n_act)
    net.train_model(run_id,n_items,n_act)

    # test predicting
    print(net.model_predict(net.x))
