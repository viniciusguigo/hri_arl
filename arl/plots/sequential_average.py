#!/usr/bin/env python
""" sequential_average.py:
Load file and compute average in a sequence of steps, instead of computing it
from multiple runs.
"""

__author__ = "Vinicius Guimaraes Goecks"
__version__ = "0.0.0"
__status__ = "Prototype"
__date__ = "August 23, 2017"

# import
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    '''
    Load csv file with data and split single column of data in n columns so it
    can be averaged.
    '''
    # load original dataset
    data = np.genfromtxt(data_addr + data_name + '.csv', delimiter=',')
    samples = data.shape[0]

    new_samples = int(samples/reduce_factor)
    new_data = np.zeros((new_samples,2))

    past_idx = 0

    for i in range(new_samples):
        # compute average of sequence of steps and append to new data
        sequential_avg = np.average(data[past_idx:(i+1)*reduce_factor,1])
        new_data[i,:] = [i,sequential_avg]

        # update idx
        past_idx = (i+1)*reduce_factor

    # save file
    np.savetxt(data_addr + data_name + '_average'+str(reduce_factor)+'.csv', new_data, delimiter=',')

if __name__ == '__main__':
    data_addr = '../../data/'
    data_name = 'test_random5_rew'
    reduce_factor = 10
    main()
