#!/usr/bin/env python
""" plot_position_traj.py:
Plot vehicle position over time.
"""

__author__ = "Vinicius Guimaraes Goecks"
__version__ = "0.0.0"
__status__ = "Prototype"
__date__ = "August 23, 2017"

# import
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main(color):
    '''
    Load csv file an dplot position.
    '''
    # load original dataset
    data = np.genfromtxt(data_addr + data_name + '.csv', delimiter=',')

    # fix cases with only one line of data
    if data.shape[0] == 10:
        data = data.reshape(1,10)

    # get trajectories
    x = data[:,1]
    y = data[:,2]

    # plot
    plt.plot(x[-1],y[-1],'x',color=color)
    plt.plot(x,y,'--',color=color)

    # fix axis to size of map
    plt.xlim([-10,100])
    # plt.ylim([-16,16])




if __name__ == '__main__':
    data_addr = '../../data/'
    n_traj = 100

    plt.figure()
    for i in range(n_traj):
        data_name = 'test_att_random_att_' + str(i)
        main(color='b')

    for i in range(n_traj):
        data_name = 'test_att_ddpg_att_' + str(i)
        main(color='r')

    for i in range(n_traj):
        data_name = 'test_att_ddpg_small_att_' + str(i)
        main(color='g')

    plt.axvline(x=90,color='k')
    plt.axvline(x=-2,color='k')
    plt.grid()
    plt.show()
