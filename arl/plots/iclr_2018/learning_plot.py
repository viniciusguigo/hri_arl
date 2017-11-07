#!/usr/bin/env python
""" learning_plot.py:
Prepare data for the learning plot (check plan).
"""

__author__ = "Vinicius Guimaraes Goecks"
__version__ = "0.0.0"
__status__ = "Prototype"
__date__ = "September 18, 2017"

# import
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_plot(data_file, avg_ratio):
    """
    Just load file and plot
    """
    data = np.genfromtxt(data_file, delimiter=',')

    # average scores
    n_epi_avg = int(data.shape[0]/avg_ratio)
    data_avg = np.zeros((n_epi_avg,2))

    for j in range (n_epi_avg):
        computed_avg = np.mean(data[j*avg_ratio:(j+1)*avg_ratio,1])
        data_avg[j,:] = np.array([j,computed_avg])


    # label and plot
    if count == 0:
        label = 'Handcrafted'
        gain = 1/(90*20+100) # what multiples the changes in distance
    elif count == 1:
        label = 'CyberSteer #1'
        gain = 1/(500*10+100) # value of r_max
    elif count == 2:
        label = 'CyberSteer #2'
        gain = 1/(500*10+100) # value of r_max

    sns.set()
    plt.plot(data_avg[:,0],gain*data_avg[:,1],label=label)
    plt.ylim([-.1,human_baseline + .1])

# plan
#    load data from iclr_prev_ddpg_0
#    load data from iclr_prev_ddpg_1
#    load data from iclr_prev_ddpg_2
#
#    plot and see how it looks like
#    possibly need to average steps and normalize rewards

print('Loading files...')
avg_ratio = 5
human_baseline = 1

root_files = ['/media/vinicius/vinicius_arl/data/ddpg_big_0_v0_rew.csv',
              '/media/vinicius/vinicius_arl/data/ddpg_big_1_v0_rew.csv',
              '/media/vinicius/vinicius_arl/data/ddpg_big_2_v0_rew.csv']


count = 0
for data_file in root_files:
    load_plot(data_file, avg_ratio)
    count += 1

# add human baseline
plt.axhline(y=human_baseline, linestyle='dashed', color='black', label='Human Intention')

plt.legend()
plt.xlabel('Episodes x'+str(avg_ratio)+' [unit]')
plt.ylabel('Total Episode Reward [unit]')
plt.savefig('ddpg_big_learning_plot.png', transparent = False)
# plt.show()
