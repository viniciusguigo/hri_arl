#!/usr/bin/env python
""" performance_plot.py:
Prepare data for the performance plot (check plan).
"""

__author__ = "Vinicius Guimaraes Goecks"
__version__ = "0.0.0"
__status__ = "Prototype"
__date__ = "September 18, 2017"

# import
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# plan
#    load position data of one episode
#    get last value e save [epi, distance]
#    repeat for every episode

def get_distance(data_file, n_epi, avg_ratio):
    """
    Get last distance of each episode and plot.
    """
    # distance per episode
    dist_epi = np.zeros((n_epi,2))

    # do the same for the rest of the files
    for i in range (n_epi):
        # load each other file
        data_name = data_file + '_att_' + str(i) +'.csv'
        data = np.genfromtxt(data_name, delimiter=',')
        dist_epi[i,:] = np.array([i,data[-1,1]])

    # average scores
    n_epi_avg = int(n_epi/avg_ratio)
    dist_epi_avg = np.zeros((n_epi_avg,2))

    for j in range (n_epi_avg):
        computed_avg = np.mean(dist_epi[j*avg_ratio:(j+1)*avg_ratio,1])
        dist_epi_avg[j,:] = np.array([j,computed_avg])

    # label and plot
    if count == 0:
        label = 'Handcrafted'
    elif count == 1:
        label = 'CyberSteer #1'
    elif count == 2:
        label = 'CyberSteer #2'

    sns.set()
    plt.plot(dist_epi_avg[:,0],dist_epi_avg[:,1]/max_dist,label=label)

print('Loading files...')
n_epi = 1000
max_dist = 90
avg_ratio = 10

root_files = ['/media/vinicius/vinicius_arl/data/ddpg_big_0_v0',
              '/media/vinicius/vinicius_arl/data/ddpg_big_1_v0',
              '/media/vinicius/vinicius_arl/data/ddpg_big_2_v0']


count = 0
for data_file in root_files:
    get_distance(data_file, n_epi, avg_ratio)
    count += 1


plt.legend()
plt.xlabel('Episodes x'+str(avg_ratio)+' [unit]')
plt.ylabel('Task Completion [unit]')
plt.savefig('ddpg_big_performance_plot.png', transparent = False)
# plt.show()
