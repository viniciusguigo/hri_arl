#!/usr/bin/env python
""" crash_plot.py:
Prepare data for the crash plot (check plan).
"""

__author__ = "Vinicius Guimaraes Goecks"
__version__ = "0.0.0"
__status__ = "Prototype"
__date__ = "September 18, 2017"

# import
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def return_crashes(data_file):
    """
    Just load file and plot
    """
    data = np.genfromtxt(data_file + '_crashes.csv', delimiter=',')

    return data

print('Loading files...')
root_files = ['/media/vinicius/vinicius_arl/data/iclr_prev_ddpg_0',
              '/media/vinicius/vinicius_arl/data/iclr_prev_ddpg_1_noclip',
              '/media/vinicius/vinicius_arl/data/iclr_prev_ddpg_2']#,
            #   '/media/vinicius/vinicius_arl/data/iclr_prev_ddpg_1_noclip']#,
            #   '/media/vinicius/vinicius_arl/data/iclr_prev_ddpg_2_noclip']#,
            #   '/media/vinicius/vinicius_arl/data/iclr_prev_imitation']

total_crashes = []
for data_file in root_files:
    data = return_crashes(data_file)
    total_crashes.append(data)

N = len(total_crashes)
x = np.arange(N)
width = 1/1.5
sns.set()

width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects = ax.bar(x, total_crashes, width)

# add some text for labels, title and axes ticks
ax.set_ylabel('# of Crashes')
ax.set_xticks(x)
ax.set_xticklabels(('Handcrafted', 'CyberSteer #1', 'CyberSteer #2'))

plt.show()
