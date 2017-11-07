#!/usr/bin/env python
""" example_main.py
Testing the use of ini files to manage the whole experiment.
"""

__author__ = "Vinicius Guimaraes Goecks"
__version__ = "0.0.0"
__status__ = "Prototype"
__date__ = "August 30, 2017"

# import
import numpy as np
import gym
import time
import sys

import argparse
from gym import spaces

import tensorflow as tf

from PythonClient import *

from support import CustomAirSim, train_model_multi, train_model, train_deep_rl, train_human, AirSimEOM, train_deep_rl_cv, train_human_cv
from learn import RandomAgent, HumanAgent, ImitationAgent, InterventionAgent, RandomAgent_AirSim
from learn import InterventionAgentMulti, DQN_AirSim, DDPG_AirSim, HumanAgentXBox, HumanAgentXBoxMulti

import configparser

def main(config):
    """
    Reads configuration file to setup experiment.
    """
    main_setup = config['DEFAULT']

    # parameters and modes
    n_episodes = main_setup.getint('n_episodes')
    n_steps = main_setup.getint('n_steps')

    # test a few iterations
    run_id =  main_setup['run_id']

    # initial setup
    use_gui = main_setup.getint('use_gui')
    if use_gui:
        print('[*] Using UAV GUI')

    mode = main_setup['mode']
    if mode == 'cv':
        print('[*] mode = Computer Vision')
        drone = AirSimEOM(n_steps,
                          use_gui=use_gui,
                          reward_signal=main_setup.getint('reward_signal'),
                          clip_action=main_setup.getint('clip_action'))
    elif mode == 'px4':
        print('[*] mode = Pixhawk')
        drone = CustomAirSim(n_steps, use_gui=use_gui)

    # start learning agents
    select_agent = main_setup['agent']
    print('AGENT: {}'.format(select_agent))

    if select_agent == 'human' and mode == 'cv':
        agent = HumanAgentXBoxMulti(drone, n_episodes)
        print('Using human to get data on computer vision mode')
        train_human_cv(run_id, drone, agent, n_episodes, n_steps)

    if select_agent == 'human' and mode == 'px4':
        agent = HumanAgentXBoxMulti(drone, n_episodes)
        print('Training human and getting data from Pixhawk.')
        train_human(run_id, drone, agent, n_episodes, n_steps)

    elif select_agent == 'imitation':
        print('***** NOT SUPPORTED ANYMORE (SEP 05, 2017 *****')
        agent = ImitationAgent(drone, n_episodes)

        print('Imitation Learning with Convolutional and Recurrent layers.')
        train_human_cv(run_id, drone, agent, n_episodes, n_steps)
        # train_model_multi(run_id, drone, agent, n_episodes, n_steps)

    elif select_agent == 'interv':
        print('***** NOT SUPPORTED ANYMORE (SEP 05, 2017 *****')

        agent = InterventionAgentMulti(drone, n_episodes)

    elif select_agent == 'random':
        print('***** NOT SUPPORTED ANYMORE (SEP 05, 2017 *****')

        agent = RandomAgent_AirSim(drone, n_episodes)

    elif select_agent == 'dqn':
        print('***** NOT SUPPORTED ANYMORE (SEP 05, 2017 *****')

        # parameters
        gamma = 0.9
        eps_max = 1
        eps_min = 1
        lbd = 0.001
        batch_size = 64
        buffer_size = 10000
        target_update_freq = 100
        dqn_agent = main_setup['dqn_agent']

        # learning agent
        agent = DQN_AirSim(drone,
                    n_steps=n_steps,
                    n_episodes=n_episodes,
                    gamma=gamma,
                    eps_min=eps_min,
                    eps_max=eps_max,
                    lbd=lbd,
                    batch_size=batch_size,
                    buffer_size=buffer_size,
                    target_update_freq=target_update_freq,
                    pre_fill_buffer=True,
                    target=False,
                    eval_factor=20,
                    dqn_agent=dqn_agent)

        print('Training Deep RL algorithms based on human demonstration.')
        train_deep_rl(run_id, drone, agent, n_episodes, n_steps)

    elif select_agent == 'ddpg':
        # read parameters from ini file
        ddpg_setup = config['DDPG']

        with tf.Session() as sess:

            # learning agent
            agent = DDPG_AirSim(sess,
                        drone,
                        n_steps=n_steps,
                        n_episodes=n_episodes,
                        actor_lrate=ddpg_setup.getfloat('actor_lrate'), # learning rate for the actor network
                        critic_lrate=ddpg_setup.getfloat('critic_lrate'), # learning rate for the critic network
                        gamma=ddpg_setup.getfloat('gamma'),         # discount factor
                        tau=ddpg_setup.getfloat('tau'),          # soft target update param
                        eps = ddpg_setup.getfloat('eps'),
                        batch_size=ddpg_setup.getint('batch_size'),
                        buffer_size=ddpg_setup.getint('buffer_size'),
                        target_update_freq=ddpg_setup.getint('target_update_freq'),
                        n_frames=ddpg_setup.getint('n_frames'))

            # train it
            sess.run(tf.global_variables_initializer())
            agent.update_nets()
            train_deep_rl_cv(run_id, drone, agent, n_episodes, n_steps)

    else:
        print('Invalid agent. Please check main.ini file')
        sys.exit(1)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--autoland", type=bool, help="lands drone if code crashes.")
    args = parser.parse_args()

    # loop for all different config files
    config_files = ['../config/hri/ddpg_big_0_v0.ini',
                    '../config/hri/ddpg_big_1_v0.ini',
                    '../config/hri/ddpg_big_2_v0.ini']

    for each_config_file in config_files:
        # start config parser
        print('READING {}'.format(each_config_file))
        config = configparser.ConfigParser()
        config.read(each_config_file)

        # check autoland and run main
        if args.autoland:
            try:
                main(config)
            except:
                print("* CODE CRASHED *")
                drone = CustomAirSim()
                # land
                drone.drone_land()
                drone.disarm()

        else:
            main(config)
