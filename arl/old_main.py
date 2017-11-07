#!/usr/bin/env python
""" main.py
OpenAI-Gym-like wrap on custom AirSim class so past written Deep RL algorithm
can be easily applied.
"""

__author__ = "Vinicius Guimaraes Goecks"
__version__ = "0.0.0"
__status__ = "Prototype"
__date__ = "June 12, 2017"

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

def main(inf_mode=False, use_gui=False):
    """
    Testing multiple inheritance and wrapping functions.
    """
    # start config parser
    config = configparser.ConfigParser()
    config.read('config_main.ini')

    main_setup = config['DEFAULT']

    # parameters and modes
    n_episodes = main_setup.getint('n_episodes')
    n_steps = main_setup.getint('n_steps')

    # test a few iterations
    run_id =  main_setup['run_id']

    # initial setup
    inf_mode = main_setup.getint('inf_mode')
    use_gui = main_setup.getint('use_gui')
    airsim_eom = main_setup.getint('airsim_eom')
    if airsim_eom:
        drone = AirSimEOM(n_steps, inf_mode=inf_mode, use_gui=use_gui)
    else:
        drone = CustomAirSim(n_steps, inf_mode=inf_mode, use_gui=use_gui)

    # start learning agents
    select_agent = main_setup['agent']
    print('AGENT: {}'.format(select_agent))
    if select_agent == 'human':
        # agent = HumanAgent(drone, n_episodes)
        agent = HumanAgentXBoxMulti(drone, n_episodes)
    elif select_agent == 'imitation':
        agent = ImitationAgent(drone, n_episodes)
    elif select_agent == 'interv':
        agent = InterventionAgentMulti(drone, n_episodes)
    elif select_agent == 'random':
        agent = RandomAgent_AirSim(drone, n_episodes)
    elif select_agent == 'dqn':
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

    elif select_agent == 'ddpg':
        with tf.Session() as sess:

            # learning agent
            agent = DDPG_AirSim(sess,
                        drone,
                        n_steps=n_steps,
                        n_episodes=n_episodes,
                        actor_lrate=.0001, # learning rate for the actor network
                        critic_lrate=.001, # learning rate for the critic network
                        gamma=.99,         # discount factor
                        tau=.001,          # soft target update param
                        eps = .1,
                        batch_size=16,
                        buffer_size=10000,
                        target_update_freq=1,
                        pre_fill_buffer=False,
                        target=False,
                        eval_factor=2)

            # train it
            sess.run(tf.global_variables_initializer())
            agent.update_nets()
            train_deep_rl_cv(run_id, drone, agent, n_episodes, n_steps)

    else:
        print('Invalid agent. Please check main.ini file')
        sys.exit(1)


    # select experiment mode
    exp_mode = main_setup.getint('exp')
    print('Experiment # {}:'.format(exp_mode))
    if exp_mode == 0:
        print('Imitation Learning with only Convolutional layers.')
        train_model(run_id, drone, agent, n_episodes, n_steps)
    elif exp_mode == 1:
        print('Imitation Learning with Convolutional and Recurrent layers.')
        train_model_multi(run_id, drone, agent, n_episodes, n_steps)
    elif exp_mode == 2:
        print('Training Deep RL algorithms based on human demonstration.')
        train_deep_rl(run_id, drone, agent, n_episodes, n_steps)
    elif exp_mode == 3:
        print('Validating the depth sensor.')
        drone.validate_depth()
    elif exp_mode == 4:
        print('Training human and getting data.')
        train_human(run_id, drone, agent, n_episodes, n_steps)
    elif exp_mode == 5:
        print('Finished DDPG.')
    elif exp_mode == 6:
        print('Using human to get data on computer vision mode')
        train_human_cv(run_id, drone, agent, n_episodes, n_steps)

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--autoland", type=bool, help="lands drone if code crashes.")
    parser.add_argument("--gui", type=bool, help="activates live gui.")
    parser.add_argument("--inf", type=bool, help="runs ad infinitum.")
    args = parser.parse_args()

    # check autoland and run main
    if args.autoland:
        try:
            main()
        except:
            print("* CODE CRASHED *")
            drone = CustomAirSim()
            # land
            drone.drone_land()
            drone.disarm()

    elif args.inf:
        if args.gui:
            print('Using GUI.')
            main(inf_mode=True, use_gui=True)

        else:
            main(inf_mode=True, use_gui=False)
    else:
        main()
