#!/usr/bin/env python
""" support.py
Support functions to connect to AirSim and others.
"""

__author__ = "Vinicius Guimaraes Goecks"
__version__ = "0.0.0"
__status__ = "Prototype"
__date__ = "June 7, 2017"

# import
import math
from math import cos, sin, tan, atan2, pi
import sys
import time
import cv2
import numpy as np
import resource
import gym
from gym import spaces
import pygame

import matplotlib.pyplot as plt

from PythonClient import *

from plotting import plot_history, process_avg
from gui import MachineGUI
from learn import ReplayBuffer
from neural import save_neural, load_neural


def train_model(run_id, env, agent, n_episodes=1, n_steps=50):
    """
    Funtion to train a model or environment using a specific learning agent.

    Inputs
    ----------
    env: defined environment/model (should follow base.py).
    agent: learning agent
    n_episodes: number of episodes
    n_steps: number of steps per episode

    Outputs
    ----------
    total_s: all states tested.
    total_a: all actions applied.
    total_r: all rewards received.
    """

    # reset environment/model
    data_folder = '/media/vinicius/vinicius_arl/data/'
    best_reward = -1e6
    n_act = env.act_n
    total_done = 0

    # run for a given number of episodes
    for i_episode in range(n_episodes):
        # get initial states after reseting environment
        print('****************************')
        print('Episode %i/%i' % (i_episode+1, n_episodes))
        # print('Memory usage: %s (kb)' %
        #       resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        # print('TESTING: took pic')
        # img = env.grab_depth()

        total_reward = 0
        start_reset_time = time.time()
        observation = env.reset()

        print('Total reset time: %.2f seconds.' %(time.time()-start_reset_time))

        # # TESTING MOVING TO A RANDOM POSITION
        # env.test_pos()

        # create object to store state-action pairs and rew-act pairs
        state_act = np.zeros((n_steps,observation.flatten().shape[0]+n_act))
        rew_act = np.zeros((n_steps,2+n_act)) # step, rew, act
        recording_already = False # flag to help identifying human intervention
        start_interv = 0

        if agent.name == 'interv':
            x = state_act[0:1,:-1]
            y = state_act[0:1,-1]
            agent.update_net(x, y)

        start_epi_time = time.time()
        for t in range(n_steps):
            if t % 100 == 0:
                print('Episode ',t)

            start_step_time = time.time()
            # select action based on current observation
            action = agent.act(observation)
            # print('ACTION: ', action)

            # record past observation
            past_observation = np.copy(observation)

            # save past observation and action taken
            state_act[t,:] = np.hstack((past_observation.flatten(),action.flatten()))

            # execute selected action, get new observation and reward
            observation, reward, done, _ = env.step(action)
            total_reward += reward

            # stream to gui
            if env.use_gui:
                # pipe image and action to gui
                env.gui.display(past_observation,action)

            # save rew and action taken
            rew_act[t,:] = np.hstack((t,reward,action))

            # check if goal or if reached any other simulation limit
            if done:
                print("Episode finished after {} steps.".format(t + 1))
                break

            # # report time/frequency of steps
            # print('Running at %i Hz.' %(1/(time.time()-start_step_time)))

        # end of episode
        print('* End of episode *')
        print('Total episode time: %.2f seconds.' %(time.time()-start_epi_time))
        print('Total reward: %.2f' % total_reward)

        if total_reward > best_reward:
            # better episode so far, keep data
            print('Found best reward: %.2f' % total_reward)
            best_reward = total_reward

        # # brake drone
        # if env.inf_mode == False:
        #     env.drone_brake()

        # save total rewards
        agent.history[i_episode,:] = [i_episode, total_reward]

        # dump zero rows and save collected data
        state_act = state_act[~(state_act==0).all(1)]
        rew_act = rew_act[~(rew_act==0).all(1)]
        np.savetxt(data_folder+run_id+'_imit_'+str(i_episode)+'.csv', state_act, delimiter=',')
        np.savetxt(data_folder+run_id+'_avg_'+str(i_episode)+'.csv', rew_act, delimiter=',')

    # REPORT
    print('\nGoal achieved in %i out of %i tries.' % (total_done, n_episodes))
    print('Success rate = ', total_done / n_episodes)

    # save and plot reward results
    np.savetxt(data_folder+run_id+'_rew.csv', agent.history, delimiter=',')
    plot_history(agent.history)
    process_avg(run_id, n_episodes)

    # go home
    print('Going HOME...')
    # env.drone_gohome()
    print('DONE! Thank you! :)')

def train_model_multi(run_id, env, agent, n_episodes=1, n_steps=50):
    """
    Funtion to train a model or environment using a specific learning agent.
    Updated to work with multiple actions.

    Inputs
    ----------
    env: defined environment/model (should follow base.py).
    agent: learning agent
    n_episodes: number of episodes
    n_steps: number of steps per episode

    Outputs
    ----------
    total_s: all states tested.
    total_a: all actions applied.
    total_r: all rewards received.
    """

    # reset environment/model
    data_folder = '/media/vinicius/vinicius_arl/data/'
    best_reward = -1e6
    total_done = 0
    n_act = 2
    updating_net = False

    # run for a given number of episodes
    for i_episode in range(n_episodes):
        # get initial states after reseting environment
        print('****************************')
        print('Episode %i/%i' % (i_episode+1, n_episodes))
        # print('Memory usage: %s (kb)' %
        #       resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        total_reward = 0
        start_reset_time = time.time()
        observation = env.reset()

        print('Total reset time: %.2f seconds.' %(time.time()-start_reset_time))

        # create object to store state-action pairs and rew-act pairs
        state_act = np.zeros((n_steps,observation.flatten().shape[0]+n_act)) # +1 for action
        rew_act = np.zeros((n_steps,2+n_act)) # step, rew, act
        recording_already = False # flag to help identifying human intervention
        start_interv = 0

        # step required to start tensorflow library before flight
        if agent.name == 'interv':
            x = state_act[0:1,:-agent.n_act]
            y = state_act[0:1,-agent.n_act:]
            agent.update_net(x, y)

        start_epi_time = time.time()
        for t in range(n_steps):
            # report time steps
            # start_step_time = time.time()
            if t % 200 == 0:
                print('Time Step %i/%i' % (t, n_steps))

            # select action based on current observation
            env.t_step = t
            action = agent.act(observation)

            # record past observation
            past_observation = np.copy(observation)

            # save past observation and action taken
            state_act[t,:] = np.hstack((past_observation.flatten(),action.flatten()))

            # check if human is intervening
            if updating_net:
                if agent.name == 'interv':
                    if agent.interv:
                        if not recording_already:
                            print('** START recording ...')
                            start_interv = t
                            recording_already = True

                    elif agent.interv == False:
                        if recording_already:
                            print('** END recording ...')
                            recording_already = False

                    # update net when have a given number of samples
                    if recording_already:
                        batch_count = t - start_interv
                        # print("Batch count = ", batch_count)
                        # print("Step count = ", t)
                        if (batch_count) >= 31:

                            # update agent network
                            x = state_act[0:1,:-agent.n_act]
                            y = state_act[0:1,-agent.n_act:]
                            agent.update_net(x, y)

                            # reset times
                            start_interv = t

            # execute selected action, get new observation and reward
            observation, reward, done, _ = env.step_multi(action)
            total_reward += reward

            # work on replay when recording human data
            if recording_already:
                agent.replay.add(past_observation, action, reward, done,observation)

            # stream to gui
            if env.use_gui:
                # pipe image and action to gui
                env.gui.display(past_observation,action)

            # save rew and action taken
            rew_act[t,:] = np.hstack((t,reward,action))

            # check if goal or if reached any other simulation limit
            if done:
                print("Episode finished after {} steps.".format(t + 1))
                break

            # # report time/frequency of steps
            # print('Running at %i Hz.' %(1/(time.time()-start_step_time)))

        # end of episode
        print('* End of episode *')
        print('Total episode time: %.2f seconds.' %(time.time()-start_epi_time))

        if total_reward > best_reward:
            # better episode so far, keep data
            print('Found best reward: %.2f' % total_reward)
            best_reward = total_reward

        # brake drone
        if env.inf_mode == False:
            env.drone_brake()

        # save total rewards
        agent.history[i_episode,:] = [i_episode, total_reward]

        # dump zero rows and save collected data
        state_act = state_act[~(state_act==0).all(1)]
        rew_act = rew_act[~(rew_act==0).all(1)]
        env.hist_attitude = env.hist_attitude[~(env.hist_attitude==0).all(1)]
        np.savetxt(data_folder+run_id+'_imit_'+str(i_episode)+'.csv', state_act, delimiter=',')
        np.savetxt(data_folder+run_id+'_avg_'+str(i_episode)+'.csv', rew_act, delimiter=',')
        np.savetxt(data_folder+run_id+'_att_'+str(i_episode)+'.csv', env.hist_attitude, delimiter=',')
        # save_neural(agent.model, name=run_id)

    # REPORT
    print('\nGoal achieved in %i out of %i tries.' % (total_done, n_episodes))
    print('Success rate = ', total_done / n_episodes)

    # save and plot reward results
    print('Hold on! Saving data and plotting stuff!')
    save_neural(agent.model, name=run_id)
    np.savetxt(data_folder+run_id+'_rew.csv', agent.history, delimiter=',')
    plot_history(agent.history)
    process_avg(run_id, n_episodes)
    print('Done! Feel free to kill the process if stuck GOING HOME.')

    # go home
    print('Going HOME...')
    env.drone_gohome()
    print('DONE! Thank you! :)')

def train_deep_rl(run_id, env, agent, n_episodes=1, n_steps=50):
    """
    Funtion to train a model or environment using a specific learning agent.
    Updated to work with multiple actions.

    It integrates human demonstration to train the deep rl algorithm during the
    initial state-space exploration.

    Inputs
    ----------
    env: defined environment/model (should follow base.py).
    agent: learning agent
    n_episodes: number of episodes
    n_steps: number of steps per episode

    """

    # reset environment/model
    data_folder = '/media/vinicius/vinicius_arl/data/'
    best_reward = -1e6
    total_done = 0
    n_act = env.act_n
    updating_net = False

    # run for a given number of episodes
    for i_episode in range(n_episodes):
        # get initial states after reseting environment
        print('****************************')
        print('Episode %i/%i' % (i_episode+1, n_episodes))
        # print('Memory usage: %s (kb)' %
        #       resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        total_reward = 0
        start_reset_time = time.time()
        observation = env.reset()

        print('Total reset time: %.2f seconds.' %(time.time()-start_reset_time))

        # create object to store state-action pairs and rew-act pairs
        state_act = np.zeros((n_steps,observation.flatten().shape[0]+n_act)) # +1 for action
        rew_act = np.zeros((n_steps,2+n_act)) # step, rew, act
        recording_already = False # flag to help identifying human intervention
        start_interv = 0

        # step required to start tensorflow library before flight
        if agent.name == 'interv':
            x = state_act[0:1,:-agent.n_act]
            y = state_act[0:1,-agent.n_act:]
            agent.update_net(x, y)

        start_epi_time = time.time()
        for t in range(n_steps):
            # report time steps
            # start_step_time = time.time()
            if t % 200 == 0:
                print('Time Step %i/%i' % (t, n_steps))

            # select action based on current observation
            env.t_step = t
            agent.t = t
            action = agent.act(observation)

            # record past observation
            past_observation = np.copy(observation)

            # save past observation and action taken
            state_act[t,:] = np.hstack((past_observation.flatten(),action))

            # execute selected action, get new observation and reward
            observation, reward, done, _ = env.step_dqn(action)

            # return max reward if the agent is intervening
            if agent.interv:
                reward = 1
                # print('Human reward = ',reward)
            total_reward += reward

            # # work on replay
            if agent.has_replay:
                agent.use_replay(past_observation, action, reward, done,
                                 observation)

            # stream to gui
            if env.use_gui:
                # pipe image and action to gui
                env.gui.display(past_observation,action)

            # save rew and action taken
            rew_act[t,:] = np.hstack((t,reward,action))

            # check if goal or if reached any other simulation limit
            # print(done)
            if done:
                print("Episode finished after {} steps.".format(t + 1))
                break

            # # report time/frequency of steps
            # print('Running at %i Hz.' %(1/(time.time()-start_step_time)))

        # end of episode
        print('* End of episode *')
        print('Total episode time: %.2f seconds.' %(time.time()-start_epi_time))

        if total_reward > best_reward:
            # better episode so far, keep data
            print('Found best reward: %.2f' % total_reward)
            best_reward = total_reward

        # brake drone
        if env.inf_mode == False:
            env.drone_brake()

        # save total rewards
        agent.history[i_episode,:] = [i_episode, total_reward]

        # dump zero rows and save collected data
        state_act = state_act[~(state_act==0).all(1)]
        rew_act = rew_act[~(rew_act==0).all(1)]
        env.hist_attitude = env.hist_attitude[~(env.hist_attitude==0).all(1)]
        np.savetxt(data_folder+run_id+'_imit_'+str(i_episode)+'.csv', state_act, delimiter=',')
        np.savetxt(data_folder+run_id+'_avg_'+str(i_episode)+'.csv', rew_act, delimiter=',')
        np.savetxt(data_folder+run_id+'_att_'+str(i_episode)+'.csv', env.hist_attitude, delimiter=',')
        save_neural(agent.model, name=run_id+'_'+str(i_episode))

    # REPORT
    print('\nGoal achieved in %i out of %i tries.' % (total_done, n_episodes))
    print('Success rate = ', total_done / n_episodes)

    # save and plot reward results
    print('Hold on! Saving data and plotting stuff!')
    save_neural(agent.model, name=run_id)
    np.savetxt(data_folder+run_id+'_rew.csv', agent.history, delimiter=',')
    plot_history(agent.history)
    process_avg(run_id, n_episodes)
    print('Done! Feel free to kill the process if stuck GOING HOME.')

    # go home
    print('Going HOME...')
    env.drone_gohome()
    print('DONE! Thank you! :)')

def train_deep_rl_cv(run_id, env, agent, n_episodes=1, n_steps=50):
    """
    Funtion to train a model or environment using a specific learning agent.
    Updated to work with multiple actions.

    It integrates human demonstration to train the deep rl algorithm during the
    initial state-space exploration.

    Inputs
    ----------
    env: defined environment/model (should follow base.py).
    agent: learning agent
    n_episodes: number of episodes
    n_steps: number of steps per episode

    """

    # reset environment/model
    data_folder = '/media/vinicius/vinicius_arl/data/'
    best_reward = -1e6
    total_done = 0
    n_act = env.act_n
    updating_net = False

    # run for a given number of episodes
    for i_episode in range(n_episodes):
        # get initial states after reseting environment
        print('****************************')
        print('Episode %i/%i' % (i_episode+1, n_episodes))
        # print('Memory usage: %s (kb)' %
        #       resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        total_reward = 0
        env.hist_attitude = np.zeros((n_steps,10))
        start_reset_time = time.time()
        observation = env.reset()

        print('Total reset time: %.2f seconds.' %(time.time()-start_reset_time))

        # create object to store state-action pairs and rew-act pairs
        state_act = np.zeros((n_steps,observation.flatten().shape[0]+n_act)) # +1 for action
        rew_act = np.zeros((n_steps,2+n_act)) # step, rew, act
        recording_already = False # flag to help identifying human intervention
        start_interv = 0

        # # update target nets on ddpg
        # if agent.name == 'DDPG':
        #     # check episode number to update target networks
        #     if (i_episode % agent.target_update_freq == 0):
        #         agent.update_nets()

        # step required to start tensorflow library before flight
        if agent.name == 'interv':
            x = state_act[0:1,:-agent.n_act]
            y = state_act[0:1,-agent.n_act:]
            agent.update_net(x, y)

        start_epi_time = time.time()
        for t in range(n_steps):
            # report time steps
            # start_step_time = time.time()
            # if t % 200 == 0:
            #     print('Time Step %i/%i' % (t, n_steps))

            # select action based on current observation
            env.t_step = t
            agent.curr_t = t
            agent.curr_ep = i_episode
            env.stacked_frames = agent.stacked_frames

            action = agent.act(observation)
            # print(action)

            # record past observation
            past_observation = np.copy(observation)
            past_replay_obs = agent.stacked_frames

            # save past observation and action taken
            state_act[t,:] = np.hstack((past_observation.flatten(),action.flatten()))

            # execute selected action, get new observation and reward
            observation, reward, done, _ = env.step(action)
            replay_obs = agent.stacked_frames

            # return max reward if the agent is intervening
            if agent.interv:
                reward = 1
                # print('Human reward = ',reward)
            total_reward += reward

            # # work on replay
            if agent.has_replay and (t % agent.n_frames == 0):
                # # original one, using replay with no stack frames
                # agent.use_replay(past_observation, action, reward, done,
                #                  observation)

                # # modified to use replay with stacked frames
                # agent.use_replay(past_replay_obs, action, reward, done,
                #                  replay_obs)

                # # modified to just add to replay, but dont use it (wait end)
                agent.add_to_replay(past_replay_obs, action, reward, done,
                                 replay_obs)

            # stream to gui
            if env.use_gui:
                # pipe image and action to gui
                env.gui.display(past_observation,action)

            # save rew and action taken
            rew_act[t,:] = np.hstack((t,reward,action))

            # check if goal or if reached any other simulation limit
            # print(done)
            if done:
                print("Episode finished after {} steps.".format(t + 1))
                break

            # # report time/frequency of steps
            # print('Running at %i Hz.' %(1/(time.time()-start_step_time)))

        # end of episode
        print('* End of episode *')
        print('Total episode time: %.2f seconds.' %(time.time()-start_epi_time))
        print('Total episode reward: ', total_reward)

        print('Using replay...')
        agent.use_offline_replay()

        if total_reward > best_reward:
            # better episode so far, keep data
            print('Found best reward: %.2f' % total_reward)
            best_reward = total_reward


        # save total rewards
        agent.history[i_episode,:] = [i_episode, total_reward]

        # dump zero rows and save collected data
        state_act = state_act[~(state_act==0).all(1)]
        rew_act = rew_act[~(rew_act==0).all(1)]
        env.hist_attitude = env.hist_attitude[~(env.hist_attitude==0).all(1)]
        np.savetxt(data_folder+run_id+'_imit_'+str(i_episode)+'.csv', state_act, delimiter=',')
        np.savetxt(data_folder+run_id+'_avg_'+str(i_episode)+'.csv', rew_act, delimiter=',')
        np.savetxt(data_folder+run_id+'_att_'+str(i_episode)+'.csv', env.hist_attitude, delimiter=',')
        np.savetxt(data_folder+run_id+'_rew.csv', agent.history, delimiter=',')
        np.savetxt(data_folder+run_id+'_crashes.csv', [env.total_crashes])
        # save_neural(agent.model, name=run_id+'_'+str(i_episode))

    # REPORT
    print('\nGoal achieved in %i out of %i tries.' % (total_done, n_episodes))
    print('Success rate = ', total_done / n_episodes)

    # save and plot reward results
    print('Hold on! Saving data and plotting stuff!')
    # save_neural(agent.model, name=run_id)
    if agent.has_replay:
        agent.actor.saver.save(agent.actor.sess, '/media/vinicius/vinicius_arl/neural_models/' + run_id + '_actor.tflearn')
        agent.critic.saver.save(agent.critic.sess, '/media/vinicius/vinicius_arl/neural_models/' +  run_id + '_critic.tflearn')
    #plot_history(agent.history)

    print('Done! Feel free to kill the process if stuck GOING HOME.')

def train_human_cv(run_id, env, agent, n_episodes=1, n_steps=50):
    """
    Funtion to train a model or environment using a specific learning agent.
    Updated to work with multiple actions.

    It integrates human demonstration to train the deep rl algorithm during the
    initial state-space exploration.

    Inputs
    ----------
    env: defined environment/model (should follow base.py).
    agent: learning agent
    n_episodes: number of episodes
    n_steps: number of steps per episode

    """

    # reset environment/model
    data_folder = '/media/vinicius/vinicius_arl/data/'
    best_reward = -1e6
    total_done = 0
    n_act = env.act_n

    # run for a given number of episodes
    for i_episode in range(n_episodes):
        # get initial states after reseting environment
        print('****************************')
        print('Episode %i/%i' % (i_episode+1, n_episodes))
        # print('Memory usage: %s (kb)' %
        #       resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        total_reward = 0
        env.hist_attitude = np.zeros((n_steps,10))
        start_reset_time = time.time()
        observation = env.reset()

        print('Total reset time: %.2f seconds.' %(time.time()-start_reset_time))

        # create object to store state-action pairs and rew-act pairs
        state_act = np.zeros((n_steps,observation.flatten().shape[0]+n_act)) # +1 for action
        rew_act = np.zeros((n_steps,2+n_act)) # step, rew, act
        recording_already = False # flag to help identifying human intervention
        start_interv = 0


        start_epi_time = time.time()
        for t in range(n_steps):
            # report time steps
            # start_step_time = time.time()
            # if t % 200 == 0:
            #     print('Time Step %i/%i' % (t, n_steps))

            # select action based on current observation
            env.t_step = t
            agent.curr_t = t
            agent.curr_ep = i_episode
            env.stacked_frames = agent.stacked_frames

            action = agent.act(observation)
            # print(action)

            # record past observation
            past_observation = np.copy(observation)

            # save past observation and action taken
            state_act[t,:] = np.hstack((past_observation.flatten(),action.flatten()))

            # execute selected action, get new observation and reward
            observation, reward, done, _ = env.step(action)

            total_reward += reward

            # stream to gui
            if env.use_gui:
                # pipe image and action to gui
                env.gui.display(past_observation,action)

            # save rew and action taken
            rew_act[t,:] = np.hstack((t,reward,action))

            # check if goal or if reached any other simulation limit
            # print(done)
            if done:
                print("Episode finished after {} steps.".format(t + 1))
                break

            # # report time/frequency of steps
            # print('Running at %i Hz.' %(1/(time.time()-start_step_time)))

        # end of episode
        print('* End of episode *')
        print('Total episode time: %.2f seconds.' %(time.time()-start_epi_time))
        print('Total episode reward: ', total_reward)


        if total_reward > best_reward:
            # better episode so far, keep data
            print('Found best reward: %.2f' % total_reward)
            best_reward = total_reward


        # save total rewards
        agent.history[i_episode,:] = [i_episode, total_reward]

        # dump zero rows and save collected data
        state_act = state_act[~(state_act==0).all(1)]
        rew_act = rew_act[~(rew_act==0).all(1)]
        env.hist_attitude = env.hist_attitude[~(env.hist_attitude==0).all(1)]
        np.savetxt(data_folder+run_id+'_imit_'+str(i_episode)+'.csv', state_act, delimiter=',')
        np.savetxt(data_folder+run_id+'_avg_'+str(i_episode)+'.csv', rew_act, delimiter=',')
        np.savetxt(data_folder+run_id+'_att_'+str(i_episode)+'.csv', env.hist_attitude, delimiter=',')
        np.savetxt(data_folder+run_id+'_rew.csv', agent.history, delimiter=',')
        np.savetxt(data_folder+run_id+'_crashes.csv', [env.total_crashes], delimiter=',')

    # REPORT
    print('\nGoal achieved in %i out of %i tries.' % (total_done, n_episodes))
    print('Success rate = ', total_done / n_episodes)

    # save and plot reward results
    print('Hold on! Saving data and plotting stuff!')
    # np.savetxt(data_folder+run_id+'_rew.csv', agent.history, delimiter=',')
    plot_history(agent.history)

    print('Done! Feel free to kill the process if stuck GOING HOME.')


def train_human(run_id, env, agent, n_episodes=1, n_steps=50):
    """
    Funtion to train a model or environment using a specific learning agent.
    Updated to work with multiple actions.

    It integrates human demonstration to train the deep rl algorithm during the
    initial state-space exploration.

    Inputs
    ----------
    env: defined environment/model (should follow base.py).
    agent: learning agent
    n_episodes: number of episodes
    n_steps: number of steps per episode

    """

    # reset environment/model
    data_folder = '/media/vinicius/vinicius_arl/data/'
    best_reward = -1e6
    total_done = 0
    n_act = 2
    updating_net = False

    # run for a given number of episodes
    for i_episode in range(n_episodes):
        # get initial states after reseting environment
        print('****************************')
        print('Episode %i/%i' % (i_episode+1, n_episodes))
        # print('Memory usage: %s (kb)' %
        #       resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        total_reward = 0
        start_reset_time = time.time()
        observation = env.reset()

        print('Total reset time: %.2f seconds.' %(time.time()-start_reset_time))

        # create object to store state-action pairs and rew-act pairs
        state_act = np.zeros((n_steps,observation.flatten().shape[0]+n_act)) # +1 for action
        rew_act = np.zeros((n_steps,2+n_act)) # step, rew, act
        recording_already = False # flag to help identifying human intervention
        start_interv = 0


        start_epi_time = time.time()
        for t in range(n_steps):
            # report time steps
            # start_step_time = time.time()
            if t % 200 == 0:
                print('Time Step %i/%i' % (t, n_steps))

            # select action based on current observation
            env.t_step = t
            action = agent.act(observation)

            # record past observation
            past_observation = np.copy(observation)

            # save past observation and action taken
            state_act[t,:] = np.hstack((past_observation.flatten(),action.flatten()))

            # execute selected action, get new observation and reward
            observation, reward, done, _ = env.step_multi(action)
            total_reward += reward

            # work on replay when recording human data
            if recording_already:
                agent.replay.add(past_observation, action, reward, done,observation)

            # stream to gui
            if env.use_gui:
                # pipe image and action to gui
                env.gui.display(past_observation,action)

            # save rew and action taken
            rew_act[t,:] = np.hstack((t,reward,action))

            # check if goal or if reached any other simulation limit
            if done:
                print("Episode finished after {} steps.".format(t + 1))
                break

            # # report time/frequency of steps
            # print('Running at %i Hz.' %(1/(time.time()-start_step_time)))

        # end of episode
        print('* End of episode *')
        print('Total episode time: %.2f seconds.' %(time.time()-start_epi_time))

        if total_reward > best_reward:
            # better episode so far, keep data
            print('Found best reward: %.2f' % total_reward)
            best_reward = total_reward

        # brake drone
        if env.inf_mode == False:
            env.drone_brake()

        # save total rewards
        agent.history[i_episode,:] = [i_episode, total_reward]

        # dump zero rows and save collected data
        state_act = state_act[~(state_act==0).all(1)]
        rew_act = rew_act[~(rew_act==0).all(1)]
        env.hist_attitude = env.hist_attitude[~(env.hist_attitude==0).all(1)]
        np.savetxt(data_folder+run_id+'_imit_'+str(i_episode)+'.csv', state_act, delimiter=',')
        np.savetxt(data_folder+run_id+'_avg_'+str(i_episode)+'.csv', rew_act, delimiter=',')
        np.savetxt(data_folder+run_id+'_att_'+str(i_episode)+'.csv', env.hist_attitude, delimiter=',')
        # save_neural(agent.model, name=run_id)

    # REPORT
    print('\nGoal achieved in %i out of %i tries.' % (total_done, n_episodes))
    print('Success rate = ', total_done / n_episodes)

    # save and plot reward results
    print('Hold on! Saving data and plotting stuff!')
    save_neural(agent.model, name=run_id)
    np.savetxt(data_folder+run_id+'_rew.csv', agent.history, delimiter=',')
    plot_history(agent.history)
    process_avg(run_id, n_episodes)
    print('Done! Feel free to kill the process if stuck GOING HOME.')

    # go home
    print('Going HOME...')
    env.drone_gohome()
    print('DONE! Thank you! :)')


class CustomAirSim(AirSimClient, gym.Env):
    """
    Custom class for handling AirSim commands like connect, takeoff, land, etc.
    """
    def __init__(self,n_steps, inf_mode=False, use_gui=False):
        AirSimClient.__init__(self, "127.0.0.1") # connect to ip
        self.max_timeout = 20 # seconds
        self.connect_AirSim()
        self.inf_mode = inf_mode

        # possible bank actions (m/s)
        self.actions = [.5, 0, -.5]
        self.dt = .05 # seconds, interval between actions
        self.forward_vel = 2
        self.vy_scale = 2
        self.n_steps = n_steps

        # parameters for turn maneuver
        self.old_yaw = 0
        self.old_roll = 0
        self.set_z = -5

        # action limits
        self.act_low = -1
        self.act_high = 1
        self.act_n = 1 # one action
        self.map_length = 110 # 380

        # depth parameters and compression factors
        # depth 1/4: 36 x 64 pics
        # depth 1/8: 18 x 32 pics
        depth_width = 256
        depth_height = 144
        self.reduc_factor = 1/4 # multiply original size by this factor
        self.depth_w = int(depth_width*self.reduc_factor)
        self.depth_h = int(depth_height*self.reduc_factor)

        # gui parameters
        self.use_gui = use_gui
        if self.use_gui:
            # create gui object
            self.gui = MachineGUI(self.depth_w, self.depth_h, start_gui=True)


        # store current and reference states
        # step, roll, pitch, yaw (current), roll, pitch, yaw (ref)
        self.t_step = 0
        # step, 3 pos, 3 vel, 3 att, 3 ref att
        self.hist_attitude = np.zeros((n_steps,13))

    # CUSTOM ARL COMMANDS
    def setOffboardModeTrue(self):
        return self.client.call('setOffboardMode', True)
    #####################

    def connect_AirSim(self):
        """
        Establish initial connection to AirSim client and set GPS.
        """
        # get GPS
        print("Waiting for home GPS location to be set...")
        # home = self.getHomePoint()
        home = self.getPosition()
        while ((home[0] == 0 and home[1] == 0 and home[2] == 0) or
               math.isnan(home[0]) or  math.isnan(home[1]) or  math.isnan(home[2])):
            time.sleep(1)
            # home = self.getHomePoint()
            home = self.getPosition()

        print("Home lat=%g, lon=%g, alt=%g" % tuple(home))

        # save home position and gps
        self.home = home
        self.home_pos = self.getPosition()

    def drone_takeoff(self):
        """
        Takeoff function.
        """
        # # arm drone
        # if (not self.arm()):
        #     print("Failed to arm the drone.")
        #     sys.exit(1)
        #
        # # takeoff
        # if (self.getLandedState() == LandedState.Landed):
        #     print("Taking off...")
        #     if (not self.takeoff(20)):
        #         print("Failed to reach takeoff altitude after 20 seconds.")
        #         sys.exit(1)
        #     print("Should now be flying...")
        # else:
        #     print("It appears the drone is already flying")

        # stabilize
        # for i in range(np.abs(self.set_z)):
        #     self.drone_altitude(-i)
        # try:
        #     self.drone_altitude(self.set_z)
        #     self.takeoff(1)
        # except:
        #     self.drone_altitude(self.set_z)
        # self.drone_altitude(self.set_z)

        self.arm()
        self.setOffboardModeTrue()
        time.sleep(5)
        self.takeoff(20)
        time.sleep(10)
        # self.hover()
        # time.sleep(6)
        # self.drone_altitude(self.set_z)
        # time.sleep(6)

    def drone_land(self):
        """
        Land function.
        """
        if (self.getLandedState() != LandedState.Landed):
            print("Landing...")
            if (not self.land(20)):
                print("Failed to reach takeoff altitude after 60 seconds.")
                sys.exit(1)
            print("Landed.")
        else:
            print("It appears the drone is already landed.")

    def drone_forward(self, vx, angle=0):
        """
        Go forward on camera frame with vx speed (m/s).
        """
        # define move parameters
        drivetrain = DrivetrainType.MaxDegreeOfFreedom
        yaw_mode = YawMode(False, angle)
        duration = self.dt # seconds
        self.moveByVelocityZ(vx, 0, 0, duration, drivetrain, yaw_mode)

        return duration

    def drone_bank(self, vy):
        """
        Bank on camera frame with vy speed (m/s).
        """
        # define move parameters
        drivetrain = DrivetrainType.MaxDegreeOfFreedom
        yaw_mode = YawMode(False,0)
        duration = self.dt # seconds
        command = np.clip(vy*self.vy_scale,-self.vy_scale,self.vy_scale)

        self.moveByVelocityZ(self.forward_vel, command, self.set_z, duration, drivetrain, yaw_mode)

        return duration

    def drone_turn(self, vy):
        """
        Combines pitch, roll, and yaw for turn maneuvers.
        """
        # parse commands
        if vy == 2: # pitch forward
            # define angular motion
            pitch = -10
            roll = 0
            yaw = self.old_yaw

        elif vy == 3: # break
            # define angular motion
            pitch = 10
            roll = 0
            yaw = self.old_yaw

        else:
            # define angula motion
            pitch = -1
            roll = 15*vy
            yaw = self.old_yaw + vy*2

        # send commands
        self.client.call('moveByAngle', pitch, roll, self.set_z, yaw, self.dt)

        # store applied yaw so we can send cumulative commands to change attitude
        self.old_yaw = yaw

        return self.dt

    def drone_turn_multi(self, act):
        """
        Combines pitch, roll, and yaw for turn maneuvers.
        Updated to work with multiple actions.
        """
        # parse commands
        lat = act[0]
        lon = act[1]

        # # scale actions to adequate control inputs
        # pitch = .5*lon
        # yaw = self.old_yaw + lat*1
        # if lat == 0:
        #     roll = 0
        # else:
        #     # roll = self.old_roll + lat*2
        #     # roll = np.clip(roll,-20,20)
        #     roll = lat*2
        #
        # # send commands
        # self.client.call('moveByAngle', pitch, roll, self.set_z, yaw, self.dt)
        #
        # # store applied yaw so we can send cumulative commands to change attitude
        # self.old_yaw = yaw
        # self.old_roll = roll

        # define move parameters
        drivetrain = DrivetrainType.MaxDegreeOfFreedom
        yaw_mode = YawMode(False,0)
        duration = self.dt # seconds
        command = np.clip(lat*self.vy_scale,-self.vy_scale,self.vy_scale)

        self.moveByVelocityZ(-self.forward_vel*lon, command, self.set_z, duration, drivetrain, yaw_mode)

        # save ref controls and attitude
        curr_pos = self.getPosition()
        curr_vel = self.getVelocity()
        curr_att = np.rad2deg(self.getRollPitchYaw())

        self.hist_attitude[self.t_step,:] = [self.t_step,
                                             curr_pos[0],
                                             curr_pos[1],
                                             curr_pos[2],
                                             curr_vel[0],
                                             curr_vel[1],
                                             curr_vel[2],
                                             curr_att[1],
                                             -curr_att[0],
                                             curr_att[2],
                                             roll,
                                             pitch,
                                             yaw]

        return self.dt


    def test_pos(self):
        """
        Testing moveToPosition. Going to a random Y position.
        """
        # define move parameters
        z = -5 # altitude
        velocity = 5
        drivetrain = DrivetrainType.MaxDegreeOfFreedom
        yaw_mode = YawMode(False, 0)
        self.moveToPosition(0, -30, z, velocity, 30, drivetrain, yaw_mode, 0,1)

    def take_pic(self):
        """
        Return rgb image.
        """
        # get rgb image
        result = self.setImageTypeForCamera(0, AirSimImageType.Scene)
        result = self.getImageForCamera(0, AirSimImageType.Scene)
        if (result != "\0"):
            # rgb
            rawImage = np.fromstring(result, np.int8)
            png = cv2.imdecode(rawImage, cv2.IMREAD_UNCHANGED)
            cv2.imwrite('rgb.png',png)

    def grab_rgb(self):
        """
        Get camera rgb image and return array of pixel values.
        Returns numpy ndarray.
        """
        # get rgb image
        result = self.setImageTypeForCamera(0, AirSimImageType.Scene)
        result = self.getImageForCamera(0, AirSimImageType.Scene)
        if (result != "\0"):
            # rgb
            rawImage = np.fromstring(result, np.int8)
            png = cv2.imdecode(rawImage, cv2.IMREAD_UNCHANGED)

            return png
    def simGetImage(self, camera_id, image_type):
        # because this method returns std::vector<uint8>, msgpack decides to encode it as a string unfortunately.
        result = self.client.call('simGetImage', camera_id, image_type)
        if (result == "" or result == "\0"):
            return None
        return np.fromstring(result, np.int8)

    def grab_depth(self):
        """
        Get camera depth image and return array of pixel values.
        Returns numpy ndarray.
        """
        # get depth image
        # result = self.setImageTypeForCamera(0, AirSimImageType.Depth)
        result = self.simGetImage(0, AirSimImageType.Depth)
        if (result != "\0"):
            # depth
            rawImage = np.fromstring(result, np.int8)
            png = cv2.imdecode(rawImage, cv2.IMREAD_UNCHANGED)
            if png is not None:
                # return pic, only first channel is enough for depth
                # apply threshold
                # png[:,:,0] = self.tsh_distance(100,png[:,:,0])
                return png[:,:,0]
            else:
                print('Couldnt take one depth pic.')
                return np.zeros((144,256)) # empty picture

    def take_depth_pic(self):
        """
        Return depth image.
        """
        # get depth image
        result = self.setImageTypeForCamera(0, AirSimImageType.Depth)
        result = self.getImageForCamera(0, AirSimImageType.Depth)
        if (result != "\0"):
            # depth
            rawImage = np.fromstring(result, np.int8)
            png = cv2.imdecode(rawImage, cv2.IMREAD_UNCHANGED)
            if png is not None:
                print('Depth pic taken.')
                png[:,:,0] = self.tsh_distance(100,png[:,:,0])
                cv2.imwrite('depth_tsh.png',png[:,:,0])
            else:
                print('Couldnt take one depth pic.')

    def tsh_distance(self, tsh_val, img):
        """
        Threshold pixel values on image. Set to zero if less than tsh_val.
        """
        low_val_idx = img < tsh_val
        img[low_val_idx] = 0

        return img

    def validate_depth(self):
        """
        Validate depth image, trying to make the ground thruth more similar
        to real sensors.
        For example, ZED Camera Depth only sees between 0.5-20 meters, blurred.
        """
        # takeoff if landed
        if (self.getLandedState() == LandedState.Landed):
            print("Landed.")
            try:
                self.drone_takeoff()
            except:
                print("Takeoff failed. Trying again...")
                # might need to reconnect first
                CustomAirSim.__init__(self)
                self.drone_takeoff()

            # arm drone
            if (not self.arm()):
                print("Failed to arm the drone.")
                sys.exit(1)

        # take a few pics
        # time.sleep(25)
        for i in range(5):
            self.take_depth_pic()

    def drone_brake(self):
        """
        Brake the drone and maintain altitude.
        """
        # break drone
        z = -3 # altitude
        drivetrain = DrivetrainType.MaxDegreeOfFreedom
        yaw_mode = YawMode(False, 0)
        duration = 1 # seconds
        self.moveByVelocityZ(0, 0, z, duration, drivetrain, yaw_mode)
        time.sleep(duration)
        self.moveByVelocityZ(0, 0, z, duration, drivetrain, yaw_mode)
        time.sleep(duration)

    def drone_altitude(self, alt):
        """
        Changes drone's altitude.
        """
        max_wait_seconds = 5
        drivetrain = DrivetrainType.MaxDegreeOfFreedom
        yaw_mode = YawMode(False,0)

        print('Changing altitude to %i meters...' %(alt*(-1)))
        self.moveToZ(alt, 3, max_wait_seconds, yaw_mode, 0, 1)
        time.sleep(-alt)

    def correct_altitude(self):
        """
        Changes drone's altitude.
        """
        drivetrain = DrivetrainType.MaxDegreeOfFreedom
        yaw_mode = YawMode(False,0)
        self.moveToZ(self.set_z, 3, 0, yaw_mode, 0, 1)

    def drone_gohome(self):
        """
        Climb high, go home, and land.
        """
        # make sure offboard mode is on
        self.setOffboardModeTrue()

        # move home
        print('Moving to HOME position...')

        # compute distance from home
        dist = self.dist_home()

        # while (self.getLandedState() != LandedState.Landed):
        while dist > 10:
            self.goHome()
            time.sleep(5)

            # compute distance from home
            dist = self.dist_home()

        print('Close enough.')
        z = -3
        max_wait_seconds = 30
        drivetrain = DrivetrainType.MaxDegreeOfFreedom
        yaw_mode = YawMode(False,0)

        print('Descending to %i meters...' %(z*(-1)))
        self.moveToZ(z, 10, max_wait_seconds, yaw_mode, 0, 1)
        time.sleep(max_wait_seconds)

    def dist_home(self):
        """
        Compute current distance from home point.
        """
        current_pos = self.getPosition()
        dist = np.sqrt((current_pos[0] - self.home_pos[0])**2 + (current_pos[1] - self.home_pos[1])**2)

        return dist

    def report_status(self):
        """
        Report position, velocity, and other current states.
        """
        print("* STATUS *")
        print("Position lat=%g, lon=%g, alt=%g" % tuple(self.getGpsLocation()))
        print("Velocity vx=%g, vy=%g, vz=%g" % tuple( self.getVelocity()))
        print("Attitude pitch=%g, roll=%g, yaw=%g" % tuple(self.getRollPitchYaw()))

    def step(self, action):
        """
        Step agent based on computed action.
        Return reward and if check if episode is done.
        """
        # take action
        # wait_time = self.drone_turn(float(action))
        wait_time = self.drone_bank(float(action))
        time.sleep(wait_time)
        self.correct_altitude()

        # get next state
        img = self.grab_depth()
        res = self.preprocess(img)

        # compute reward
        reward = self.compute_reward(res)

        # check if done
        current_pos = self.getPosition()
        if current_pos[0] < 110: # 50 for small course / 105 for big
            done = 0
        else:
            done = 1

        return res, reward, done, {}

    def step_dqn(self, action):
        """
        Step agent based on computed action.
        Return reward and if check if episode is done.
        """
        # map action (example: convert from 0,1,2 to -1,0,1)
        action = action - 1

        # wait_time = self.drone_turn(float(action))
        wait_time = self.drone_bank(float(action))
        time.sleep(wait_time)

        # get next state
        img = self.grab_depth()
        res = self.preprocess(img)

        # compute reward
        reward = self.compute_reward(res)

        # check if done
        current_pos = self.getPosition()
        if current_pos[0] < self.map_length: # 50 for small course / 105 for big
            done = 0
        else:
            done = 1

        return res, reward, done, {}

    def step_multi(self, action):
        """
        Step agent based on computed action.
        Return reward and if check if episode is done.
        """
        # take action
        wait_time = self.drone_turn_multi(action)
        time.sleep(wait_time)

        # get next state
        img = self.grab_depth()
        res = self.preprocess(img)

        # compute reward
        reward = self.compute_reward(res)

        # check if done
        done = 0
        current_pos = self.getPosition()
        if current_pos[0] < 110: # 50 for small course / 105 for big
            done = 0
        else:
            done = 1

        return res, reward, done, {}

    def compute_reward(self, img):
        """
        Compute reward based on image received.
        """
        # normalize pixels
        # all white = 0, all black = 1
        reward = 1 - np.sum(img) / (36*64)
        return reward

    def reset(self):
        """
        Take initial camera data.
        """
        # # make sure offboard mode is on
        # self.setOffboardModeTrue()

        # skip going home if "inf mode"
        if self.inf_mode:
            print("Inf Mode: Data saved. Keep going.")

            # takeoff if landed
            if (self.getLandedState() == LandedState.Landed):
                print("Landed.")
                try:
                    self.drone_takeoff()
                except:
                    print("Takeoff failed. Trying again...")
                    # might need to reconnect first
                    CustomAirSim.__init__(self)
                    self.drone_takeoff()

                # arm drone
                if (not self.arm()):
                    print("Failed to arm the drone.")
                    sys.exit(1)

        else:
            # self.arm()
            # # self.drone_takeoff()
            # go home
            if (self.getLandedState() != LandedState.Landed):
                self.drone_gohome()
            else:
                try:
                    self.drone_takeoff()
                except:
                    print("Takeoff failed. Trying again. Reconnecting...")
                    # might need to reconnect first
                    CustomAirSim.__init__(self, self.n_steps)
                    self.drone_takeoff()

            # arm drone
            if (not self.arm()):
                print("Failed to arm the drone.")
                sys.exit(1)

        # get next state
        img = self.grab_depth()
        res = self.preprocess(img)

        return res

    def preprocess(self, img):
        """
        Resize image. Converts and down-samples the input image.
        """
        # resize img
        res = cv2.resize(img,None,fx=self.reduc_factor, fy=self.reduc_factor, interpolation = cv2.INTER_AREA)

        # normalize image
        res = res / 255

        return res

    @property
    def action_space(self):
        """
        Maximum roll command. It can be scaled after.
        """
        return spaces.Box(low=np.array(-1),
                          high=np.array(1))

    @property
    def observation_space(self):
        """
        2D image of depth sensor.
        """
        screen_height = 144
        screen_width = 256
        # frames = 1
        return spaces.Box(low=0, high=255, shape=(screen_height, screen_width))

class AirSimEOM(object):
    """
    Integrates the Equations of Motion (EOM) and fly UAV in AirSim using
    computer vision mode.
    """

    def __init__(self, n_steps, learning_agent=True, inf_mode=False, use_gui=False, reward_signal=0, clip_action=0):
        # initialize airsim
        self.client = AirSimClient('127.0.0.1')
        self.control = 'joystick' # 'keyboard'
        self.inf_mode = inf_mode
        self.use_gui = use_gui
        self.reward_signal = reward_signal
        self.clip_action = clip_action

        # load net that classify actions (CS1)
        self.cs1_model = load_neural('cybersteer_1', {'main_output': 'binary_crossentropy'}, 'adam')
        self.cs1_r_max = 10

        # load net that suggests actions (CS2)
        self.cs2_model = load_neural('cybersteer_2', 'mean_squared_error', 'adam', metrics=['mean_absolute_error'])
        self.cs2_r_max = 10

        # store current and reference states
        # step, roll, pitch, yaw (current), roll, pitch, yaw (ref)
        self.t_step = 0
        # step, 3 pos, 3 vel, 3 att, 3 ref att
        self.hist_attitude = np.zeros((n_steps,10))

        self.act_n = 2
        # depth parameters and compression factors
        # depth 1/4: 36 x 64 pics
        # depth 1/8: 18 x 32 pics
        depth_width = 256
        depth_height = 144
        self.reduc_factor = 1/4 # multiply original size by this factor
        self.depth_w = int(depth_width*self.reduc_factor)
        self.depth_h = int(depth_height*self.reduc_factor)
        self.stacked_frames = 0 # to save multiple image frames

        self.total_crashes = 0
        self.stat = True
        self.V_ref = 0
        self.original_dt = 0.02
        self.scale_dt = 1.0
        self.learning_agent = True # disable or enable learning agent
                                   # if disabled, can use joystick/keyboard to control
        self.max_ore = 0.99 # maximum pitch/roll orientation fraction

        # fix inital position at x=y=0, z=-0.15
        # pos = self.client.getPosition()
        pos = [0,0,-.15]
        orq = [1,0,0,0]
        self.pos0,self.orq0 = (pos,orq)
        self.pos_new,self.orq_new = (pos,orq)
        self.ore0 = self.client.toEulerianAngle(orq)

        print('Initial Position: ', self.pos0)

        self.rll_prv,self.pch_prv,self.yaw_prv,self.thr_prv = (0.0,0.0,0.0,0.0)
        self.drll,self.dpch,self.dyaw,self.dthr = (0.0,0.0,0.0,0.0)
        self.drll_prv,self.dpch_prv,self.dyaw_prv,self.dthr_prv = (0.0,0.0,0.0,0.0)
        self.mode_prv = 0 # 0 = independent axis control, 1 = forward velocity and coordinated turn control

        self.orq_new = [0.0,0.0,0.0,0.0]
        self.ore_new = [0.0,0.0,0.0]
        self.ore_check = [0.0,0.0,0.0]

        self.ps = 0

        self.g = 9.81 # gravity [m/s^2]
        self.m = 0.6 # mass [kg]
        self.Ix,self.Iy,self.Iz = (0.00115,0.00115,0.00598) # inertia tensor principle components [kg m^2] (based on Gremillion 2016 https://arc.aiaa.org/doi/abs/10.2514/1.J054408)
        self.Lp,self.Mq,self.Nr,self.Xu,self.Yv,self.Zw = (-0.01,-0.01,-0.2,-0.05,-0.05,-0.05) # aerodynamic drag derivatives [kg / s]

        self.K_ph,self.K_p = (0.5,-0.02)
        self.K_th,self.K_q = (self.K_ph,self.K_p)
        self.K_r,self.K_dr = (0.2,0.0)
        self.K_dps = 0.2
        self.K_z,self.K_dz,self.K_z_i = (-20.0,5.0,-0.05)
        self.K_dv = 0.5
        self.K_v = 1.0

        # initialize states
        self.ph,self.th,self.ps,self.p,self.q,self.r,self.u,self.v,self.w,self.x,self.y,self.z = (self.ore0[0],self.ore0[1],self.ore0[2],0.0,0.0,0.0,0.0,0.0,0.0,self.pos0[0],self.pos0[1],self.pos0[2])
        # initialize state derivatives
        self.ph_prv,self.th_prv,self.ps_prv,self.p_prv,self.q_prv,self.r_prv,self.u_prv,self.v_prv,self.w_prv,self.x_prv,self.y_prv,self.z_prv = (self.ph,self.th,self.ps,self.p,self.q,self.r,self.u,self.v,self.w,self.x,self.y,self.z)
        # initialize state derivatives
        self.dph,self.dth,self.dps,self.dp,self.dq,self.dr,self.du,self.dv,self.dw,self.dx,self.dy,self.dz = (0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
        # initialize aerodynamic states
        self.tawx,self.tawy,self.tawz,self.fwx,self.fwy,self.fwz = (0.0,0.0,0.0,0.0,0.0,0.0) # wind forces [N] torques [N m]

        self.z_i = 0.0 # z error integral

        # get initial collision status
        collision = self.client.getCollisionInfo()
        self.initial_col1 = collision[1]

        # initialize pygame for inputs
        if self.control == 'joystick':
            pygame.init()
            if pygame.joystick.get_count() != 1:
                self.control = 'keyboard'
            else:
                self.my_joystick = pygame.joystick.Joystick(0)
                self.my_joystick.init()
                self.console = pygame.joystick.Joystick(0).get_name()

        if self.control == 'keyboard':
            pygame.display.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode((500, 120))
            pygame.display.set_caption('KEYBOARD INPUT: Click Here to Control Drone')
            sysfont = pygame.font.SysFont('ubuntu',15)
            rendered = sysfont.render('Longitudinal/Lateral Control = Arrow keys',0,(200,200,200))
            self.screen.blit(rendered,(10,10))
            rendered = sysfont.render('Yaw Control = A/D keys',0,(200,200,200))
            self.screen.blit(rendered,(10,30))
            rendered = sysfont.render('Heave Control = Page Up/Down keys',0,(200,200,200))
            self.screen.blit(rendered,(10,50))
            rendered = sysfont.render('Mode Select = M key',0,(200,200,200))
            self.screen.blit(rendered,(10,70))
            rendered = sysfont.render('Reset = Space key',0,(200,200,200))
            self.screen.blit(rendered,(10,90))
            pygame.display.update()

        # initialize time
        self.previous_time = time.time() + self.original_dt


    def getKeyboardCommands(self,stat,rset,mode,rll,pch,yaw,thr,drll,dpch,dyaw,dthr):

        if pygame.joystick.get_count() != 1:
            self.control = 'keyboard'

        if self.control == 'joystick':
            pygame.event.get()
            # xbox one controller
            if self.console == 'Microsoft X-Box One pad (Firmware 2015)':
                # print("Xbox")
                # control
                drll = self.my_joystick.get_axis(3) # 0 = right stick lat
                dpch = self.my_joystick.get_axis(4) # 0 = right stick long
                dyaw = self.my_joystick.get_axis(0) # 0 = left stick lat
                dthr = self.my_joystick.get_axis(1) # 0 = left stick long

                # options
                if self.my_joystick.get_button(0): # a button
                    mode = 0

                if self.my_joystick.get_button(1): # b button
                    mode = 1

                if  self.my_joystick.get_button(7): # "start" button
                    rset = True # reset position/controls
                    rll,pch,yaw,thr = (0.0,0.0,0.0,0.0)
                    drll,dpch,dyaw,dthr = (0.0,0.0,0.0,0.0)
            # ps3 controller
            elif self.console == 'Sony PLAYSTATION(R)3 Controller':
                # print("PS")
                # control
                drll = self.my_joystick.get_axis(2) # 0 = right stick lat
                dpch = self.my_joystick.get_axis(3) # 0 = right stick long
                dyaw = self.my_joystick.get_axis(0) # 0 = left stick lat
                dthr = self.my_joystick.get_axis(1) # 0 = left stick long

                # options
                if self.my_joystick.get_button(14): # x button
                    mode = 0

                if self.my_joystick.get_button(13): # circle button
                    mode = 1

                if  self.my_joystick.get_button(3): # start button
                    rset = True # reset position/controls
                    rll,pch,yaw,thr = (0.0,0.0,0.0,0.0)
                    drll,dpch,dyaw,dthr = (0.0,0.0,0.0,0.0)
            # kinobo controller
            elif self.console == 'DragonRise Inc.   Generic   USB  Joystick  ':
                # print("PS")
                # control
                drll = self.my_joystick.get_axis(3) # 0 = right stick lat
                dpch = self.my_joystick.get_axis(4) # 0 = right stick long
                dyaw = self.my_joystick.get_axis(2) # 0 = left stick lat
                dthr = self.my_joystick.get_axis(1) # 0 = left stick long

                # options
                if self.my_joystick.get_button(2): # 3 button
                    mode = 0

                if self.my_joystick.get_button(1): # 2 button
                    mode = 1

                if  self.my_joystick.get_button(9): # start button
                    rset = True # reset position/controls
                    rll,pch,yaw,thr = (0.0,0.0,0.0,0.0)
                    drll,dpch,dyaw,dthr = (0.0,0.0,0.0,0.0)
            # no controller -> switch to joystick mode
            else:
                print("ERROR: Joystick not recognized! Run validate_joystick.py to confirm Joystick Name")
                rll,pch,yaw,thr = (0.0,0.0,0.0,0.0)
                drll,dpch,dyaw,dthr = (0.0,0.0,0.0,0.0)
                mode = 0
                rset = 0
                stat = False

            # flush rest and display
            pygame.event.clear()

            rll = drll / 4.0
            pch = dpch / 4.0
            yaw = dyaw / 1.0
            thr += dthr / 80.0

        elif self.control == 'keyboard':
            for event in pygame.event.get():
                if event.type == pygame.KEYUP:
                    # LATERAL COMMANDS
                    if event.key == pygame.K_RIGHT:
                        drll = 0.0
                    if event.key == pygame.K_LEFT:
                        drll = 0.0
                    # LONGITUDINAL COMMANDS
                    if event.key == pygame.K_UP:
                        dpch = 0.0 # speed forward
                    if event.key == pygame.K_DOWN:
                        dpch = 0.0 # break drone
                    # HEAVE COMMANDS
                    if event.key == pygame.K_PAGEUP:
                        dthr = 0.0 # heave up
                    if event.key == pygame.K_PAGEDOWN:
                        dthr = 0.0 # heave down
                    # HEAVE COMMANDS
                    if event.key == pygame.K_a:
                        dyaw = 0.0 # heave up
                    if event.key == pygame.K_d:
                        dyaw = 0.0 # heave down
                if event.type == pygame.KEYDOWN:
                    # LATERAL COMMANDS
                    if event.key == pygame.K_RIGHT:
                        drll = 1.0
                    if event.key == pygame.K_LEFT:
                        drll = -1.0
                    # LONGITUDINAL COMMANDS
                    if event.key == pygame.K_UP:
                        dpch = -1.0 # speed forward
                    if event.key == pygame.K_DOWN:
                        dpch = 1.0 # break drone
                    # HEAVE COMMANDS
                    if event.key == pygame.K_PAGEUP:
                        dthr = -1.0 # heave up
                    if event.key == pygame.K_PAGEDOWN:
                        dthr = 1.0 # heave down
                    # HEAVE COMMANDS
                    if event.key == pygame.K_a:
                        dyaw = -1.0 # yaw left
                    if event.key == pygame.K_d:
                        dyaw = 1.0 # yaw right
                    if event.key == pygame.K_DELETE:
                        stat = False # break drone
                    if event.key == pygame.K_SPACE:
                        rset = True # reset position/controls
                        rll,pch,yaw,thr = (0.0,0.0,0.0,0.0)
                        drll,dpch,dyaw,dthr = (0.0,0.0,0.0,0.0)
                    if event.key == pygame.K_m:
                        mode += 1
                        if mode > 1:
                            mode = 0
            rll += drll / 100.0
            pch += dpch / 100.0
            yaw += dyaw / 100.0
            thr += dthr / 20.0

        else:
            print("ERROR: Missing input method! Set ''control'' to ''keyboard'' or ''joystick''")
            rll,pch,yaw,thr = (0.0,0.0,0.0,0.0)
            drll,dpch,dyaw,dthr = (0.0,0.0,0.0,0.0)
            mode = 0
            rset = 0
            stat = False

        return (stat,rset,mode,rll,pch,yaw,thr,drll,dpch,dyaw,dthr)

    def step(self, action):
        """
        Compute next states based on joystick inputs and defined equations.
        """
        # rl parameters
        info = 0
        # print('got action: ', action)
        if self.clip_action:
            action = np.clip(action,-.5,.5)

        # compute current dt
        current_time = time.time()
        dt = current_time - self.previous_time
        dt = np.clip(dt,0,self.original_dt)

        self.rset = False
        mode = self.mode_prv

        # retrieve states from buffer
        ph,th,ps,p,q,r,u,v,w,x,y,z = (self.ph_prv,self.th_prv,self.ps_prv,self.p_prv,self.q_prv,self.r_prv,self.u_prv,self.v_prv,self.w_prv,self.x_prv,self.y_prv,self.z_prv)

        # retrieve commands from buffer
        rll,pch,yaw,thr = (self.rll_prv,self.pch_prv,self.yaw_prv,self.thr_prv)
        drll,dpch,dyaw,dthr = (self.drll_prv,self.dpch_prv,self.dyaw_prv,self.dthr_prv)

        # get commands from keyboard
        stat,rset,mode,rll,pch,yaw,thr,drll,dpch,dyaw,dthr = self.getKeyboardCommands(self.stat,self.rset,mode,rll,pch,yaw,thr,drll,dpch,dyaw,dthr)
        self.rset = rset

        # REMOVING YAW FOR NOW -> just 2-DOF
        yaw = 0

        # override with learning agent actions
        if self.learning_agent:
            # commands from learning agent
            rll = action[0] / 4
            pch = action[1] / 4

        # zero commands if mode changes
        if mode != self.mode_prv:
            rll,pch,yaw,thr = (0.0,0.0,0.0,self.thr_prv)
            tawx,tawy,tawz,fwx,fwy,fwz = (0.0,0.0,0.0,0.0,0.0,0.0)
            z,ph,th,p,q,r,u,v,w = (self.z_prv,self.ore0[0],self.ore0[1],0.0,0.0,0.0,0.0,0.0,0.0)
            dph,dth,dps,dp,dq,dr,du,dv,dw,dx,dy,dz = (0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
            be = 0.0

        # wind angles
        al = atan2(w,u)
        be = atan2(v,u)

        # control mode case
        if mode == 0:
            # state reference
            ph_ref = 3.0 * rll
            th_ref = 3.0 * pch
            dps_ref = 40.0 * yaw

            # state error
            ph_err = ph_ref - ph
            th_err = th_ref - th
            dps_err = dps_ref - self.dps

            # force/torque control
            tax = 3.0 * self.K_ph * ph_err + 4.0 * self.K_p * p
            tay = 3.0 * self.K_th * th_err + 4.0 * self.K_q * q
            # tax = self.K_ph * ph_err + self.K_p * p
            # tay = self.K_th * th_err + self.K_q * q
            taz = self.K_dps * dps_err
        else:
            # state reference
            ph_ref = 3.0 * rll
            # pch = self.pch_prv + 0.1
            # pch = np.clip(pch,-10.0,10.0)
            self.V_ref = 20.0 * -pch
            dps_ref = 5.0 * be
            r_ref = 40.0 * be
            V = u * cos(th) + w * sin(th)
            V_err = self.V_ref - V
            # th_ref = 1.0 * -V_err
            th_ref = -1.0 + 1.0 * abs(ph) + 0.08 * V
            # th_ref = -0.25 + 4.0 * pch + 1.0 * abs(ph)

            # state error
            ph_err = ph_ref - ph
            th_err = th_ref - th
            dps_err = dps_ref - self.dps
            r_err = r_ref - r

            # print("Mode %f, Vr %f, V %f, pch %f, th_r %f, dps_r %f, dps %f, r_r %f, r %f" %(mode,self.V_ref,V,pch,th_ref,dps_ref,self.dps,r_ref,r))

            # force/torque control
            tax = 3.0 * self.K_ph * ph_err + 4.0 * self.K_p * p
            tay = 3.0 * self.K_th * th_err + 4.0 * self.K_q * q
            # tmp = 2.0
            # tax = self.K_ph * ph_err + tmp * self.K_p * p
            # tay = self.K_th * th_err + tmp * self.K_q * q
            if V <= 0.1:
                taz = 0
            else:
                taz = 2.5 * self.K_dps * dps_err
                taz = 2.5 * self.K_r * r_err

            # taz = 3.0 * self.K_dps * dps_err

        # heave control
        z_ref = 10.0 * thr + self.pos0[2]
        z_err = z_ref - z
        ft = self.K_z * z_err + self.K_dz * self.dz + self.K_z_i * self.z_i + self.m * self.g

        # drag terms
        tawx = self.Lp * p
        tawy = self.Mq * q
        tawz = self.Nr * r
        fwx = self.Xu * u
        fwy = self.Yv * v
        fwz = self.Zw * w

        # nonlinear dynamics
        self.dph = p + r * (cos(ph) * tan(th)) + q * (sin(ph) * tan(th))
        self.dth = q * (cos(ph)) - r * (sin(ph))
        self.dps = r * (cos(ph) / cos(th)) + q * (sin(ph) / cos(th))
        self.dp = r * q * (self.Iy - self.Iz) / self.Ix + (tax + tawx) / self.Ix
        self.dq = p * r * (self.Iz - self.Ix) / self.Iy + (tay + tawy) / self.Iy
        self.dr = p * q * (self.Ix - self.Iy) / self.Iz + (taz + tawz) / self.Iz
        self.du = r * v - q * w - self.g * (sin(th)) + fwx / self.m
        self.dv = p * w - r * u + self.g * (sin(ph) * cos(th)) + fwy / self.m
        self.dw = q * u - p * v + self.g * (cos(th) * cos(ph)) + (fwz - ft) / self.m
        self.dx = w * (sin(ph) * sin(ps) + cos(ph) * cos(ps) * sin(th)) - v * (cos(ph) * sin(ps) - cos(ps) * sin(ph) * sin(th)) + u * (cos(ps) * cos(th))
        self.dy = v * (cos(ph) * cos(ps) + sin(ph) * sin(ps) * sin(th)) - w * (cos(ps) * sin(ph) - cos(ph) * sin(ps) * sin(th)) + u * (cos(th) * sin(ps))
        self.dz = w * (cos(ph) * cos(th)) - u * (sin(th)) + v * (cos(th) * sin(ph))

        # numerically integrate states
        ph += self.dph * dt
        th += self.dth * dt
        ps += self.dps * dt
        p += self.dp * dt
        q += self.dq * dt
        r += self.dr * dt
        u += self.du * dt
        v += self.dv * dt
        w += self.dw * dt
        x += self.dx * dt
        y += self.dy * dt
        z += self.dz * dt

        ## GENERATE POSITION/ORIENTATION VISUALIZATION
        # update position/orientation
        ore_new = [ph,th,ps] # new orientation (Euler)
        orq_new = self.client.toQuaternion(ore_new) # convert to quaternion
        pos_new = [x,y,z] # new position

        # convert back to Euler (check quaternion conversion)
        ore_check = self.client.toEulerianAngle(orq_new)

        if self.rset == True:
            done = 1

        # get vehicle status
        pos = self.client.getPosition() # get current position
        orq = self.client.getOrientation() # get current orientation
        collision = self.client.getCollisionInfo() # get collision status

        # tryin gto fix fake collisions at the beginning
        if self.t_step < 5:
            self.initial_col1 = collision[1]

        if (collision[1] != self.initial_col1): # if collision, reset position/orientation/controls
            print("COLLISION")
            reward = -100
            done = 1
            self.total_crashes += 1
            print('** total crashes: ', self.total_crashes)

        elif self.rset == True:
            print("RESET")
            reward = 0
            done = 1

        elif (ore_new[0] > self.max_ore * np.pi/2):
            print("MAX ROLL")
            reward = -100
            done = 1
            self.total_crashes += 1
            print('** total crashes: ', self.total_crashes)

        elif (ore_new[1] > self.max_ore * np.pi/2):
            print("MAX PITCH")
            reward = -100
            done = 1
            self.total_crashes += 1
            print('** total crashes: ', self.total_crashes)

        else: # if no collision update position/orientation
            # print(pos_new,orq_new)
            self.client.simSetPose(pos_new,orq_new)
            reward = 20*(pos_new[0]-pos[0]) # positive if going forward
            done = 0

        # z error integral
        self.z_i += z_err * dt

        # send states to buffer
        self.ph_prv,self.th_prv,self.ps_prv,self.p_prv,self.q_prv,self.r_prv,self.u_prv,self.v_prv,self.w_prv,self.x_prv,self.y_prv,self.z_prv = (ph,th,ps,p,q,r,u,v,w,x,y,z)

        # send commands to buffer
        self.rll_prv,self.pch_prv,self.yaw_prv,self.thr_prv = (rll,pch,yaw,thr)
        self.drll_prv,self.dpch_prv,self.dyaw_prv,self.dthr_prv = (drll,dpch,dyaw,dthr)
        self.mode_prv = mode

        # print debug output command/orientation
        # print("phr %f, dpsr %f, be %f, V %f, Vref %f, roll %f, phi %f, pitch %f, theta %f, yaw %f, r %f, throttle %f, z %f" %(ph_ref,dps_ref,be,V,V_ref,rll,ph,pch,th,yaw,r,thr,z))
        # print("mode %f, roll %f, pitch %f, yaw %f, throttle %f" %(mode,rll,pch,yaw,thr))

        self.previous_time = current_time

        # get next state
        img = self.grab_depth()
        res = self.preprocess(img)

        # check if out of bounds
        if pos[0] < -2:
            # went backward on the beginning
            print('[*] Backwards!')
            reward = -100
            done = 1

        if pos[0] > 90:
            # completed track
            print('[*] Completed Track!')
            reward = 100
            done = 1

        # save ref controls and attitude
        curr_pos = self.client.getPosition()
        curr_vel = self.client.getVelocity()
        curr_att = np.rad2deg(self.client.getRollPitchYaw())

        self.hist_attitude[self.t_step,:] = [self.t_step,
                                             curr_pos[0],
                                             curr_pos[1],
                                             curr_pos[2],
                                             curr_vel[0],
                                             curr_vel[1],
                                             curr_vel[2],
                                             curr_att[1],
                                             -curr_att[0],
                                             curr_att[2]]

        # update reward signal based on cybersteer
        if self.reward_signal != 0: # not standard
            reward += self.compute_alt_reward(action)

        return res, reward, done, info

    def compute_alt_reward(self, action):
        """
        Compute alternative reward based on CS architecture.
        """
        if self.reward_signal == 1:
            # classify if action was human-like
            pred = self.cs1_model.predict({'main_input': self.stacked_frames.reshape(1,36,64,3),
                                           'aux_input': action.reshape(1,2)})

            reward = self.cs1_r_max*(2*pred[0] - 1)
            reward = reward[0]

        elif self.reward_signal == 2:
            # check how close it was from the behavior cloning network
            pred = self.cs2_model.predict(self.stacked_frames.reshape(1,36,64,3))
            a_mse = ((pred[0] - action) ** 2).mean(axis=0)
            reward = self.cs2_r_max*(1 - np.sqrt(a_mse))

        return reward

    def reset(self):
        """
        Go to initial position and restart simulation.
        """
        random_pos0 = self.pos0
        random_pos0[1] = np.random.uniform(-14,14) # random at y
        print('[*] Respawn at x = {}, y = {}, z = {}'.format(random_pos0[0],random_pos0[1],random_pos0[2]))

        self.client.simSetPose(random_pos0,self.orq0) # reset to origin
        pos_new,orq_new = (self.pos0,self.orq0) # update current position/orientation to origin
        rll,pch,yaw,thr = (0.0,0.0,0.0,0.0)
        ph,th,ps,p,q,r,u,v,w,x,y,z = (self.ore0[0],self.ore0[1],self.ore0[2],0.0,0.0,0.0,0.0,0.0,0.0,self.pos0[0],self.pos0[1],self.pos0[2])
        self.ph_prv,self.th_prv,self.ps_prv,self.p_prv,self.q_prv,self.r_prv,self.u_prv,self.v_prv,self.w_prv,self.x_prv,self.y_prv,self.z_prv = (ph,th,ps,p,q,r,u,v,w,x,y,z)
        self.z_i = 0.0
        self.V_ref = 0

        # get initial collision status
        collision = self.client.getCollisionInfo()
        self.initial_col1 = collision[1]

        # get next state
        img = self.grab_depth()
        res = self.preprocess(img)

        return res

    def act(self):
        """
        Temporary function to simulate a learning agent.
        In this case, just do random actions.
        """
        # take random pitch and roll actions
        drll = np.random.uniform(-1,1)
        dpch = np.random.uniform(-1,1)

        action = (drll,dpch)
        # print(action)

        return action

    def preprocess(self, img):
        """
        Resize image. Converts and down-samples the input image.
        """
        # resize img
        res = cv2.resize(img,None,fx=self.reduc_factor, fy=self.reduc_factor, interpolation = cv2.INTER_AREA)

        # normalize image
        res = res / 255

        return res

    def grab_depth(self):
        """
        Get camera depth image and return array of pixel values.
        Returns numpy ndarray.
        """
        # get depth image
        # result = self.setImageTypeForCamera(0, AirSimImageType.Depth)
        result = self.client.simGetImage(0, AirSimImageType.Depth)
        if (result != "\0"):
            # depth
            rawImage = np.fromstring(result, np.int8)
            png = cv2.imdecode(rawImage, cv2.IMREAD_UNCHANGED)
            if png is not None:
                # return pic, only first channel is enough for depth
                # apply threshold
                # png[:,:,0] = self.tsh_distance(100,png[:,:,0])
                return png[:,:,0]
            else:
                print('Couldnt take one depth pic.')
                return np.zeros((144,256)) # empty picture
