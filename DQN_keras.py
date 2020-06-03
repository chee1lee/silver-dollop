from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
'''
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils.common import function
from tf_agents.metrics import tf_metrics
from tf_agents.eval.metric_utils import log_metrics
from tf_agents.utils import common
from env_ChamberModel_standalone import EnvChamberModel
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies.random_tf_policy import RandomTFPolicy
'''
from FabChamberModel_standalone import FabModel
from collections import deque
from scipy import stats
import random


def get_run_logdir(prefix):
    import time
    run_id = time.strftime(prefix + "_%Y_%m_%d-%H_%M")
    return os.path.join(root_logdir, run_id)


def epsilon_greedy_policy(obs, epsilon=0, valid_action_mask=None):
    action_chosen = 0
    if valid_action_mask is None:
        valid_action_mask = np.ones(n_actions, dtype=int)

    if np.random.rand() < epsilon:
        while True:
            rand_index = random.randint(0, n_actions-1)
            if valid_action_mask[rand_index] == 1:
                action_chosen = rand_index
                break
    else:
        Q_values = model.predict(obs.reshape(1,-1))
        for i in range(n_actions):
            if valid_action_mask[i] == 0:
                Q_values[0][i] = np.NINF
        action_chosen = np.argmax(Q_values[0])
    return action_chosen


def get_rnd_indices_by_action(act_queue):
    act_prob = np.zeros(replay_buffer_size, np.float)
    act_unique, act_count = np.unique(act_queue, return_counts=True)
    num_bins = act_count.shape
    act_freq = stats.relfreq(act_queue, numbins=22)
    # if random.random() > 0.99:
    # print(act_freq)
    cal_value = np.zeros(22)
    for i in range(22):
        if act_freq.frequency[i] == 0:
            act_count = np.insert(act_count, i, 0)
            cal_value[i] = 0
        else:
            cal_value[i] = act_freq.frequency[i] / act_count[i]

    for i in range(act_queue.__len__()):
        sel_act = act_queue[i]
        act_prob[i] = cal_value[sel_act]
    #print('sum:', np.sum(act_prob))
    prob_list = np.asarray(act_prob).flatten()
    rand_indices = np.random.choice(replay_buffer_size, batch_size, p=prob_list)
    return rand_indices


def sample_experiences():
    # rand_indices = get_rnd_indices_by_action(np.array([item[1] for item in replay_buffer]))
    # batch = [replay_buffer[index] for index in rand_indices]
    # use all candidates
    batch = replay_buffer

    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch]) for field_index in range(5)]
    return states, actions, rewards, next_states, dones


def play_one_step(state, epsilon):
    action = epsilon_greedy_policy(state, epsilon, valid_action_mask=env.get_valid_action_mask())
    next_state, reward, done = env.step(action)
    replay_buffer.append((state, action, reward, next_state, done))
    return next_state, reward, done


def training_step(episode):
    experiences = sample_experiences()
    states, actions, rewards, next_states, dones = experiences
    #next_Q_values = model.predict(next_states)
    next_Q_values = target.predict(next_states)
    max_next_Q_values = np.max(next_Q_values, axis=1)
    target_Q_values = (rewards + (1-dones) * discount_factor * max_next_Q_values)
    mask = tf.one_hot(actions, n_actions)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims= True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    with run_summary_writer.as_default():
        tf.summary.scalar("Loss", loss, step=episode)


def set_hyper_parameters():
    global batch_size, H1, H2, H3, observation_shape, n_actions, replay_buffer_size, discount_factor, learning_rate, episode_length
    ###################
    # HyperParameters #
    ###################
    batch_size = 1000
    H1, H2, H3 = 128, 128, 128
    observation_shape = [14]
    n_actions = 22
    replay_buffer_size = 1000
    discount_factor = 0.99
    learning_rate = 1e-3
    episode_length = 80000


def create_model(load_filename=""):
    if load_filename == "":
        model = keras.models.Sequential([
            # keras.layers.InputLayer(input_shape= observation_shape),
            keras.layers.Dense(H1, activation='relu', input_shape=observation_shape),
            keras.layers.Dense(H2, activation='relu'),
            keras.layers.Dense(H3, activation='relu'),
            # keras.layers.Dense(H4, activation='relu'),
            keras.layers.Dense(n_actions)
        ])
        model.compile(loss='mean_squared_error', optimizer='adam')
    else:
        model = keras.models.load_model(load_filename)
        print('loading the model completed: ', load_filename)

    # The replay memory (observation, action, reward, next_states, done
    replay_buffer = deque(maxlen=replay_buffer_size)
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    loss_fn = keras.losses.mean_squared_error

    target = keras.models.clone_model(model)
    target.set_weights(model.get_weights())

    return model, target, replay_buffer, optimizer, loss_fn


if __name__ == "__main__":
    set_hyper_parameters()

    root_logdir = os.path.join(os.curdir, "DQN_logs")

    experiment_name = "N128x3_F_eps_waferT_swing_3"
    run_logdir = get_run_logdir(experiment_name)  # e.g., './my_logs/run_2019_06_07-15_15_22'
    run_summary_writer = tf.summary.create_file_writer(run_logdir)


    env = FabModel(wafer_number=10)

    model, target, replay_buffer, optimizer, loss_fn = create_model()
    # model, target, replay_buffer, optimizer, loss_fn = create_model(experiment_name + ".h5")

    for episode in range(episode_length):
        env.reset()
        obs, _, _ = env.get_observation()
        reward_sum = 0
        for step in range(replay_buffer_size):
            epsilon = max(1.0 - episode / 35000, 0.001)
            # epsilon = 0.001
            obs, reward, done = play_one_step(obs, epsilon)
            reward_sum += reward
            if done:
                with run_summary_writer.as_default():
                    tf.summary.scalar('Reward', reward_sum, step=episode)
                    tf.summary.scalar('#produced wafers', env.airlock[1].store.items.__len__(), step=episode)
                    if env.airlock[1].store.items.__len__() == env.wafer_number:
                        tf.summary.scalar('makespan', env.env.now, step=episode)
                    tf.summary.scalar('eps', epsilon, step=episode)
                reward_sum = 0
                break
            if replay_buffer.__len__() == replay_buffer_size:
                training_step(episode)
                replay_buffer.clear()
                break
        # Update Target Model
        if episode % 100 == 0:
            target.set_weights(model.get_weights())
        print('\rprogress {0}/{1} episodes'.format(episode, episode_length), end=' ')
        if episode % 10000 == 0:
            model.save(experiment_name + '_' + str(episode) + '.h5')

    # model.save(save_model_filename + ".h5")
