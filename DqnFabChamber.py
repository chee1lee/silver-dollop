# import FabChamberModel as fc
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
# import tensorflow as tf
import random
import numpy as np
from argparse import ArgumentParser
import socket
import time

MAX_SCORE_QUEUE_SIZE = 100  # number of episode scores to calculate average performance


def get_options():
    parser = ArgumentParser()
    parser.add_argument('--MAX_EPISODE', type=int, default=400000,
                        help='max number of episodes iteration')
    parser.add_argument('--ACTION_DIM', type=int, default=22,
                        help='number of actions one can take')
    parser.add_argument('--OBSERVATION_DIM', type=int, default=14,
                        help='number of observations one can see')
    parser.add_argument('--GAMMA', type=float, default=0.9,
                        help='discount factor of Q learning')
    parser.add_argument('--INIT_EPS', type=float, default=0.5,
                        help='initial probability for randomly sampling action')
    parser.add_argument('--FINAL_EPS', type=float, default=1e-3,
                        help='finial probability for randomly sampling action')
    parser.add_argument('--EPS_DECAY', type=float, default=0.99,
                        help='epsilon decay rate')
    parser.add_argument('--EPS_ANNEAL_STEPS', type=int, default=100,
                        help='steps interval to decay epsilon')
    parser.add_argument('--LR', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--MAX_EXPERIENCE', type=int, default=30000,
                        help='size of experience replay memory')
    parser.add_argument('--BATCH_SIZE', type=int, default=256,
                        help='mini batch size'),
    parser.add_argument('--H1_SIZE', type=int, default=128,
                        help='size of hidden layer 1')
    parser.add_argument('--H2_SIZE', type=int, default=128,
                        help='size of hidden layer 2')
    parser.add_argument('--H3_SIZE', type=int, default=128,
                        help='size of hidden layer 3')
    options = parser.parse_args()
    return options


'''
The DQN model itself.
Remain unchanged when applied to different problems.
'''


class QAgent:

    # A naive neural network with 3 hidden layers and relu as non-linear function.
    def __init__(self, options):
        self.W1 = self.weight_variable([options.OBSERVATION_DIM, options.H1_SIZE])
        self.b1 = self.bias_variable([options.H1_SIZE])
        self.W2 = self.weight_variable([options.H1_SIZE, options.H2_SIZE])
        self.b2 = self.bias_variable([options.H2_SIZE])
        self.W3 = self.weight_variable([options.H2_SIZE, options.H3_SIZE])
        self.b3 = self.bias_variable([options.H3_SIZE])
        self.W4 = self.weight_variable([options.H3_SIZE, options.ACTION_DIM])
        self.b4 = self.bias_variable([options.ACTION_DIM])

    # Weights initializer
    def xavier_initializer(self, shape):
        dim_sum = np.sum(shape)
        if len(shape) == 1:
            dim_sum += 1
        bound = np.sqrt(6.0 / dim_sum)
        return tf.random.uniform(shape, minval=-bound, maxval=bound)

    # Tool function to create weight variables
    def weight_variable(self, shape):
        return tf.Variable(self.xavier_initializer(shape))

    # Tool function to create bias variables
    def bias_variable(self, shape):
        return tf.Variable(self.xavier_initializer(shape))

    # Add options to graph
    def add_value_net(self, options):
        observation = tf.placeholder(tf.float32, [None, options.OBSERVATION_DIM])
        # observation = tf.Variable(tf.ones(shape=[None, options.OBSERVATION_DIM]), dtype=tf.float32)
        h1 = tf.nn.relu(tf.matmul(observation, self.W1) + self.b1)
        h2 = tf.nn.relu(tf.matmul(h1, self.W2) + self.b2)
        h3 = tf.nn.relu(tf.matmul(h2, self.W3) + self.b3)
        Q = tf.squeeze(tf.matmul(h3, self.W4) + self.b4)
        return observation, Q

    # Sample action with random rate eps
    def sample_action(self, Q, feed, eps, options):
        act_values = Q.eval(feed_dict=feed)

        if random.random() <= eps:
            # action_index = env.action_space.sample()
            action_index = random.randrange(options.ACTION_DIM)
            # print('Random choose: ', action_index)
        else:
            action_index = np.argmax(act_values)
            # print('Q-value: ', np.round(act_values, decimals=2))
        action = np.zeros(options.ACTION_DIM)
        action[action_index] = 1
        return action


def string_to_nparray(given):
    status = given.split('|')
    return np.array([int(i) for i in status], np.float64)


def env_reset():
    client.send('reset'.encode())
    received = client.recv(1024).decode()

    parsed = received.split(' ')
    observation = string_to_nparray(parsed[0])
    return observation


def env_step(action):
    client.send(str(action).encode())
    received = client.recv(1024).decode()

    parsed = received.split(' ')
    observation = string_to_nparray(parsed[0])
    return observation, int(parsed[1]), parsed[2] == 'True'


def train(TARGET_REWARD):
    # Define placeholders to catch inputs and add options
    global time_begin
    options = get_options()
    agent = QAgent(options)
    sess = tf.compat.v1.InteractiveSession()

    obs, Q1 = agent.add_value_net(options)
    act = tf.placeholder(tf.float32, [None, options.ACTION_DIM])
    # act = tf.Variable(tf.ones(shape=[None, options.ACTION_DIM]), dtype=tf.float32)
    rwd = tf.placeholder(tf.float32, [None, ])
    # rwd = tf.Variable(tf.ones(shape=[None, None]), dtype=tf.float32)
    next_obs, Q2 = agent.add_value_net(options)

    values1 = tf.reduce_sum(tf.multiply(Q1, act), reduction_indices=1)
    values2 = rwd + options.GAMMA * tf.reduce_max(Q2, reduction_indices=1)
    loss = tf.reduce_mean(tf.square(values1 - values2))
    train_step = tf.train.AdamOptimizer(options.LR).minimize(loss)

    sess.run(tf.initialize_all_variables())

    # saving and loading networks
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state("checkpoints-DQN_FabChamberModel")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # Some initial local variables
    feed = {}
    eps = options.INIT_EPS
    global_step = 0
    exp_pointer = 0
    learning_finished = False

    # The replay memory
    obs_queue = np.empty([options.MAX_EXPERIENCE, options.OBSERVATION_DIM])
    act_queue = np.empty([options.MAX_EXPERIENCE, options.ACTION_DIM])
    rwd_queue = np.empty([options.MAX_EXPERIENCE])
    next_obs_queue = np.empty([options.MAX_EXPERIENCE, options.OBSERVATION_DIM])

    # Score cache
    score_queue = []

    # The episode loop
    for i_episode in range(options.MAX_EPISODE):

        observation = env_reset()
        done = False
        score = 0
        sum_loss_value = 0

        # The step loop
        while not done:
            global_step += 1
            if global_step % options.EPS_ANNEAL_STEPS == 0 and eps > options.FINAL_EPS:
                eps = eps * options.EPS_DECAY
            # env.render()

            obs_queue[exp_pointer] = observation
            action = agent.sample_action(Q1, {obs: np.reshape(observation, (1, -1))}, eps, options)
            act_queue[exp_pointer] = action

            action_index = np.argmax(action)
            observation, reward, done = env_step(action_index)

            print('action: ', action_index, ', reward: ', reward)

            score += reward
            reward = score  # Reward will be the accumulative score

            if done and score < TARGET_REWARD:
                reward = TARGET_REWARD * (-2.5)  # If it fails, punish hard
                observation = np.zeros_like(observation)

            rwd_queue[exp_pointer] = reward
            next_obs_queue[exp_pointer] = observation

            exp_pointer += 1
            if exp_pointer == options.MAX_EXPERIENCE:
                exp_pointer = 0  # Refill the replay memory if it is full

            if global_step >= options.MAX_EXPERIENCE:
                rand_indexs = np.random.choice(options.MAX_EXPERIENCE, options.BATCH_SIZE)
                feed.update({obs: obs_queue[rand_indexs]})
                feed.update({act: act_queue[rand_indexs]})
                feed.update({rwd: rwd_queue[rand_indexs]})
                feed.update({next_obs: next_obs_queue[rand_indexs]})
                if not learning_finished:  # If not solved, we train and get the step loss
                    step_loss_value, _ = sess.run([loss, train_step], feed_dict=feed)
                else:  # If solved, we just get the step loss
                    step_loss_value = sess.run(loss, feed_dict=feed)
                # Use sum to calculate average loss of this episode
                sum_loss_value += step_loss_value

        print(
            "{0:7.1f}====== Episode {1} ended with score = {2}, avg_loss = {3} ======".format(time.time() - time_begin,
                                                                                              i_episode + 1, score,
                                                                                              sum_loss_value / score))
        score_queue.append(score)
        if len(score_queue) > MAX_SCORE_QUEUE_SIZE:
            score_queue.pop(0)
            if np.mean(score_queue) > TARGET_REWARD * 0.975:  # The threshold of being solved
                learning_finished = True
            else:
                learning_finished = False
        if learning_finished:
            print("Testing !!!")
        # save progress every 100 episodes
        if learning_finished and i_episode % 100 == 0:
            saver.save(sess, 'checkpoints-DQN_FabChamberModel', global_step=global_step)


if __name__ == "__main__":
    # env = gym.make(GAME)
    # Monitor(env, './test/', force=True)

    Host = 'localhost'
    Port = 8080

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((Host, Port))

    rcv = client.recv(1024)
    print(rcv.decode())

    target_reward = 1010
    time_begin = time.time()
    train(target_reward)

    client.send('terminate'.encode())
    client.close()
