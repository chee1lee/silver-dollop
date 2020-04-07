# Refer: https://github.com/tensorflow/agents/blob/master/docs/tutorials/1_dqn_tutorial.ipynb

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

# from evn_chamberModel import EnvChamberModel
from env_ChamberModel_standalone import EnvChamberModel

tf.compat.v1.enable_v2_behavior()

#################
# RL environment Setup#
#################
# FabModel = EnvChamberModel()
FabModel = EnvChamberModel(wafer=10, discount=0.99)
FabModel_1 = EnvChamberModel(wafer=10, discount=0.99)

utils.validate_py_environment(FabModel, episodes=5)
train_tf_env = tf_py_environment.TFPyEnvironment(FabModel)
eval_tf_env = tf_py_environment.TFPyEnvironment(FabModel_1)
print('Obseravtion Spec:')
print(train_tf_env.observation_spec())
# print('Reward Spec:')
# print(tf_env_FabModel.time_step_spec().reward)
print('Action Spec:')
print(train_tf_env.action_spec())

#################
# DQN Agent Setup#
#################

# Hyperparameters
num_iterations = 100000  # @param {type:"integer"}
initial_collect_steps = 100  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_max_length = 1000  # @param {type:"integer"}
batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}
num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}

# Building Q-Network
fc_layer_params = (64, 64, 64)  # No. of Hidden layers
q_net = q_network.QNetwork(train_tf_env.observation_spec(),
                           train_tf_env.action_spec(),
                           preprocessing_layers=None,
                           preprocessing_combiner=None,
                           conv_layer_params=None,
                           fc_layer_params=fc_layer_params,
                           activation_fn=tf.keras.activations.relu,
                           kernel_initializer=None,
                           batch_squash=True,
                           dtype=tf.float32,
                           name='QNetwork'
                           )
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(train_tf_env.time_step_spec(),
                           train_tf_env.action_spec(),
                           q_network=q_net,
                           epsilon_greedy=0.9,
                           observation_and_action_constraint_splitter=None,
                           # boltzmann_temperature= 0.5,
                           # emit_log_probability= True,
                           gamma=0.9,
                           optimizer=optimizer,
                           target_update_period=1000,
                           td_errors_loss_fn=common.element_wise_squared_loss,
                           train_step_counter=train_step_counter,
                           summarize_grads_and_vars=True,
                           debug_summaries=True)
agent.initialize()
##############
# Policy setup#
##############
# Todo: Design random policy
'''
eval_policy = agent.policy
collect_policy = agent.collect_policy
random_policy = random_tf_policy.RandomTFPolicy(train_tf_env.time_step_spec(),
                                                train_tf_env.action_spec())

policy = epsilon_greedy_policy.EpsilonGreedyPolicy(policy=random_policy,
                                                   epsilon= 0.3,
                                                   name='epsilon_greedy')
FabModel_2 = EnvChamberModel(wafer=10, discount=0.99)
exam_tf_env = tf_py_environment.TFPyEnvironment(FabModel_2)
time_step = exam_tf_env.reset()
policy.action(time_step)
'''

##########################
# Metrics and evaluation #
##########################
def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


################
# Driver setup #
################
'''
#Driver implementation
num_episodes = tf_metrics.NumberOfEpisodes()
env_steps = tf_metrics.EnvironmentSteps()
observers = [num_episodes, env_steps]
driver = dynamic_episode_driver.DynamicEpisodeDriver(
    env=train_tf_env,
    policy=agent.policy,
    observers=observers,
    num_episodes=num_eval_episodes)
final_time_step, policy_state = driver.run()

'''
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_tf_env.batch_size,
    max_length=replay_buffer_max_length)

###################
# Data Collection #
###################
def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    # Add trajectory to the replay buffer
    buffer.add_batch(traj)


def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)


collect_data(train_tf_env, agent.policy, replay_buffer, steps=100)

# This loop is so common in RL, that we provide standard implementations.
# For more details see the drivers module.
# https://github.com/tensorflow/agents/blob/master/tf_agents/docs/python/tf_agents/drivers.md
# Dataset generates trajectories with shape [Bx2x...]
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)

iterator = iter(dataset)
print(iterator)

######################
# Training the agent #
######################

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_tf_env, agent.policy, num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):
    # Collect a few steps using collect_policy and save to the replay buffer.
    for _ in range(collect_steps_per_iteration):
        collect_step(train_tf_env, agent.collect_policy, replay_buffer)

    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    train_loss = agent.train(experience).loss

    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss))

    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_tf_env, agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)