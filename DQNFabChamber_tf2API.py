# Refer: https://github.com/tensorflow/agents/blob/master/docs/tutorials/1_dqn_tutorial.ipynb

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.utils import common

from evn_chamberModel import EnvChamberModel

tf.compat.v1.enable_v2_behavior()

#################
# RL environment Setup#
#################
FabModel = EnvChamberModel()
utils.validate_py_environment(FabModel, episodes=5)
tf_env_FabModel = tf_py_environment.TFPyEnvironment(FabModel)
print('Oberavtion Spec:')
print(tf_env_FabModel.observation_spec())
# print('Reward Spec:')
# print(tf_env_FabModel.time_step_spec().reward)
print('Action Spec:')
print(tf_env_FabModel.action_spec())

#################
# DQN Agent Setup#
#################

# Hyperparameters
num_iterations = 20000  # @param {type:"integer"}
initial_collect_steps = 1000  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}
batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}
num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}

# Building Q-Network
fc_layer_params = (64, 64, 64)  # No. of Hidden layers
q_net = q_network.QNetwork(tf_env_FabModel.observation_spec(),
                           tf_env_FabModel.action_spec(),
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
agent = dqn_agent.DqnAgent(tf_env_FabModel.time_step_spec(),
                           tf_env_FabModel.action_spec(),
                           q_network=q_net,
                           optimizer=optimizer,
                           td_errors_loss_fn=common.element_wise_squared_loss,
                           train_step_counter=train_step_counter)
agent.initialize()
##############
# Policy setup#
##############
# Todo: Design random policy

##############
# Driver setup#
##############
num_episodes = tf_metrics.NumberOfEpisodes()
env_steps = tf_metrics.EnvironmentSteps()
observers = [num_episodes, env_steps]
driver = dynamic_episode_driver.DynamicEpisodeDriver(
    env=tf_env_FabModel,
    policy=agent.policy,
    observers=observers,
    num_episodes=num_eval_episodes)

final_time_step, policy_state = driver.run()
