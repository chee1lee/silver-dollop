# Refer : https://www.mikulskibartosz.name/how-to-create-an-environment-for-a-tensorflow-agent/
# Refer : https://www.tensorflow.org/agents/tutorials/2_environments_tutorial
# install: pip --user tf-agents
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import socket

import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.WARNING)


class EnvChamberModel(py_environment.PyEnvironment):

    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.int32, minimum=0, maximum=21, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(13,), dtype=np.int32, minimum=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            maximum=[1, 99, 1, 99, 1, 99, 1, 99, 1, 1, 1, 1, 1], name='observation')
        # self._current_time_step = None
        self._episode_ended = False
        self._Host = 'localhost'
        self._Port = 8080
        self._client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._connect()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def current_time_step(self):
        return self._current_time_step

    def _reset(self):
        self._episode_ended = False
        rcv = self._client.recv(1024).decode()
        logging.debug('rcvd: %s', rcv)
        parsed = rcv.split(' ')
        self._client.send('reset'.encode())

        observation = self._string_to_nparray(parsed[0])
        logging.debug("state: %r:", observation)
        return ts.restart(observation)

    def _step(self, action):
        if self._episode_ended:
            return self.reset(self)
        observation, reward, done = self._env_step(action)
        if done:
            return ts.termination(observation, reward=reward)
        else:
            return ts.transition(observation, reward=reward, discount=0.99)

    def _env_step(self, action):
        value = action[0]
        logging.debug("step send: %s", value.astype(str))
        self._client.send(value.astype(str).encode())
        received = self._client.recv(1024).decode()
        logging.debug("step rcvd: %s", received)
        parsed = received.split(' ')
        observation = self._string_to_nparray(parsed[0])
        return observation, int(parsed[1]), parsed[2] == 'True'

    def _connect(self):
        self._client.connect((self._Host, self._Port))
        rcv = self._client.recv(1024)
        logging.debug('cnt rcvd: %s', rcv.decode())
        self._client.send("reset".encode())
        logging.debug('cnt sent: reset')

    def close(self):
        self._client('terminate'.encode())
        self._client.close()

    def _string_to_nparray(self, given):
        status = given.split('|')
        return np.array([int(i) for i in status], np.int)
