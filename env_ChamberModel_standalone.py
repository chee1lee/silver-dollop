# Refer : https://www.mikulskibartosz.name/how-to-create-an-environment-for-a-tensorflow-agent/
# Refer : https://www.tensorflow.org/agents/tutorials/2_environments_tutorial
# install: pip --user tf-agents
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from FabChamberModel_standalone import FabModel

tf.compat.v1.enable_v2_behavior()

logging.basicConfig(format='%(asctime)s L[%(lineno)d] %(message)s ', level=logging.DEBUG)


class EnvChamberModel(py_environment.PyEnvironment):

    def __init__(self, wafer=20, discount=0.9):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.int32, minimum=0, maximum=21, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(13,), dtype=np.int32, minimum=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            maximum=[1, 99, 1, 99, 1, 99, 1, 99, 1, 99, 1, 99, 1], name='observation')
        # self._current_time_step = None
        self._episode_ended = False
        self.model = FabModel(wafer)
        self.discount_ratio = discount

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def current_time_step(self):
        return self._current_time_step

    def _reset(self):
        self._episode_ended = False
        self.model.reset()
        # observation = [state, reward, done]
        observation = self.model.get_observation()
        obs_status = np.asarray(observation[0])
        logging.debug("state: %s:", observation)
        return ts.restart(obs_status)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()
        observation, reward, done = self._env_step(action)
        if done:
            self._episode_ended = True
            return ts.termination(np.asarray(observation), reward=np.asarray(reward))
        else:
            return ts.transition(np.asarray(observation), reward=np.asarray(reward), discount=self.discount_ratio)

    def _env_step(self, action):
        return self.model.step(int(action[0]))
