import glob
import sys

import numpy as np
from FabChamberModel_standalone import FabModel
import tensorflow as tf


def discard_invalid_action(q_values):
    valid_action_mask = env.get_valid_action_mask()
    for i in range(valid_action_mask.__len__()):
        if valid_action_mask[i] == 0:
            q_values[0][i] = np.NINF


if __name__ == "__main__":
    num_wafers = int(sys.argv[1])
    # num_wafers = 10

    result_file = open('test_' + str(num_wafers) + '.txt', 'w')
    env = FabModel(wafer_number=num_wafers)

    for file in glob.glob("*.h5"):
        model = tf.keras.models.load_model(file)

        env.reset()
        obs, _, _ = env.get_observation()

        num_steps = 0
        done = False
        while not done:
            q_values = model.predict(obs.reshape(1, -1))
            discard_invalid_action(q_values)

            action_chosen = np.argmax(q_values[0])

            obs, _, done = env.step(action_chosen)
            num_steps += 1

            if num_steps > num_wafers * 1000:
                break

        if env.airlock[1].store.items.__len__() < env.wafer_number:
            done = False

        if done is True:
            print('{0}\t{1}'.format(file, env.env.now))
            result_file.write('{0}\t{1}\n'.format(file, env.env.now))
        else:
            print('{0}\tfail.'.format(file))
            result_file.write('{0}\tfail\n'.format(file))
    result_file.close()


