import logging
import random

import numpy as np
import simpy

# for import plotly, Dash should be installed.
#    pip install dash==1.7.0
#    pip install numpy
# FIRST_DATE = date(2020, 1, 1)
logging.basicConfig(format='%(asctime)s L[%(lineno)d] %(message)s ', level=logging.WARNING)


# logging.basicConfig(format='%(asctime)s L[%(lineno)d] %(message)s ', level=logging.DEBUG)

# manages chamber wafer_state and reward information.
class chamber_profiler(object):
    def __init__(self, chambers_name, tot_wafer):
        self.reward = 0
        self.total_wafer = tot_wafer
        self.state = 0
        self.ch_names = chambers_name
        # It indicates whether wafer in producer slot is existed or not.
        self.entry_wafer = 0
        self.exit_wafer = 0
        self.prev_exit_wafer = 0
        self.robotarm = [0, 0]
        self.armtime = [0, 0]
        self.status_values = list()
        for name in self.ch_names:
            self.status_values.append({'name': name, 'cnt': 0, 'time_remaining': 0})

    def update_chamber_status(self, target, wafer, time_left):
        for item in self.status_values:
            if target == item['name']:
                item['cnt'] = wafer
                if wafer != 0:
                    item['time_remaining'] = time_left

    def update_entry_exit_status(self, entry_cnt, exit_cnt):
        self.entry_wafer = entry_cnt
        self.exit_wafer = exit_cnt

    def update_arm_status(self, cnt1, time1, cnt2, time2):
        self.robotarm[0] = cnt1
        self.armtime[0] = time1
        self.robotarm[1] = cnt2
        self.armtime[1] = time2

    def get_state(self):
        # To Do: Design wafer_state generation logic in here.
        # refer to https://docs.google.com/document/d/19hd7xoSbEZaNUVrmMcvwLhZL25RYdhKh-MmL4x0mEMY
        # Example...
        """
        offset_1d = 10  # 1 digit offest
        offset_2d = 100  # 2 digit offset
        self.state = self.entry_wafer * offset_1d + self.robotarm[0]
        self.state = self.state * offset_1d + self.robotarm[1]
        self.state = self.state * offset_2d + self.armtime[0]
        self.state = self.state * offset_2d + self.armtime[1]
        for item in self.status_values:
            self.state = self.state * offset_2d + item['time_remaining']
        # To Do: Design Special wafer_state like termination wafer_state.
        if fail_flag:
            self.state = 987654321
        return self.state
        """
        '''
        self.state = ""
        for item in self.status_values:
            self.state += str(item['cnt']) + '|' + str(item['time_remaining']) + '|'

        self.state += str(self.robotarm[0]) + '|' + str(self.armtime[0]) + '|'
        self.state += str(self.robotarm[1]) + '|' + str(self.armtime[1]) + '|'
        self.state += str(self.entry_wafer)  # + '|'
        # self.state += str(self.exit_wafer)
        '''
        self.state = list()
        for item in self.status_values:
            self.state.append(item['cnt'])
            self.state.append(item['time_remaining'])
        self.state.append(self.robotarm[0])
        self.state.append(self.armtime[0])
        self.state.append(self.robotarm[1])
        self.state.append(self.armtime[1])
        self.state.append(self.entry_wafer)
        return self.state

    def get_reward(self, fail_flag, success_flag):
        # To Do: Design reward generation logic in here.
        # Example.. refer to design docs.
        self.reward = 0
        for item in self.status_values:
            if item['cnt'] == 0:
                if (item['name'] == 'ch2nd_1') | (item['name'] == 'ch2nd_2'):
                    self.reward = self.reward - 3
                else:
                    self.reward = self.reward - 2
            if item['cnt'] == 1 and item['time_remaining'] == 0:
                self.reward = self.reward - 3

        '''
        for i in self.robotarm:
            if i == 0:
                self.reward = self.reward - 1
        '''

        if self.entry_wafer == 0:
            self.reward = self.reward - 1

        if self.prev_exit_wafer != self.exit_wafer:
            self.reward = 1000
            self.prev_exit_wafer = self.exit_wafer
            if self.prev_exit_wafer == self.total_wafer:
                success_flag = True

        # To Do: Design Terminate reward -1000
        # and Finish wafer_state +1000
        # if fail_flag:
            # self.reward = -1000
        if success_flag:
            self.reward = +1000
        return self.reward

    def print_info(self, reward, env):
        logging.info('State: {0}, Reward: {1}'.format(self.get_state(), reward))
        logging.debug("Chamber Wafer Time_remaining")
        for item in self.status_values:
            logging.debug('{0} {1:5d} {2:14d}'.format(item['name'], item['cnt'], item['time_remaining']))
        logging.debug('Arm1    {0:5d} {1:14d}'.format(self.robotarm[0], self.armtime[0]))
        logging.debug('Arm2    {0:5d} {1:14d}'.format(self.robotarm[1], self.armtime[1]))
        logging.debug('Entry   {0:5d}'.format(self.entry_wafer))
        logging.debug('Exit    {0:5d}'.format(self.exit_wafer))
        logging.debug('time: %s-----------------------------\n', env.now)
        # print('-----------------------------')


# Chamber_model class makes a time-out event to handler after it gets the wafer.
# See the example : https://simpy.readthedocs.io/en/latest/examples/latency.html
class chamber_model(object):
    def __init__(self, env, chamber_time, chamber_name, pre, post):
        self.env = env
        # chamber_name is the identifier where event comes from.
        self.chamber_name = chamber_name
        # chamber_time should be 'time_ch1" or 'time_ch2'
        self.chamber_type = chamber_time
        self.store = simpy.Store(self.env)
        # count of wafers in execution.
        # accumulated count of completed wafers.
        self.wafer_completion_count = 0
        # handler should read this attribute for wafer_state description.
        self.wafer_start_time = 0
        self.execution_time = 0
        self.pre_state = pre
        self.post_state = post
        self.fail = False  # Result of get, put method. should be call after them.

    def put_proc(self, wafer, finish_t, evt_finish):
        self.store.put(wafer)
        self.wafer_start_time = self.env.now
        yield self.env.timeout(finish_t)
        # Add chamber processing finish event
        if not evt_finish.triggered:
            evt_finish.succeed(value=FabModel.no_chm)

    def put(self, wafer, evt):
        if self.store.items.__len__() == 1:
            logging.debug("chamber put fail")
            self.fail = True
        if wafer['wafer_state'] == self.pre_state:
            self.execution_time = wafer[self.chamber_type]
            self.env.process(self.put_proc(wafer, self.execution_time, evt))
        else:
            logging.debug('error wafer state doesnt match %s %s', wafer['wafer_state'],
                          self.pre_state)  # go to terminate.
            self.fail = True

    def get(self):
        if self.store.items.__len__() == 0:
            logging.debug("[ERR]chamber get fail. it is empty.")
            self.fail = True
        if self.env.now - self.wafer_start_time < self.execution_time:
            logging.debug('[ERR]chamber get fail. execution time violated.')
            self.fail = True

        wafer = self.store.get()
        self.wafer_completion_count += 1
        self.wafer_start_time = 0
        wafer.value['wafer_state'] = self.post_state
        return wafer

    def get_time_remaining(self):
        cur_time = self.env.now
        remaining_time = 0
        if self.store.items.__len__():
            remaining_time = self.wafer_start_time + self.execution_time - cur_time
        if remaining_time < 0:
            remaining_time = 0
        return remaining_time

    def get_count(self):
        return self.store.items.__len__()


# It differs from chamber_model that it doesn't check wafer's wafer_state.
class arm_model(object):
    def __init__(self, env, arm_time, arm_name):
        self.env = env
        # arm_name is the identifier where event comes from.
        self.arm_name = arm_name
        self.arm_time = arm_time
        self.store = simpy.Store(self.env)
        # accumulated count of completed wafers.
        self.wafer_completion_count = 0
        # handler should read this attribute for wafer_state description.
        self.wafer_start_time = 0
        self.fail = False  # Result of get, put method. should be call after them.

    def put_proc(self, wafer, time):
        logging.debug('at %s, arm put wafer', self.env.now)
        self.store.put(wafer)
        yield self.env.timeout(time)
        self.wafer_start_time = self.env.now

    def put(self, wafer, evt):
        # To Do: wafer's process status should be updated at this moment.
        if self.arm_name == 'exit' and wafer['wafer_state'] != 'ch2 done':
            logging.debug("[ERR]arm tries to put the wrong wafer")
            self.fail = True
        if self.store.items.__len__() == 1 and self.arm_name != 'exit':
            logging.debug("[ERR]Arm access full slot to put")
            self.fail = True
        if not self.fail:
            self.env.process(self.put_proc(wafer, self.arm_time))

    # Put wafer to monitored store after timeout event.
    # Handler process will catch the timeout event.
    # Handler should calculate every chamber's execution status in time.
    # get wafer from Monitored store.
    def get(self):
        wafer = self.store.get()
        self.wafer_completion_count += 1
        self.wafer_start_time = 0
        return wafer

    def get_time_remaining(self):
        time_remain = 0
        if self.store.items.__len__() != 0:
            wafer = self.store.items[0]
            if wafer['wafer_state'] == 'raw':
                time_remain = wafer['time_ch1']
            elif wafer['wafer_state'] == 'ch1 done':
                time_remain = wafer['time_ch2']
            else:
                time_remain = 1
        else:
            time_remain = 0
        return time_remain

    def get_count(self):
        return self.store.items.__len__()


class FabModel(object):
    action_dict = {21: "Nop",
                   1: "airlock entry to 1st arm", 2: "airlock entry to 2nd arm",
                   3: "1st arm to airlock exit", 4: "2nd arm to airlock exit",
                   5: 'CH1_1 to 1st arm', 6: "CH1_1 to 2nd arm",
                   7: 'CH1_2 to 1st arm', 8: "CH1_2 to 2nd arm",
                   9: 'CH2_1 to 1st arm', 10: 'CH2_1 to 2nd arm',
                   11: 'CH2_2 to 1st arm', 12: 'CH2_2 to 2nd arm',
                   13: '1st arm to CH1_1', 14: '2nd arm to CH1_1',
                   15: '1st arm to CH1_2', 16: '2nd arm to CH1_2',
                   17: '1st arm to CH2_1', 18: '2nd arm to CH2_1',
                   19: '1st arm to CH2_2', 20: '2nd arm to CH2_2',
                   0: 'allocate wafer to entry'}
    name_chambers = ['ch1st_1', 'ch1st_2', 'ch2nd_1', 'ch2nd_2']
    time_chambers = ['time_ch1', 'time_ch2']
    wafer_state = ['raw', 'ch1 done', 'ch2 done']
    no_entry, no_exit, no_hdlr, no_step, no_chm = 1, 2, 3, 4, 5  # Event value assignment.
    curr_nope_count = 0

    def __init__(self, wafer_number):
        self.env = simpy.Environment()
        self.robot_arm = list()
        self.airlock = list()
        self.chambers = list()
        self.wafer_number = wafer_number
        self.wafer_in_proc = 0
        self.curr_nope_count = 0
        self.initialize()

    def initialize(self):
        self.env = simpy.Environment()
        # Allocate wafers to processing on the chamber system.
        wafers = self.generate_wafers(self.wafer_number, 3, 15, 5, 30)
        # wafers = self.generate_wafers(self.wafer_number, 3, 3, 5, 5)
        # Allocate robot arm and chamber, airlock resources.
        self.robot_arm.clear()
        self.robot_arm.append(arm_model(self.env, 2, '1st_arm'))
        self.robot_arm.append(arm_model(self.env, 2, '2nd_arm'))
        self.airlock.clear()
        self.airlock.append(arm_model(self.env, arm_time=2, arm_name='entry'))
        self.airlock.append(arm_model(self.env, arm_time=2, arm_name='exit'))
        self.chambers.clear()
        # chambers_name = {'ch1st_1': 'time_ch1', 'ch1st_2': 'time_ch1', 'ch2nd_1': 'time_ch2', 'ch2nd_2': 'time_ch2'}
        self.chambers.append(
            chamber_model(self.env, chamber_time=self.time_chambers[0], chamber_name=self.name_chambers[0],
                          pre=self.wafer_state[0], post=self.wafer_state[1]))
        self.chambers.append(
            chamber_model(self.env, chamber_time=self.time_chambers[0], chamber_name=self.name_chambers[1],
                          pre=self.wafer_state[0], post=self.wafer_state[1]))
        self.chambers.append(
            chamber_model(self.env, chamber_time=self.time_chambers[1], chamber_name=self.name_chambers[2],
                          pre=self.wafer_state[1], post=self.wafer_state[2]))
        self.chambers.append(
            chamber_model(self.env, chamber_time=self.time_chambers[1], chamber_name=self.name_chambers[3],
                          pre=self.wafer_state[1], post=self.wafer_state[2]))
        # Initialize the event variables.
        self.event_entry = self.env.event()
        self.event_exit = self.env.event()
        self.event_hdlr = self.env.event()
        self.event_step = self.env.event()
        self.event_chm = self.env.event()
        self.event_action = self.env.event()
        # Initialize variables on states and rewards.
        self.fail_flag, self.success_flag, self.done = False, False, False
        self.action, self.reward = 0, 0
        self.state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 13 items

        self.process_handler = self.env.process(self.proc_handler(self.airlock, self.robot_arm, self.chambers))
        self.process_airlock_entry = self.env.process(self.proc_entry(self.wafer_number, self.airlock[0], wafers))
        self.profiler = self.init_chamber_profiler()
        self.curr_nope_count = 0
        self.wafer_in_proc = 0
        return

    def reset(self):
        del self.env
        self.initialize()

    def step(self, action):
        # Do NOT manipulate statements sequence!
        self.action = action
        self.event_action.succeed()
        self.env.run(self.event_step)
        self.event_step = self.env.event()
        obs = self.get_observation()
        if self.done:
            logging.debug("Done flag raised.")
            return obs
        return obs


    def init_chamber_profiler(self):
        ch_names = list()
        for ch in self.chambers:
            ch_names.append(ch.chamber_name)
        profiler = chamber_profiler(ch_names, self.wafer_number)
        return profiler

    def get_observation(self):
        for ch in self.chambers:
            self.profiler.update_chamber_status(ch.chamber_name,
                                                ch.store.items.__len__(),
                                                ch.get_time_remaining())
        self.profiler.update_entry_exit_status(self.airlock[0].get_count(),
                                               self.airlock[1].get_count())
        self.profiler.update_arm_status(self.robot_arm[0].get_count(),
                                        self.robot_arm[0].get_time_remaining(),
                                        self.robot_arm[1].get_count(),
                                        self.robot_arm[1].get_time_remaining())
        self.reward = self.profiler.get_reward(self.fail_flag,
                                               self.success_flag)
        self.state = self.profiler.get_state()
        # self.state.append(self.curr_nope_count)

        # for debug
        self.profiler.print_info(self.reward, self.env)

        if self.fail_flag is True:
            self.done = True
            logging.info("--------Terminate state!!!--------")
            if not self.event_step.triggered:
                self.event_step.succeed(value=self.event_step)
            # after this statement, step() method should check done flag and terminate.
        else:
            self.done = False
        # logging.debug("at %s state: %r reward: %r Done:%r", self.env.now, self.state, self.reward, self.done)

        if self.env.now > self.wafer_number * 1000:
            self.done = True

        # if self.curr_nope_count > 30:
        #    self.done = True

        return (self.state, self.reward, self.done)

    def proc_handler(self, airlock_list, arm_list, chambers_list):
        # This while loop should run all of the posterior events were completed.
        self.event_hdlr.succeed(value=self.no_hdlr)
        while True:
            # Simplified events and handler choose corrective action by observing its wafer_state and reward.
            # action = yield event # if events comes from the other processes, it means there is a job for robot
            # Profiling current wafer_state and information.
            # Update chamber, entry, arm status.

            timeout_no_op = 1  # Elasped time at no operation.

            yield (self.event_action)
            if self.event_action.triggered:
                logging.debug('at %s action event detected', self.env.now)
                self.event_action = self.env.event()
            yield (self.event_hdlr | self.event_chm)
            if self.event_hdlr.triggered:
                logging.debug('at %s hdlr event detected', self.env.now)
                self.event_hdlr = self.env.event()
            if self.event_chm.triggered:
                logging.debug('at %s chamber event detected', self.env.now)
                self.event_chm = self.env.event()
                continue

            logging.debug('------------------------- ')
            logging.debug('at %s Action Taken:[%s] %s', self.env.now, self.action, FabModel.action_dict[self.action])
            action_taken = int(self.action)
            # Select Action
            if action_taken == 21:
                self.curr_nope_count += 1
                yield self.env.timeout(timeout_no_op)
                if not self.event_step.triggered:
                    self.event_step.succeed(value=self.no_step)
                if not self.event_hdlr.triggered:
                    logging.debug('at %s, hdlr trigger on proc_hdlr', self.env.now)
                    self.event_hdlr.succeed(value=self.no_hdlr)

            elif action_taken == 1:
                self.curr_nope_count = 0
                self.env.process(self.move_wafer_A_from_B(airlock_list[0], arm_list[0]))
            elif action_taken == 2:
                self.curr_nope_count = 0
                self.env.process(self.move_wafer_A_from_B(airlock_list[0], arm_list[1]))
            elif action_taken == 3:
                self.curr_nope_count = 0
                self.env.process(self.move_wafer_A_from_B(arm_list[0], airlock_list[1]))
            elif action_taken == 4:
                self.curr_nope_count = 0
                self.env.process(self.move_wafer_A_from_B(arm_list[1], airlock_list[1]))
            elif action_taken == 5:
                self.curr_nope_count = 0
                self.env.process(self.move_wafer_A_from_B(chambers_list[0], arm_list[0]))
            elif action_taken == 6:
                self.curr_nope_count = 0
                self.env.process(self.move_wafer_A_from_B(chambers_list[0], arm_list[1]))
            elif action_taken == 7:
                self.curr_nope_count = 0
                self.env.process(self.move_wafer_A_from_B(chambers_list[1], arm_list[0]))
            elif action_taken == 8:
                self.curr_nope_count = 0
                self.env.process(self.move_wafer_A_from_B(chambers_list[1], arm_list[1]))
            elif action_taken == 9:
                self.curr_nope_count = 0
                self.env.process(self.move_wafer_A_from_B(chambers_list[2], arm_list[0]))
            elif action_taken == 10:
                self.curr_nope_count = 0
                self.env.process(self.move_wafer_A_from_B(chambers_list[2], arm_list[1]))
            elif action_taken == 11:
                self.curr_nope_count = 0
                self.env.process(self.move_wafer_A_from_B(chambers_list[3], arm_list[0]))
            elif action_taken == 12:
                self.curr_nope_count = 0
                self.env.process(self.move_wafer_A_from_B(chambers_list[3], arm_list[1]))
            elif action_taken == 13:
                self.curr_nope_count = 0
                self.env.process(self.move_wafer_A_from_B(arm_list[0], chambers_list[0]))
            elif action_taken == 14:
                self.curr_nope_count = 0
                self.env.process(self.move_wafer_A_from_B(arm_list[1], chambers_list[0]))
            elif action_taken == 15:
                self.curr_nope_count = 0
                self.env.process(self.move_wafer_A_from_B(arm_list[0], chambers_list[1]))
            elif action_taken == 16:
                self.curr_nope_count = 0
                self.env.process(self.move_wafer_A_from_B(arm_list[1], chambers_list[1]))
            elif action_taken == 17:
                self.curr_nope_count = 0
                self.env.process(self.move_wafer_A_from_B(arm_list[0], chambers_list[2]))
            elif action_taken == 18:
                self.curr_nope_count = 0
                self.env.process(self.move_wafer_A_from_B(arm_list[1], chambers_list[2]))
            elif action_taken == 19:
                self.curr_nope_count = 0
                self.env.process(self.move_wafer_A_from_B(arm_list[0], chambers_list[3]))
            elif action_taken == 20:
                self.curr_nope_count = 0
                self.env.process(self.move_wafer_A_from_B(arm_list[1], chambers_list[3]))
            elif action_taken == 0:
                self.curr_nope_count = 0
                yield self.env.timeout(timeout_no_op)
                self.wafer_in_proc += 1
                if self.wafer_in_proc <= self.wafer_number:
                    if self.event_entry.triggered:
                        self.event_entry = self.env.event()
                    self.event_entry.succeed(value=self.event_entry)
                else:
                    logging.debug("wafer is sold out!")
                    self.fail_flag = True
                    if not self.event_step.triggered:
                        self.event_step.succeed(value=self.no_step)
                    if not self.event_hdlr.triggered:
                        # logging.debug('at %s, hdlr trigger on proc_hdlr', self.env.now)
                        self.event_hdlr.succeed(value=self.no_hdlr)

            else:
                logging.debug('[ERR] undefined action taken: %d', action_taken)

        return


    # check current state and return valid action set
    # for example,
    # return {0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}
    # means 2, 3, and 21 actions are valid
    # if the definition of actions in proc_handler are changed,
    # this function has to be changed.
    def get_valid_action_mask(self):
        valid_action_mask = np.ones(22)  # --ACTION_DIM

        # get errors
        if self.airlock[0].store.items.__len__() == 0:
            # self.env.process(self.move_wafer_A_from_B(airlock_list[0], arm_list[0]))
            valid_action_mask[1] = 0
            # self.env.process(self.move_wafer_A_from_B(airlock_list[0], arm_list[1]))
            valid_action_mask[2] = 0

        if self.robot_arm[0].store.items.__len__() == 0:
            # self.env.process(self.move_wafer_A_from_B(arm_list[0], airlock_list[1]))
            valid_action_mask[3] = 0
            # self.env.process(self.move_wafer_A_from_B(arm_list[0], chambers_list[0]))
            valid_action_mask[13] = 0
            # self.env.process(self.move_wafer_A_from_B(arm_list[0], chambers_list[1]))
            valid_action_mask[15] = 0
            # self.env.process(self.move_wafer_A_from_B(arm_list[0], chambers_list[2]))
            valid_action_mask[17] = 0
            # self.env.process(self.move_wafer_A_from_B(arm_list[0], chambers_list[3]))
            valid_action_mask[19] = 0

        if self.robot_arm[1].store.items.__len__() == 0:
            # self.env.process(self.move_wafer_A_from_B(arm_list[1], airlock_list[1]))
            valid_action_mask[4] = 0
            # self.env.process(self.move_wafer_A_from_B(arm_list[1], chambers_list[0]))
            valid_action_mask[14] = 0
            # self.env.process(self.move_wafer_A_from_B(arm_list[1], chambers_list[1]))
            valid_action_mask[16] = 0
            # self.env.process(self.move_wafer_A_from_B(arm_list[1], chambers_list[2]))
            valid_action_mask[18] = 0
            # self.env.process(self.move_wafer_A_from_B(arm_list[1], chambers_list[3]))
            valid_action_mask[20] = 0

        if (self.chambers[0].store.items.__len__() == 0) \
                or (self.chambers[0].store.items.__len__() == 1
                    and (self.chambers[0].env.now - self.chambers[0].wafer_start_time < self.chambers[0].execution_time)):
            # self.env.process(self.move_wafer_A_from_B(chambers_list[0], arm_list[0]))
            valid_action_mask[5] = 0
            # self.env.process(self.move_wafer_A_from_B(chambers_list[0], arm_list[1]))
            valid_action_mask[6] = 0

        if (self.chambers[1].store.items.__len__() == 0) \
                or (self.chambers[1].store.items.__len__() == 1
                    and (self.chambers[1].env.now - self.chambers[1].wafer_start_time < self.chambers[1].execution_time)):
            # self.env.process(self.move_wafer_A_from_B(chambers_list[1], arm_list[0]))
            valid_action_mask[7] = 0
            # self.env.process(self.move_wafer_A_from_B(chambers_list[1], arm_list[1]))
            valid_action_mask[8] = 0

        if (self.chambers[2].store.items.__len__() == 0) \
                or (self.chambers[2].store.items.__len__() == 1
                    and (self.chambers[2].env.now - self.chambers[2].wafer_start_time < self.chambers[0].execution_time)):
            # self.env.process(self.move_wafer_A_from_B(chambers_list[2], arm_list[0]))
            valid_action_mask[9] = 0
            # self.env.process(self.move_wafer_A_from_B(chambers_list[2], arm_list[1]))
            valid_action_mask[10] = 0

        if (self.chambers[3].store.items.__len__() == 0) \
                or (self.chambers[3].store.items.__len__() == 1
                    and (self.chambers[3].env.now - self.chambers[3].wafer_start_time < self.chambers[3].execution_time)):
            # self.env.process(self.move_wafer_A_from_B(chambers_list[3], arm_list[0]))
            valid_action_mask[11] = 0
            # self.env.process(self.move_wafer_A_from_B(chambers_list[3], arm_list[1]))
            valid_action_mask[12] = 0

        # put errors
        if self.airlock[0].store.items.__len__() != 0:
            valid_action_mask[0] = 0

        if self.robot_arm[0].store.items.__len__() != 0:
            # self.env.process(self.move_wafer_A_from_B(airlock_list[0], arm_list[0]))
            valid_action_mask[1] = 0
            # self.env.process(self.move_wafer_A_from_B(chambers_list[0], arm_list[0]))
            valid_action_mask[5] = 0
            # self.env.process(self.move_wafer_A_from_B(chambers_list[1], arm_list[0]))
            valid_action_mask[7] = 0
            # self.env.process(self.move_wafer_A_from_B(chambers_list[2], arm_list[0]))
            valid_action_mask[9] = 0
            # self.env.process(self.move_wafer_A_from_B(chambers_list[3], arm_list[0]))
            valid_action_mask[11] = 0

        if self.robot_arm[1].store.items.__len__() != 0:
            # self.env.process(self.move_wafer_A_from_B(airlock_list[0], arm_list[1]))
            valid_action_mask[2] = 0
            # self.env.process(self.move_wafer_A_from_B(chambers_list[0], arm_list[1]))
            valid_action_mask[6] = 0
            # self.env.process(self.move_wafer_A_from_B(chambers_list[1], arm_list[1]))
            valid_action_mask[8] = 0
            # self.env.process(self.move_wafer_A_from_B(chambers_list[2], arm_list[1]))
            valid_action_mask[10] = 0
            # self.env.process(self.move_wafer_A_from_B(chambers_list[3], arm_list[1]))
            valid_action_mask[12] = 0

        if self.chambers[0].store.items.__len__() != 0:
            # self.env.process(self.move_wafer_A_from_B(arm_list[0], chambers_list[0]))
            valid_action_mask[13] = 0
            # self.env.process(self.move_wafer_A_from_B(arm_list[1], chambers_list[0]))
            valid_action_mask[14] = 0

        if self.chambers[1].store.items.__len__() != 0:
            # self.env.process(self.move_wafer_A_from_B(arm_list[0], chambers_list[1]))
            valid_action_mask[15] = 0
            # self.env.process(self.move_wafer_A_from_B(arm_list[1], chambers_list[1]))
            valid_action_mask[16] = 0

        if self.chambers[2].store.items.__len__() != 0:
            # self.env.process(self.move_wafer_A_from_B(arm_list[0], chambers_list[2]))
            valid_action_mask[17] = 0
            # self.env.process(self.move_wafer_A_from_B(arm_list[1], chambers_list[2]))
            valid_action_mask[18] = 0

        if self.chambers[3].store.items.__len__() != 0:
            # self.env.process(self.move_wafer_A_from_B(arm_list[0], chambers_list[3]))
            valid_action_mask[19] = 0
            # self.env.process(self.move_wafer_A_from_B(arm_list[1], chambers_list[3]))
            valid_action_mask[20] = 0

        #invalid put
        if self.robot_arm[0].store.items.__len__() != 0:
            if self.robot_arm[0].store.items[0]['wafer_state'] == 'raw':
                # self.env.process(self.move_wafer_A_from_B(arm_list[0], airlock_list[1]))
                valid_action_mask[3] = 0
                # self.env.process(self.move_wafer_A_from_B(arm_list[0], chambers_list[2]))
                valid_action_mask[17] = 0
                # self.env.process(self.move_wafer_A_from_B(arm_list[0], chambers_list[3]))
                valid_action_mask[19] = 0
            if self.robot_arm[0].store.items[0]['wafer_state'] == 'ch1 done':
                # self.env.process(self.move_wafer_A_from_B(arm_list[0], airlock_list[1]))
                valid_action_mask[3] = 0
                # self.env.process(self.move_wafer_A_from_B(arm_list[0], chambers_list[0]))
                valid_action_mask[13] = 0
                # self.env.process(self.move_wafer_A_from_B(arm_list[0], chambers_list[1]))
                valid_action_mask[15] = 0
            if self.robot_arm[0].store.items[0]['wafer_state'] == 'ch2 done':
                # self.env.process(self.move_wafer_A_from_B(arm_list[0], chambers_list[0]))
                valid_action_mask[13] = 0
                # self.env.process(self.move_wafer_A_from_B(arm_list[0], chambers_list[1]))
                valid_action_mask[15] = 0
                # self.env.process(self.move_wafer_A_from_B(arm_list[0], chambers_list[2]))
                valid_action_mask[17] = 0
                # self.env.process(self.move_wafer_A_from_B(arm_list[0], chambers_list[3]))
                valid_action_mask[19] = 0

        if self.robot_arm[1].store.items.__len__() != 0:
            if self.robot_arm[1].store.items[0]['wafer_state'] == 'raw':
                # self.env.process(self.move_wafer_A_from_B(arm_list[1], airlock_list[1]))
                valid_action_mask[4] = 0
                # self.env.process(self.move_wafer_A_from_B(arm_list[1], chambers_list[2]))
                valid_action_mask[18] = 0
                # self.env.process(self.move_wafer_A_from_B(arm_list[1], chambers_list[3]))
                valid_action_mask[20] = 0
            if self.robot_arm[1].store.items[0]['wafer_state'] == 'ch1 done':
                # self.env.process(self.move_wafer_A_from_B(arm_list[1], airlock_list[1]))
                valid_action_mask[4] = 0
                # self.env.process(self.move_wafer_A_from_B(arm_list[1], chambers_list[0]))
                valid_action_mask[14] = 0
                # self.env.process(self.move_wafer_A_from_B(arm_list[1], chambers_list[1]))
                valid_action_mask[16] = 0
            if self.robot_arm[1].store.items[0]['wafer_state'] == 'ch2 done':
                # self.env.process(self.move_wafer_A_from_B(arm_list[1], chambers_list[0]))
                valid_action_mask[14] = 0
                # self.env.process(self.move_wafer_A_from_B(arm_list[1], chambers_list[1]))
                valid_action_mask[16] = 0
                # self.env.process(self.move_wafer_A_from_B(arm_list[1], chambers_list[2]))
                valid_action_mask[18] = 0
                # self.env.process(self.move_wafer_A_from_B(arm_list[1], chambers_list[3]))
                valid_action_mask[20] = 0

        return valid_action_mask


    # move wafer function.
    def move_wafer_A_from_B(self, A, B):
        yield self.env.timeout(1)  # Deliver Time
        if A.store.items.__len__() == 0:
            logging.debug("[ERR] Get Fail. Target is empty.")
            self.fail_flag = True
            if not self.event_step.triggered:
                self.event_step.succeed(value=FabModel.no_step)

            return
        # if not self.event_hdlr.triggered:
        #    logging.debug('at %s, hdlr trigger on move wafer', self.env.now)
        #    self.event_hdlr.succeed(value=self.no_hdlr)

        wafer = yield A.get()
        self.fail_flag = A.fail
        if not self.fail_flag:
            B.put(wafer, self.event_chm)
            self.fail_flag = B.fail
            if not self.event_step.triggered:
                self.event_step.succeed(value=FabModel.no_step)

            if not self.event_hdlr.triggered and not B.fail:
                logging.debug('at %s, hdlr trigger on move wafer', self.env.now)
                self.event_hdlr.succeed(value=FabModel.no_hdlr)
        else:
            if not self.event_step.triggered:
                self.event_step.succeed(value=FabModel.no_step)

        return

    # A process moves wafers to airlock entry, time-out event after a wafer placed on the airlock entry.
    def proc_entry(self, tot_wafers, airlock_entry, wafers):
        for i in range(tot_wafers):
            yield self.env.timeout(1)
            yield self.event_entry
            self.event_entry = self.env.event()

            if airlock_entry.get_count() == 1:
                logging.debug("[ERR] Entry put fail. Airlock is already full.")
                self.fail_flag = True
                if not self.event_step.triggered:
                    self.event_step.succeed()
                break
            airlock_entry.put(wafers[i], self.event_hdlr)
            self.fail_flag = airlock_entry.fail
            if not self.event_step.triggered:
                self.event_step.succeed()

            if not self.event_hdlr.triggered:
                logging.debug('at %s, hdlr trigger on proc_entry', self.env.now)
                self.event_hdlr.succeed()

    # Generate wafer list randomly and return it.
    def generate_wafers(self, tot_wafers, ch1_t_min, ch1_t_max, ch2_t_min, ch2_t_max):
        wafer_list = list()
        for i in range(tot_wafers):
            wafer_list.append({'id': i,
                               'wafer_state': 'raw',
                               'time_ch1': random.randint(ch1_t_min, ch1_t_max),
                               'time_ch2': random.randint(ch2_t_min, ch2_t_max)})
        return wafer_list


"""
input: action
do: change the status
return: observation, reward, done(True/False)
"""
# For debugging.
if __name__ == "__main__":
    model = FabModel(20)
    # alist = [0, 0, 0, 0,]
    # alist = [0, 1, 0, 2, 15, 14, 21, 0, 7, 6, 18, 2, 0, 14]
    # alist = [0, 2, 14, 21, 21, 0, 2, 21, 16, 6, 0, 21, 7, 20, 17, 2, 16]
    alist = [0, 1, 13, 0, 21, 1]
    for i in alist:
        result = model.step(action=i)
        if result[2]:
            break

    print('1st epich finish')
    exit()
    model.reset()

    while True:
        action_index = int(input())
        result = model.step(action=action_index)
        if result[2]:
            model.reset()
            break

    model.reset()

    print('2st epich finish')
    alist = [0 for _ in range(100)]
    action_count = 0
    for i in alist:
        result = model.step(action=i)
        action_count += 1
        if result[2]:
            model.reset()
            break
    print('action count=', action_count)
