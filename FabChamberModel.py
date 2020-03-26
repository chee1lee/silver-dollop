import random
import socket
from datetime import date

import simpy

# for import plotly, Dash should be installed.
#    pip install dash==1.7.0
#    pip install numpy
FIRST_DATE = date(2020, 1, 1)

# manages chamber wafer_state and reward information.
class chamber_profiler(object):
    def __init__(self, chambers_name):
        global fail_flag, success_flag
        self.reward = 0
        self.state = 0
        self.ch_names = chambers_name
        # It indicates whether wafer in producer slot is existed or not.
        self.entry_wafer = 0
        self.exit_wafer = 0
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
        self.state = ""
        for item in self.status_values:
            self.state += str(item['cnt'] )+ '|' + str(item['time_remaining']) + '|'

        self.state += str(self.robotarm[0]) + '|' + str(self.armtime[0]) + '|'
        self.state += str(self.robotarm[1]) + '|' + str(self.armtime[1]) + '|'
        self.state += str(self.entry_wafer) + '|'
        self.state += str(self.exit_wafer)

        return self.state


    def get_reward(self):
        # To Do: Design reward generation logic in here.
        # Example.. refer to design docs.
        self.reward = 0
        for item in self.status_values:
            if item['time_remaining'] == 0:
                self.reward = self.reward - 1
        for i in self.robotarm:
            if i == 0:
                self.reward = self.reward - 1
        # To Do: Design Terminate reward -10000
        # and Finish wafer_state +10000
        if fail_flag:
            self.reward = -10000
        if success_flag:
            self.reward = +10000
        return self.reward

    def print_info(self):
        print('State: {0}, Reward: {1}'.format(self.get_state(), self.get_reward()))
        print("Chamber Wafer Time_remaining")
        for item in self.status_values:
            print('{0} {1:5d} {2:14d}'.format(item['name'], item['cnt'], item['time_remaining']))
        print('Arm1    {0:5d} {1:14d}'.format(self.robotarm[0], self.armtime[0]))
        print('Arm2    {0:5d} {1:14d}'.format(self.robotarm[1], self.armtime[1]))
        print('Entry   {0:5d}'.format(self.entry_wafer))
        print('Exit    {0:5d}'.format(self.exit_wafer))
        print('time:', env.now, '-----------------------------')

# process make a decision of robot arm.
def proc_handler(env, airlock_list, arm_list, chambers_list):
    global event_entry, event_hdlr, fail_flag, success_flag, conn, recv_data
    action_dict = {0: "Nop",
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
                   21: 'allocate wafer to entry'}
    # Initiate profiler.
    ch_names = list()
    for ch in chambers_list:
        ch_names.append(ch.chamber_name)
    profiler = chamber_profiler(ch_names)
    timeout_value = 1
    # This while loop should run all of the posterior events were completed.
    while True:
        # Simplified events and handler choose corrective action by observing its wafer_state and reward.
        # action = yield event # if events comes from the other processes, it means there is a job for robot
        # event = env.event()
        # Profiling current wafer_state and information.
        # Update chamber, entry, arm status.
        yield (env.timeout(timeout_value) | event_hdlr)
        event_hdlr = env.event()
        timeout_value = 5
        time_now = env.now
        for ch in chambers_list:
            profiler.update_chamber_status(ch.chamber_name, ch.store.items.__len__(), ch.get_time_remaining())
        profiler.update_entry_exit_status(airlock_list[0].get_count(), airlock_list[1].get_count())
        profiler.update_arm_status(arm_list[0].get_count(), arm_list[0].get_time_remaining(),
                                   arm_list[1].get_count(), arm_list[1].get_time_remaining())
        profiler.print_info()
        state = profiler.get_state()
        reward = profiler.get_reward()
        done = False
        if fail_flag is True:
            done = True
        send_data = str(state) + ' ' + str(reward) + ' ' + str(done)
        conn.send(send_data.encode())

        if fail_flag is True:
            print("--------Termininate state!!!--------")
            # env.exit()

        # select action from socket

        byte_action = conn.recv(1024)
        if byte_action.decode() == 'reset':
            env.exit()
        # print('aaaaaaaaa', str(byte_action))
        action_taken = int(byte_action)
        # Select Action
        # print(action_dict)
        # action_taken = int(input('Select actions(0~21)?'))
        if action_taken == 0:
            i = 0  # do nothing
        elif action_taken == 1:
            env.process(move_wafer_A_from_B(airlock_list[0], arm_list[0]))
        elif action_taken == 2:
            env.process(move_wafer_A_from_B(airlock_list[0], arm_list[1]))
        elif action_taken == 3:
            env.process(move_wafer_A_from_B(arm_list[0], airlock_list[1]))
        elif action_taken == 4:
            env.process(move_wafer_A_from_B(arm_list[1], airlock_list[1]))
        elif action_taken == 5:
            env.process(move_wafer_A_from_B(chambers_list[0], arm_list[0]))
        elif action_taken == 6:
            env.process(move_wafer_A_from_B(chambers_list[0], arm_list[1]))
        elif action_taken == 7:
            env.process(move_wafer_A_from_B(chambers_list[1], arm_list[0]))
        elif action_taken == 8:
            env.process(move_wafer_A_from_B(chambers_list[1], arm_list[1]))
        elif action_taken == 9:
            env.process(move_wafer_A_from_B(chambers_list[2], arm_list[0]))
        elif action_taken == 10:
            env.process(move_wafer_A_from_B(chambers_list[2], arm_list[1]))
        elif action_taken == 11:
            env.process(move_wafer_A_from_B(chambers_list[3], arm_list[0]))
        elif action_taken == 12:
            env.process(move_wafer_A_from_B(chambers_list[3], arm_list[1]))
        elif action_taken == 13:
            env.process(move_wafer_A_from_B(arm_list[0], chambers_list[0]))
        elif action_taken == 14:
            env.process(move_wafer_A_from_B(arm_list[1], chambers_list[0]))
        elif action_taken == 15:
            env.process(move_wafer_A_from_B(arm_list[0], chambers_list[1]))
        elif action_taken == 16:
            env.process(move_wafer_A_from_B(arm_list[1], chambers_list[1]))
        elif action_taken == 17:
            env.process(move_wafer_A_from_B(arm_list[0], chambers_list[2]))
        elif action_taken == 18:
            env.process(move_wafer_A_from_B(arm_list[1], chambers_list[2]))
        elif action_taken == 19:
            env.process(move_wafer_A_from_B(arm_list[0], chambers_list[3]))
        elif action_taken == 20:
            env.process(move_wafer_A_from_B(arm_list[1], chambers_list[3]))
        elif action_taken == 21:
            if event_entry.triggered:
                print("Error Event interaction.")
                event_entry = env.event()
            event_entry.succeed()
        else:
            print('Error undefined action taken.', action_taken)


# move wafer function.
def move_wafer_A_from_B(A, B):
    global fail_flag
    if A.store.items.__len__() == 0:
        print("chamber get fail")
        fail_flag = True
        return

    global event_hdlr
    yield env.timeout(2)
    wafer = yield A.get()
    B.put(wafer)
    if not event_hdlr.triggered:
        event_hdlr.succeed()
    return

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

    def proc_put(self, wafer, time):
        self.store.put(wafer)
        self.wafer_start_time = env.now
        yield self.env.timeout(time)
        print(env.now, '\tPut ', self.chamber_name, wafer)

    def put(self, wafer):
        global fail_flag
        if self.store.items.__len__() == 1:
            print("chamber put fail")
            fail_flag = True

        if wafer['wafer_state'] == self.pre_state:
            self.execution_time = wafer[self.chamber_type]
            self.env.process(self.proc_put(wafer, self.execution_time))
        else:
            print('error wafer state doesnt match', wafer['wafer_state'], self.pre_state)  # go to terminate.
            fail_flag = True
            assert 1

    # get wafer from Monitored store.
    def get(self):
        global fail_flag
        if self.store.items.__len__() == 0:
            print("chamber get fail")
            fail_flag = True

        wafer = self.store.get()
        self.wafer_completion_count += 1
        self.wafer_start_time = 0
        wafer.value['wafer_state'] = self.post_state
        return wafer

    def get_time_remaining(self):
        cur_time = env.now
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

    def late(self, wafer, time):
        self.store.put(wafer)
        print(self.env.now, '\tPut ', self.arm_name, wafer)
        yield self.env.timeout(time)
        self.wafer_start_time = env.now

    def put(self, wafer):
        global fail_flag
        # To Do: wafer's process status should be updated at this moment.
        execution_time = self.arm_time
        if self.store.items.__len__() == 1:
            print("arm put fail")
            fail_flag = True
        self.env.process(self.late(wafer, execution_time))

    # Put wafer to monitored store after timeout event.
    # Handler process will catch the timeout event.
    # Handler should calculate every chamber's execution status in time.
    # get wafer from Monitored store.
    def get(self):
        global fail_flag
        if self.store.items.__len__() == 0:
            print("arm get fail")
            fail_flag = True

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


'''
def convert_to_gantt_data(wafer_log_list, task_id, resource_prefix):
    ret = []
    for i in range(1, len(wafer_log_list), 2):
        ret.append(dict(Task=task_id, Start=FIRST_DATE + timedelta(days=wafer_log_list[i - 1][0]),
                        Finish=FIRST_DATE + timedelta(days=wafer_log_list[i][0]),
                        Resource=resource_prefix + str(wafer_log_list[i][1])))
    return ret


def generate_colors(prefix, n):
    ret = {}
    for i in range(n):
        ret[prefix + str(i)] = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))

    return ret
'''


# A process moves wafers to airlock entry, time-out event after a wafer placed on the airlock entry.
def proc_entry(env, tot_wafers, airlock_entry, wafers):
    global event_entry, event_hdlr, fail_flag
    for i in range(tot_wafers):
        yield event_entry
        event_entry = env.event()
        if airlock_entry.get_count() == 1:
            print("entry put fail")
            fail_flag = True
            break;
        airlock_entry.put(wafers[i])
        yield env.timeout(1)
        # print(env.now, 'Entry:', wafers[i])
        if not event_hdlr.triggered:
            event_hdlr.succeed()


# Generate wafer list randomly and return it.
def generate_wafers(tot_wafers, ch1_t_min, ch1_t_max, ch2_t_min, ch2_t_max):
    wafer_list = list()
    for i in range(tot_wafers):
        wafer_list.append({'id': i, 'wafer_state': 'raw',
                           'time_ch1': random.randint(ch1_t_min, ch1_t_max),
                           'time_ch2': random.randint(ch2_t_min, ch2_t_max)})
    return wafer_list


def start_sim():
    global env, event_entry, event_hdlr, fail_flag, success_flag
    env = simpy.Environment()
    # Allocate wafers to processing on the chamber system.
    total_wafers = 20
    wafers = generate_wafers(total_wafers, 3, 15, 5, 30)
    name_chambers = ['ch1st_1', 'ch1st_2', 'ch2nd_1', 'ch2nd_2']
    time_chambers = ['time_ch1', 'time_ch2']
    wafer_state = ['raw', 'ch1 done', 'ch2 done']
    # Allocate robot arm and chamber, airlock resources.
    robot_arm = [arm_model(env, 2, '1st_arm'), arm_model(env, 2, '2nd_arm')]
    airlock = [arm_model(env, arm_time=2, arm_name='entry'),
               arm_model(env, arm_time=2, arm_name='exit')]
    chambers = list()
    # chambers_name = {'ch1st_1': 'time_ch1', 'ch1st_2': 'time_ch1', 'ch2nd_1': 'time_ch2', 'ch2nd_2': 'time_ch2'}
    chambers.append(chamber_model(env, chamber_time=time_chambers[0],
                                  chamber_name=name_chambers[0], pre=wafer_state[0], post=wafer_state[1]))
    chambers.append(chamber_model(env, chamber_time=time_chambers[0],
                                  chamber_name=name_chambers[1], pre=wafer_state[0], post=wafer_state[1]))
    chambers.append(chamber_model(env, chamber_time=time_chambers[1],
                                  chamber_name=name_chambers[2], pre=wafer_state[1], post=wafer_state[2]))
    chambers.append(chamber_model(env, chamber_time=time_chambers[1],
                                  chamber_name=name_chambers[3], pre=wafer_state[1], post=wafer_state[2]))
    # Generate processes.
    event_entry = env.event()
    event_exit = env.event()
    event_hdlr = env.event()
    fail_flag = False
    success_flag = False
    process_handler = env.process(proc_handler(env, airlock, robot_arm, chambers))
    process_airlock_entry = env.process(proc_entry(env, total_wafers, airlock[0], wafers))
    return env.run()


"""
input: action
do: change the status
return: observation, reward, done(True/False)
"""

if __name__ == "__main__":
    # create an INET, STREAMing socket server
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # bind the socket to a public host, and a well-known port
    serversocket.bind(('localhost', 8080))
    # become a server socket
    serversocket.listen(5)
    print('starting server...')
    conn, addr = serversocket.accept()
    print('client connected from:', addr)
    conn.send('type reset'.encode())
    recv_data = conn.recv(1024)
    while recv_data.decode() != 'terminate':
        if recv_data.decode() == 'reset':
            ret = start_sim()
            if ret == 0:
                print("----------------------")
                print("Simulation terminated.")
                print('----------------------')
    serversocket.server_close()

'''
#print(Chamber_1st.data)
#print(Chamber_2nd.data)
#print(arm_list.monitoring_data)

#data_gantt_ch1 = convert_to_gantt_data(Chamber_1st.data, 'Chamber1', 'Wafer')
#data_gantt_ch2 = convert_to_gantt_data(Chamber_2nd.data, 'Chamber2', 'Wafer')
#data_gantt_robot = convert_to_gantt_data(arm_list.monitoring_data, 'Robot', 'Wafer')

#gen_colors = generate_colors('Wafer', 100)

#fig = ff.create_gantt(data_gantt_ch1 + data_gantt_ch2 + data_gantt_robot, colors=gen_colors, index_col='Resource',
#                      show_colorbar=True, showgrid_x=True, showgrid_y=True, group_tasks=True)

# following doesn't work in pycharm.
# work in jupyter notebook
# fig.show()
'''
