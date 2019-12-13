import simpy
import random
import threading
from datetime import datetime, date, time, timedelta

# for import plotly, Dash should be installed.
#    pip install dash==1.7.0
#    pip install numpy
import plotly.figure_factory as ff


FIRST_DATE = date(2010, 1, 1)


class GenericArm:
    items = list()
    _capacity = 1
    _key_lock = threading.Lock()

    def __init__(self, env, capacity):
        self._env = env
        self._capacity = capacity
        self.monitoring_data = []

    def put(self, item):
        ret = False
        with self._key_lock:
            if self.items.__len__() < self._capacity:
                self.items.append(item)
                ret = True
            else:
                ret = False

        self.monitoring_data.append((self._env.now, item.value['id']))

        return ret

    def get(self, state):
        ret = False
        with self._key_lock:
            for item in self.items:
                if self.items.__len__() == 0:
                    ret = False
                if item.value['state'] == state:
                    buffer = item
                    try:
                        self.items.remove(item)
                        ret = buffer
                    except ValueError:
                        print("Error to remove")

        self.monitoring_data.append((self._env.now, ret.value['id']))

        return ret

    def clear(self):
        self.items.clear()
        return True

    def count(self):
        return self.items.__len__()


def wafer_producer(env, robotram, wafers, events):
    for i in range(TOT_WAFERS):
        yield robotram.put(wafers[i])
        print(env.now, 'Produced wafer', wafers[i])
        yield env.timeout(2)
        events['produced'].succeed(value='produced')
        # events['produced'] = env.event()  # update triggered old event as new one.
        yield env.timeout(15)


def handler(env, entry, robotArm, chamber_1st, chamber_2nd, events):
    while True:
        # if robot arm slot is available...
        # yield env.timeout(1)
        action = yield events['produced'] | events['consumed'] | events['chamber1 finished'] | \
                       events['chamber1 requested'] | events['chamber2 finished'] | events['chamber2 requested'] | \
                       events['handler']

        fromwho = action.events[0].value

        if fromwho == 'chamber1 requested':  # deliver chamber 2 wafer to arm storage
            events['chamber1 requested'] = env.event()  # update triggered old event as new one.
            yield env.timeout(2)
            wafer = robotArm.get('raw')
            wafer.value['state'] = '1st'
            yield chamber_1st.put(wafer)
            print(env.now, 'from chamber1 requested 1st chamber processing on %s' % wafer.value)
            events['chamber1 finished'].succeed(value='chamber1 finished')
            yield env.timeout(wafer.value['time_ch1'])

        elif fromwho == 'chamber1 finished':  # deliver chamber1 wafer to arm storage
            events['chamber1 finished'] = env.event()  # update triggered old event as new one.
            wafer = yield chamber_1st.get()
            print(env.now, 'from chamber1 finished on %s' % wafer.value)
            yield env.timeout(2)
            robotArm.put(wafer)
            if not events['handler'].triggered:
                events['handler'].succeed(value='handler')

        elif fromwho == 'chamber2 requested':  # deliver chamber 2 wafer to arm storage
            events['chamber2 requested'] = env.event()  # update triggered old event as new one.
            wafer = robotArm.get("1st")
            wafer.value['state'] = 'complete'
            yield env.timeout(2)
            yield chamber_2nd.put(wafer)
            print(env.now, 'from chamber2 requested', wafer.value)
            events['chamber2 finished'].succeed(value='chamber2 finished')
            yield env.timeout(wafer.value['time_ch2'])

        elif fromwho == 'chamber2 finished':  # deliver chamber1 wafer to arm storage
            # print(env.now, 'from chamber2 finished')
            events['chamber2 finished'] = env.event()  # update triggered old event as new one.
            wafer = yield chamber_2nd.get()
            robotArm.put(wafer)
            print(env.now, 'from chamber2 finished', wafer.value)
            events['consumed'].succeed(value='consumed')
            yield env.timeout(2)

        elif fromwho == 'consumed':
            events['consumed'] = env.event()
            wafer = robotArm.get('complete')
            print(env.now, 'from consumer, consumed wafer is %s' % wafer.value)
            yield env.timeout(2)
            if not events['handler'].triggered:
                events['handler'].succeed(value='handler')
            # break

        elif fromwho == 'handler':
            events['handler'] = env.event()  # update triggered old event as new one.
            if robotArm.count() == 0:
                print(env.now, 'from handler')
                events['produced'].succeed(value='produced')

            for item in robotArm.items:  # need to update wafer selecting algorithms
                # information to be retrieved : wafer elapsed time, etc.
                if item.value['state'] == 'complete':
                    print(env.now, 'handler moving wafer to complete', item.value)
                    yield env.timeout(2)
                    events['consumed'].succeed(value='consumed')

                elif item.value['state'] == '1st':
                    yield env.timeout(2)
                    print(env.now, 'handler moving to 2nd chamber ', item.value)
                    events['chamber2 requested'].succeed(value='chamber2 requested')

                elif item.value['state'] == 'raw':
                    yield env.timeout(2)
                    print(env.now, 'handler moving to 1st chamber ', item.value)
                    events['chamber1 requested'].succeed(value='chamber1 requested')

        elif fromwho == 'produced':
            # print(env.now, 'from producer')
            events['produced'] = env.event()  # update triggered old event as new one.
            if robotArm.count() == 0:
                wafer = entry.get()
                yield env.timeout(2)
                robotArm.put(wafer)
                print (env.now, 'from producer get wafer from entry', wafer.value)
            else:
                print(env.now, 'from producer do nothing, robot arm is full.')

            if not events['handler'].triggered:
                events['handler'].succeed(value='handler')


class MonitoredStore(simpy.Store):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = []

    def put(self, *args, **kwargs):
        self.data.append((self._env.now, args[0].value['id']))
        return super().put(*args, **kwargs)

    def get(self, *args, **kwargs):
        ret = super().get(*args, **kwargs)
        self.data.append((self._env.now, ret.value.value['id']))
        return ret


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


TOT_WAFERS = 20
env = simpy.Environment()
wafers = list()
for i in range(TOT_WAFERS):
    wafers.append({'id': i, 'state': 'raw', 'time_ch1': random.randint(3, 10), 'time_ch2': random.randint(2, 6)})
events_names = ['produced', 'consumed', 'chamber1 finished', 'chamber1 requested', 'chamber2 finished',
                'chamber2 requested', 'handler']
events = {event: env.event() for event in events_names}
entry = simpy.Store(env, capacity=1)
RobotArm = GenericArm(env, capacity=2)
Chamber_1st = MonitoredStore(env, capacity=2)
Chamber_2nd = MonitoredStore(env, capacity=2)

prod = env.process(wafer_producer(env, entry, wafers, events))
robotHandler = env.process(handler(env, entry, RobotArm, Chamber_1st, Chamber_2nd, events))
env.run(until=515)

print(Chamber_1st.data)
print(Chamber_2nd.data)
print(RobotArm.monitoring_data)

data_gantt_ch1 = convert_to_gantt_data(Chamber_1st.data, 'Chamber1', 'Wafer')
data_gantt_ch2 = convert_to_gantt_data(Chamber_2nd.data, 'Chamber2', 'Wafer')
data_gantt_robot = convert_to_gantt_data(RobotArm.monitoring_data, 'Robot', 'Wafer')

gen_colors = generate_colors('Wafer', 100)

fig = ff.create_gantt(data_gantt_ch1 + data_gantt_ch2 + data_gantt_robot, colors=gen_colors, index_col='Resource', show_colorbar=True, showgrid_x=True, showgrid_y=True, group_tasks=True)

# following doesn't work in pycharm.
# work in jupyter notebook
# fig.show()
