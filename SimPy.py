import simpy
import random


def wafer_producer(env, robotram, wafers, events):
    for i in range(TOT_WAFERS):
        yield robotram.put(wafers[i])
        print(env.now, 'Produced wafer', wafers[i])
        yield env.timeout(2)
        events['produced'].succeed(value='produced')
        events['produced'] = env.event() # update triggered old event as new one.
        yield env.timeout(50)



def handler(env, robotArm, chamber_1st, chamber_2nd, events):
    while True:
        #if robot arm slot is available...
        yield env.timeout(1)
        action = yield events['produced'] | events['consumed'] | events['chamber1 finished'] | \
                       events['chamber1 requested'] | events['chamber2 finished'] | events['chamber2 requested'] | \
                       events['handler']
        fromwho = (action.events[0].value)

        if fromwho == 'chamber1 requested':# deliver chamber 2 wafer to arm storage
            events['chamber1 requested'] = env.event()  # update triggered old event as new one.
            wafer = dict()
            for item in robotArm.items:
                if item['state'] == 'raw':
                    wafer = yield robotArm.get()
                    wafer['state'] = '1st'
                    break
            yield env.timeout(2)
            yield chamber_1st.put(wafer)
            print(env.now, 'from chamber1 requested 1st chamber processing on %s' % wafer)
            events['chamber1 finished'].succeed(value='chamber1 finished')

            yield env.timeout(wafer['time_ch1'])

        elif fromwho == 'chamber1 finished':#deliver chamber1 wafer to arm storage
            events['chamber1 finished'] = env.event()  # update triggered old event as new one.
            wafer = dict()
            for item in chamber_1st.items:
                if item['state'] == '1st':
                    wafer = yield chamber_1st.get()
                    break
            print(env.now, 'from chamber1 finished on %s' % wafer)
            yield env.timeout(2)
            yield robotArm.put(wafer)
            events['handler'].succeed(value='handler')

        elif fromwho == 'chamber2 requested':  # deliver chamber 2 wafer to arm storage
            events['chamber2 requested'] = env.event()# update triggered old event as new one.
            wafer = dict()
            for item in robotArm.items:
                if item['state'] == '1st':
                    wafer = yield robotArm.get()
                    wafer['state'] = 'complete'
                    break
            yield env.timeout(2)
            yield chamber_2nd.put(wafer)
            print(env.now, 'from chamber2 requested', wafer)
            events['chamber2 finished'].succeed(value='chamber2 finished')
            yield env.timeout(wafer['time_ch2'])

        elif fromwho == 'chamber2 finished':  # deliver chamber1 wafer to arm storage
            #print(env.now, 'from chamber2 finished')
            events['chamber2 finished'] = env.event()# update triggered old event as new one.
            wafer = dict()
            for item in chamber_2nd.items:
                if item['state'] == 'complete':
                    wafer = yield chamber_2nd.get()
                    break
            yield robotArm.put(wafer)
            print(env.now, 'from chamber2 finished', wafer)
            events['consumed'].succeed(value='consumed')
            yield env.timeout(2)

        elif fromwho == 'consumed':
            events['consumed'] = env.event()
            wafer = dict()
            for item in robotArm.items:
                if item['state'] == 'complete':
                    wafer = yield robotArm.get()
                    print(env.now, 'from consumer, consumed wafer is %s' % wafer)
                    break
            yield env.timeout(2)
            #yield robotArm.get(wafer)

        elif fromwho == 'handler':
            print(env.now, 'from handler')
            events['handler'] = env.event()  # update triggered old event as new one.
            for item in robotArm.items:
                if item['state'] == 'complete':
                    print(env.now, 'handler moving wafer to complete', item)
                    yield env.timeout(2)
                    events['consumed'].succeed(value='consumed')
                    break

                elif item['state'] == '1st':
                    yield env.timeout(2)
                    print(env.now, 'handler moving to 2nd chamber ', item)
                    events['chamber2 requested'].succeed(value='chamber2 requested')
                    break

                elif item['state'] == 'raw':
                    yield env.timeout(2)
                    print(env.now, 'handler moving to 1st chamber ', item)
                    events['chamber1 requested'].succeed(value='chamber1 requested')
                    break

        elif fromwho == 'produced':
            print(env.now, 'from producer')
            yield env.timeout(2)
            events['handler'].succeed(value='handler')

TOT_WAFERS = 20
env = simpy.Environment()
wafers = list()
for i in range(TOT_WAFERS):
    wafers.append({'id': i, 'state': 'raw', 'time_ch1': random.randint(3,10), 'time_ch2' : random.randint(2,6)})
eventsname = ['produced', 'consumed', 'chamber1 finished', 'chamber1 requested', 'chamber2 finished',
              'chamber2 requested', 'handler']
events = {event: env.event() for event in eventsname}
RobotArm = simpy.Store(env, capacity=2)
Chamber_1st = simpy.Store(env, capacity=2)
Chamber_2nd = simpy.Store(env, capacity=2)

prod = env.process(wafer_producer(env, RobotArm, wafers, events))
robotHandler = env.process(handler(env, RobotArm, Chamber_1st, Chamber_2nd, events))
env.run()
