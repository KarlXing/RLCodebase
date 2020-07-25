import numpy as np
from collections import deque


class Logger:
    def __init__(self, writer, num_echo_episodes):
        self.writer = writer
        assert(num_echo_episodes >= 0)
        self.last_rewards = deque(maxlen=num_echo_episodes)

    def save_episodic_return(self, infos, done_steps):
        for (i, info) in enumerate(infos):
            if info['episodic_return'] is not None:
                if self.writer is not None:
                    self.writer.add_scalar('episodic_return', info['episodic_return'], done_steps+i)
                self.last_rewards.append(info['episodic_return'])

    def print_last_rewards(self):
        avg_return = 'None' if len(self.last_rewards) == 0 else str(np.mean(self.last_rewards))
        print ('Average episodic return of %d episodes is: %s' % (len(self.last_rewards), avg_return))


    def add_scalar(self, tags, values, step):
        if self.writer is None:
            return
        for (i, tag) in enumerate(tags):
            self.writer.add_scalar(tag, values[i], step if isinstance(step, int) else step[i])


    def close(self):
        self.writer.close()

class MultiDeque:
    def __init__(self, tags = None):
        self.tags = tags
        if self.tags is not None:
            self.queues = [deque() for _ in range(len(tags))]
        else:
            self.queues = None

    def add(self, data):
        if self.queues is None:
            self.queues = [deque() for _ in range(len(data))]
        for i in range(len(data)):
            self.queues[i].append(data[i])

    def clear(self):
        for q in self.queues:
            q.clear()

    def set_tags(self, tags):
        self.tags = tags

    def return_summary(self):
        values  = [np.mean(q) for q in self.queues]
        self.clear()
        return self.tags, values

    # could only write if self.tags exists
    def write(self, writer, step):
        if writer is None:
            return
        assert(self.tags is not None)
        result = [np.mean(q) for q in self.queues]
        for i, r in enumerate(result):
            writer.add_scalar(self.tags[i], result[i], step)