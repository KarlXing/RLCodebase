import numpy as np
from collections import deque

def log_rewards(writer, infos, done_steps, queue = None):
    if writer is None:
        return
    for (i, info) in enumerate(infos):
        if info['episodic_return'] is not None:
            writer.add_scalar('episodic_return', info['episodic_return'], done_steps+i)
            if queue is not None:
                queue.append(info['episodic_return'])

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

    # could only write if self.tags exists
    def write(self, writer, step):
        if writer is None:
            return
        assert(self.tags is not None)
        result = [np.mean(q) for q in self.queues]
        for i, r in enumerate(result):
            writer.add_scalar(self.tags[i], result[i], step)