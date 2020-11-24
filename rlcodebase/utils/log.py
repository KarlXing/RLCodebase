import numpy as np
from collections import deque


class Logger:
    def __init__(self, writer, num_echo_episodes, episodes_avg_window = -1):
        self.writer = writer
        assert(num_echo_episodes >= 0)
        self.last_rewards = deque(maxlen=num_echo_episodes)
        self.episodes_avg_window = episodes_avg_window
        self.window_rewards = []
        self.window_end = self.episodes_avg_window

    def save_episodic_return(self, infos, done_steps):
        for (i, info) in enumerate(infos):
            if info['episodic_return'] is not None:
                global_step = done_steps + i
                # save each episode return or save average episodic return in windows
                if self.episodes_avg_window == -1:
                    if self.writer is not None:
                        self.writer.add_scalar('episodic_return', info['episodic_return'], done_steps+i)
                else:
                    if global_step < self.window_end:
                        self.window_rewards.append(info['episodic_return']) 
                    else:
                        if self.writer is not None and len(self.window_rewards) != 0:
                            self.writer.add_scalar('episodic_return', np.mean(self.window_rewards), self.window_end)
                        self.window_rewards = [info['episodic_return']]
                        self.window_end = (global_step // self.episodes_avg_window + 1) * self.episodes_avg_window
                self.last_rewards.append(info['episodic_return'])

    def print_last_rewards(self):
        avg_return = 'None' if len(self.last_rewards) == 0 else str(np.mean(self.last_rewards))
        print ('Average episodic return of %d episodes is: %s' % (len(self.last_rewards), avg_return))


    def add_scalar(self, tags, values, step):
        if self.writer is None:
            return
        for (i, tag) in enumerate(tags):
            if values[i] is not None:
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