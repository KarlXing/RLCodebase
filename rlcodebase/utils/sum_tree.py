import numpy as np

# SumTree
# Adapted from https://github.com/rlcode/per/blob/master/SumTree.py
class SumTree:

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # update priority
    def update(self, data_idx, p):
        tree_idx = data_idx + self.capacity - 1
        change = p - self.tree[tree_idx]

        self.tree[tree_idx] = p
        self._propagate(tree_idx, change)

    # get priority and sample
    def get(self, s):
        tree_idx = self._retrieve(0, s)
        data_idx = tree_idx - self.capacity + 1

        return tree_idx, self.tree[tree_idx], data_idx