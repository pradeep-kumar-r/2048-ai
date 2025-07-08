import random
import collections


Transition = collections.namedtuple('Transition',
                                ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.memory = collections.deque([], maxlen=capacity)

    def push(self, *args) -> None:
        """Save a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> list:
        if len(self.memory) < batch_size:
            return self.memory
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)