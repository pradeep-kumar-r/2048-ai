from enum import Enum
import numpy as np


class Direction(Enum):
    RIGHT = (1, 0)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    UP = (0, -1)
    
    def to_numpy(self):
        return np.array(self.value)