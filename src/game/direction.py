from enum import Enum
import numpy as np


class Direction(Enum):
    RIGHT = np.array([1,0])
    DOWN = np.array([0,1])
    LEFT = np.array([-1,0])
    UP = np.array([0,-1])