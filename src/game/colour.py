from enum import Enum


class Colour(Enum):
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    PURPLE = (128, 0, 128)
    BLUE = (0, 0, 255)
    GREY = (128, 128, 128)
    LIGHT_GREY = (211, 211, 211)
    TRANSPARENT_GREY = (200, 200, 200, 100)
    ORANGE = (255, 165, 0)
    YELLOW = (255, 255, 0)
    BEIGE = (255, 255, 224)