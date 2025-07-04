from enum import Enum


class TileBGColour(Enum):
    T0 = "GREY"
    T2 = "BEIGE"
    T4 = "BEIGE"
    T8 = "GREEN"
    T16 = "RED"
    T32 = "PURPLE"
    T64 = "BLUE"
    T128 = "ORANGE"
    T256 = "YELLOW"
    T512 = "GREEN"
    T1024 = "RED"
    T2048 = "BLACK"
    T4096 = "BLACK"
    T8192 = "BLACK"
    T16384 = "BLACK"
    T32768 = "BLACK"
    T65536 = "BLACK"
    T131072 = "BLACK"
    
    @classmethod
    def get_color_for_value(cls, value: int) -> str:
        try:
            return getattr(cls, f"T{value}").value
        except AttributeError:
            return "GREY"
        
class TileFontColour(Enum):
    T0 = "WHITE"
    T2 = "BLACK"
    T4 = "BLACK"
    T8 = "BLACK"
    T16 = "WHITE"
    T32 = "WHITE"
    T64 = "WHITE"
    T128 = "BLACK"
    T256 = "BLACK"
    T512 = "BLACK"
    T2048 = "WHITE"
    T4096 = "WHITE"
    T8192 = "WHITE"
    T16384 = "WHITE"
    T32768 = "WHITE"
    T65536 = "WHITE"
    T131072 = "WHITE"
    
    @classmethod
    def get_color_for_value(cls, value: int) -> str:
        try:
            return getattr(cls, f"T{value}").value
        except AttributeError:
            return "WHITE"