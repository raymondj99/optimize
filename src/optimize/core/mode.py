from enum import Enum 

class Mode(Enum):
    FORWARD = 1
    BACKWARD = 2

class AutoDiffMode:
    _mode = Mode.FORWARD

    @classmethod
    def set_mode(cls, new_mode: Mode):
        if isinstance(new_mode, Mode):
            cls._mode = new_mode

    @classmethod
    def get_mode(cls) -> Mode:
        return cls._mode