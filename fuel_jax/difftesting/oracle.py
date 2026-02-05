from enum import Enum, auto


class Oracle(Enum):
    CONSISTENCY = auto
    INCONSISTENCY = auto
    CRASH = auto
