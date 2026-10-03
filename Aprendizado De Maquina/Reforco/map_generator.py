from enum import Enum


class CellType(Enum):
    EMPTY = 0
    WALL = 1
    TRAP = 2
    COIN = 3
    KEY = 4
    DOOR = 5
    TREASURE = 6
    EMPTY_DOOR = 7

def create_fixed_map():
    return [
        [CellType.WALL,CellType.WALL,CellType.WALL,CellType.WALL,CellType.WALL,CellType.WALL,CellType.WALL,CellType.WALL,CellType.WALL,CellType.WALL],
        [CellType.WALL,CellType.EMPTY,CellType.EMPTY,CellType.EMPTY,CellType.EMPTY,CellType.EMPTY,CellType.COIN,CellType.EMPTY,CellType.EMPTY,CellType.WALL],
        [CellType.WALL,CellType.EMPTY,CellType.WALL,CellType.WALL,CellType.EMPTY,CellType.WALL,CellType.WALL,CellType.EMPTY,CellType.EMPTY,CellType.WALL],
        [CellType.WALL,CellType.EMPTY,CellType.COIN,CellType.TRAP,CellType.EMPTY,CellType.COIN,CellType.EMPTY,CellType.EMPTY,CellType.EMPTY,CellType.WALL],
        [CellType.WALL,CellType.WALL,CellType.WALL,CellType.EMPTY,CellType.EMPTY,CellType.WALL,CellType.EMPTY,CellType.WALL,CellType.EMPTY,CellType.WALL],
        [CellType.WALL,CellType.EMPTY,CellType.EMPTY,CellType.EMPTY,CellType.KEY,CellType.EMPTY,CellType.EMPTY,CellType.COIN,CellType.EMPTY,CellType.WALL],
        [CellType.WALL,CellType.EMPTY,CellType.WALL,CellType.WALL,CellType.WALL,CellType.EMPTY,CellType.WALL,CellType.WALL,CellType.EMPTY,CellType.WALL],
        [CellType.WALL,CellType.EMPTY,CellType.EMPTY,CellType.EMPTY,CellType.DOOR,CellType.TRAP,CellType.EMPTY,CellType.EMPTY,CellType.EMPTY,CellType.WALL],
        [CellType.WALL,CellType.COIN,CellType.EMPTY,CellType.EMPTY,CellType.EMPTY,CellType.EMPTY,CellType.TREASURE,CellType.EMPTY,CellType.EMPTY,CellType.WALL],
        [CellType.WALL,CellType.WALL,CellType.WALL,CellType.WALL,CellType.WALL,CellType.WALL,CellType.WALL,CellType.WALL,CellType.WALL,CellType.WALL],
    ]

