from map_generator import create_fixed_map, CellType
from config import (
    REWARD_MOVE,
    REWARD_WALL,
    REWARD_TRAP,
    REWARD_COIN,
    REWARD_KEY,
    REWARD_DOOR,
    REWARD_TREASURE,
    MAX_STEPS
)
from enum import Enum

class Action(Enum):
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)


class Environment:   
    def __init__(self):
        self.reset()


    def reset(self):
        self.map = create_fixed_map()
        self.agent_row = 1
        self.agent_col = 1
        self.has_key = False
        self.score = 0
        self.steps = 0
        self.done = False
        self.coins_collected = 0

        return self.get_state() 


    def get_state(self):
        return (
            self.agent_row,
            self.agent_col,
            self.has_key
        )


    def step(self, action):

        if self.done:
            info = {
                "position": (self.agent_row, self.agent_col),
                "cell": "FINISHED",
                "score": self.score,
                "steps": self.steps,
                "has_key": self.has_key
            }

            return self.get_state(), 0, True, info

        reward = REWARD_MOVE

        new_row, new_col = self._calculate_new_position(action)

        if not self._can_move(new_row, new_col):
            reward = REWARD_WALL
            info = {
                        "position": (self.agent_row, self.agent_col),
                        "cell": "WALL",
                        "score": self.score,
                        "steps": self.steps,
                        "has_key": self.has_key
                    }
            self.score += reward
            return self.get_state(), reward, self.done, info

        self.agent_row = new_row
        self.agent_col = new_col

        reward += self._calculate_reward()

        self.steps += 1

        if self.steps >= MAX_STEPS:
            self.done = True

        cell = self.map[self.agent_row][self.agent_col]
        info = {
                    "position": (self.agent_row, self.agent_col),
                    "cell": cell.name,
                    "score": self.score,
                    "steps": self.steps,
                    "has_key": self.has_key
                }

        return self.get_state(), reward, self.done, info

    def render(self):
        symbols = {
            CellType.EMPTY: ".",
            CellType.WALL: "#",
            CellType.TRAP: "X",
            CellType.COIN: "$",
            CellType.KEY: "K",
            CellType.DOOR: "D",
            CellType.TREASURE: "T",
            CellType.EMPTY_DOOR: "E"
        }

        print()

        for r, row in enumerate(self.map):

            line = ""

            for c, cell in enumerate(row):

                if r == self.agent_row and c == self.agent_col:
                    line += "A"
                else:
                    line += symbols[cell]

            print(line)

        print(f"\nScore: {self.score}")
        print(f"Key: {self.has_key}")
        print(f"Steps: {self.steps}")
        print(f"Coins Collected: {self.coins_collected}")

    # ==========================
    # Métodos privados
    # ==========================

    def _calculate_new_position(self, action):
        dr, dc = action.value
        return self.agent_row + dr, self.agent_col + dc

    def _can_move(self, row, col):
        cell = self.map[row][col]

        if cell == CellType.WALL:
            return False

        if cell == CellType.DOOR and not self.has_key:
            return False

        return True

    def _calculate_reward(self):
        cell = self.map[self.agent_row][self.agent_col]
        reward = 0

        if cell == CellType.TRAP:
            reward += REWARD_TRAP

        elif cell == CellType.COIN:
            reward += REWARD_COIN
            self.coins_collected += 1
            self.map[self.agent_row][self.agent_col] = CellType.EMPTY

        elif cell == CellType.KEY:
            reward += REWARD_KEY
            self.has_key = True
            self.map[self.agent_row][self.agent_col] = CellType.EMPTY

        elif cell == CellType.DOOR:
            reward += REWARD_DOOR
            self.has_key = False
            self.map[self.agent_row][self.agent_col] = CellType.EMPTY_DOOR

        elif cell == CellType.TREASURE:
            reward += REWARD_TREASURE
            self.done = True

        self.score += reward

        return reward