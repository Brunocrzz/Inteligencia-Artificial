from environment import Action
from numpy import random
import pickle

class QLearningAgent():
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.998, min_exploration_rate=0.05):
        self.actions = actions
        self.learning_rate = learning_rate          # α
        self.discount_factor = discount_factor      # γ
        self.exploration_rate = exploration_rate    # ε
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.q_table = {}


    def _get_q_values(self, state):
        if state not in self.q_table:

            self.q_table[state] = {
                action: 0.0
                for action in self.actions
            }

        return self.q_table[state]


    def get_q_value(self, state, action):
        return self._get_q_values(state)[action]


    def choose_action(self, state):
        if random.random() < self.exploration_rate:
            return random.choice(self.actions)
        
        q_values = self._get_q_values(state)
        max_q = max(q_values.values())
        
        best_actions = [action for action, value in q_values.items() if value == max_q]
        return random.choice(best_actions)

    def update_q_value(self, state, action, reward, next_state):
        q = self.get_q_value(state, action)
        best_next_q = max(self._get_q_values(next_state).values())
        new_q = q + self.learning_rate * (reward + self.discount_factor * best_next_q - q)
        self.q_table[state][action] = new_q
        best_next_q = max(self._get_q_values(next_state).values())


    def decay_exploration(self):
        self.exploration_rate *= self.exploration_decay
        if self.exploration_rate < self.min_exploration_rate:
            self.exploration_rate = self.min_exploration_rate


    def get_policy(self, state):
        q_values = self._get_q_values(state)
        max_q = max(q_values.values())
        
        best_actions = [action for action, value in q_values.items() if value == max_q]
        return random.choice(best_actions)


    def reset_q_table(self):
        self.q_table = {}


    def save(self, filename="q_table.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)   


    def load(self, filename="q_table.pkl"):
        with open(filename, "rb") as file:
            self.q_table = pickle.load(file)
        

    def print_state_q_values(self, state):

        q_values = self._get_q_values(state)

        print(f"\nEstado: {state}")

        for action, value in q_values.items():
            print(f"{action.name:<6}: {value:.2f}")    