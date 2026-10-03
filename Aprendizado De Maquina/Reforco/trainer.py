from environment import Environment, Action
from agent import QLearningAgent

class Trainer:

    def __init__(self, episodes=1000):
        self.episodes = episodes
        self.environment = Environment()
        self.agent = QLearningAgent(actions=list(Action))
        self.episode_rewards = []
        self.episode_steps = []
        self.success_rate = []

    def train(self):
        for episode in range(self.episodes):
            state = self.environment.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.agent.choose_action(state)
                next_state, reward, done, info = self.environment.step(action)
                self.agent.update_q_value(
                    state,
                    action,
                    reward,
                    next_state
                )
                state = next_state
                total_reward += reward

            self.agent.decay_exploration()
            self.episode_rewards.append(total_reward)
            self.episode_steps.append(info["steps"])
            self.success_rate.append(info["cell"] == "TREASURE")

            if (episode + 1) % 200 == 0:
                print(f"Episódio {episode+1}/{self.episodes}")
                print(f"Recompensa: {total_reward}")
                print(f"ε = {self.agent.exploration_rate:.3f}")    

            self.agent.save()    

    def evaluate(self):
        state = self.environment.reset()
        done = False

        while not done:
            self.agent.print_state_q_values(state)
            action = self.agent.get_policy(state)
            state, reward, done, info = self.environment.step(action)
            self.environment.render()  

    def print_state_q_values(self, state):

        q_values = self._get_q_values(state)

        print(f"\nEstado: {state}")

        for action, value in q_values.items():
            print(f"{action.name:<6}: {value:.2f}")

    def evaluate1(self):

        print("\n========== AVALIAÇÃO ==========\n")

        state = self.environment.reset()

        # Desliga completamente a exploração
        old_epsilon = self.agent.exploration_rate
        self.agent.exploration_rate = 0

        total_reward = 0
        done = False

        self.environment.render()

        while not done:

            self.agent.print_state_q_values(state)
            action = self.agent.get_policy(state)

            print(f"\nAção escolhida: {action.name}")

            next_state, reward, done, info = self.environment.step(action)

            total_reward += reward

            print(f"Estado: {next_state}")
            print(f"Recompensa: {reward}")
            print(f"Recompensa acumulada: {total_reward}")

            self.environment.render()

            state = next_state

        print("\n===============================")
        print("Fim da avaliação")
        print(f"Recompensa total: {total_reward}")
        print(f"Passos: {info['steps']}")
        print("===============================\n")

        # Restaura o epsilon original
        self.agent.exploration_rate = old_epsilon      