import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum

# Valores de recompensa e penalidade para cada ação e evento no ambiente
MAP_WIDTH = 10
MAP_HEIGHT = 10
REWARD_MOVE = -1
REWARD_WALL = -10
REWARD_TRAP = -50 
REWARD_COIN = 30
REWARD_KEY = 35
REWARD_DOOR = 50
REWARD_TREASURE = 200
REWARD_EMPTY_DOOR = 0
MAX_STEPS = 200

# Classes enumeradas para representar os tipos de células no mapa e as ações possíveis do agente
class CellType(Enum):
    EMPTY = 0
    WALL = 1
    TRAP = 2
    COIN = 3
    KEY = 4
    DOOR = 5
    TREASURE = 6
    EMPTY_DOOR = 7

class Action(Enum):
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)

# Função para criar um mapa fixo com paredes, moedas, armadilhas, chave, porta e tesouro
def create_fixed_map():
    return [
        [CellType.WALL]*10,
        [CellType.WALL, CellType.EMPTY, CellType.EMPTY, CellType.EMPTY, CellType.EMPTY, CellType.EMPTY, CellType.COIN, CellType.EMPTY, CellType.EMPTY, CellType.WALL],
        [CellType.WALL, CellType.EMPTY, CellType.WALL, CellType.WALL, CellType.EMPTY, CellType.WALL, CellType.WALL, CellType.EMPTY, CellType.EMPTY, CellType.WALL],
        [CellType.WALL, CellType.EMPTY, CellType.COIN, CellType.TRAP, CellType.EMPTY, CellType.COIN, CellType.EMPTY, CellType.EMPTY, CellType.EMPTY, CellType.WALL],
        [CellType.WALL, CellType.WALL, CellType.WALL, CellType.EMPTY, CellType.EMPTY, CellType.WALL, CellType.EMPTY, CellType.WALL, CellType.EMPTY, CellType.WALL], # Bloqueio ###
        [CellType.WALL, CellType.EMPTY, CellType.EMPTY, CellType.EMPTY, CellType.KEY, CellType.EMPTY, CellType.EMPTY, CellType.COIN, CellType.EMPTY, CellType.WALL],
        [CellType.WALL, CellType.EMPTY, CellType.WALL, CellType.WALL, CellType.WALL, CellType.EMPTY, CellType.WALL, CellType.WALL, CellType.EMPTY, CellType.WALL],
        [CellType.WALL, CellType.EMPTY, CellType.EMPTY, CellType.EMPTY, CellType.DOOR, CellType.TRAP, CellType.EMPTY, CellType.EMPTY, CellType.EMPTY, CellType.WALL],
        [CellType.WALL, CellType.COIN, CellType.EMPTY, CellType.EMPTY, CellType.EMPTY, CellType.EMPTY, CellType.TREASURE, CellType.EMPTY, CellType.EMPTY, CellType.WALL],
        [CellType.WALL]*10
    ]

# Definição da classe Environment que gerencia o estado do ambiente, incluindo a posição do agente, a coleta de recompensa
class Environment:
    def __init__(self):
        self.reset()

    # reinicia o ambiente para o estado inicial
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

    # retorna o estado atual do agente, incluindo sua posição e se possui a chave
    def get_state(self):
        return (self.agent_row, self.agent_col, self.has_key)

    # executa uma ação no ambiente, atualizando o estado do agente e retornando a recompensa obtida
    def step(self, action):
        # Verifica se o episódio já terminou e retorna informações finais 
        if self.done:
            return self.get_state(), 0, True, self._get_info("FINISHED")

        new_row = self.agent_row + action.value[0]
        new_col = self.agent_col + action.value[1]

        # Verificação de colisão com estruturas proibidas
        cell_destino = self.map[new_row][new_col]
        if cell_destino == CellType.WALL or (cell_destino == CellType.DOOR and not self.has_key):
            self.score += REWARD_WALL
            self.steps += 1
            if self.steps >= MAX_STEPS: self.done = True
            return self.get_state(), REWARD_WALL, self.done, self._get_info("WALL")

        # Movimento validado com sucesso
        self.agent_row = new_row
        self.agent_col = new_col
        self.steps += 1
        
        # Processamento único de recompensas de transição
        step_reward = REWARD_MOVE
        cell = self.map[self.agent_row][self.agent_col]

        # Cálculo de recompensas baseado no tipo de célula visitada
        if cell == CellType.TRAP:
            step_reward += REWARD_TRAP
        elif cell == CellType.COIN:
            step_reward += REWARD_COIN
            self.coins_collected += 1
            self.map[self.agent_row][self.agent_col] = CellType.EMPTY
        elif cell == CellType.KEY:
            step_reward += REWARD_KEY
            self.has_key = True
            self.map[self.agent_row][self.agent_col] = CellType.EMPTY
        elif cell == CellType.DOOR:
            step_reward += REWARD_DOOR
            self.has_key = False
            self.map[self.agent_row][self.agent_col] = CellType.EMPTY_DOOR
        elif cell == CellType.TREASURE:
            step_reward += REWARD_TREASURE
            self.done = True

        # Atualiza a pontuação total do agente e verifica se o número máximo de passos foi atingido
        self.score += step_reward
        if self.steps >= MAX_STEPS: self.done = True

        return self.get_state(), step_reward, self.done, self._get_info(cell.name)

    # Retorna informações detalhadas sobre o estado atual do agente
    def _get_info(self, cell_name):
        return {
            "position": (self.agent_row, self.agent_col),
            "cell": cell_name,
            "score": self.score,
            "steps": self.steps,
            "has_key": self.has_key
        }

    # Renderiza o estado atual do ambiente no console, mostrando a posição do agente e os elementos do mapa
    def render(self):
        symbols = {
            CellType.EMPTY: ".", CellType.WALL: "#", CellType.TRAP: "X",
            CellType.COIN: "$", CellType.KEY: "K", CellType.DOOR: "D",
            CellType.TREASURE: "T", CellType.EMPTY_DOOR: "E"
        }
        print()
        for r, row in enumerate(self.map):
            line = ""
            for c, cell in enumerate(row):
                if r == self.agent_row and c == self.agent_col: line += "A"
                else: line += symbols[cell]
            print(line)
        print(f"Score Sincronizado: {self.score} | Key: {self.has_key} | Passos: {self.steps}")


# Classe QLearningAgent com métodos de aprendizado por reforço Q-Learning
class QLearningAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.999, min_exploration_rate=0.01):
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.q_table = {}

    def _get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}
        return self.q_table[state]

    def choose_action(self, state):
        if np.random.random() < self.exploration_rate:
            return np.random.choice(self.actions)
        
        q_values = self._get_q_values(state)
        max_q = max(q_values.values())
        # Filtro ativo para desempate aleatório justo entre os melhores valores Q
        best_actions = [action for action, val in q_values.items() if val == max_q]
        return np.random.choice(best_actions)

    def update_q_value(self, state, action, reward, next_state):
        q_atual = self._get_q_values(state)[action]
        max_futuro_q = max(self._get_q_values(next_state).values())
        # Equação Matemática Clássica de Bellman
        novo_q = q_atual + self.learning_rate * (reward + self.discount_factor * max_futuro_q - q_atual)
        self.q_table[state][action] = novo_q

    def decay_exploration(self):
        self.exploration_rate *= self.exploration_decay
        if self.exploration_rate < self.min_exploration_rate:
            self.exploration_rate = self.min_exploration_rate

    def get_policy(self, state):
        q_values = self._get_q_values(state)
        max_q = max(q_values.values())
        best_actions = [action for action, val in q_values.items() if val == max_q]
        return np.random.choice(best_actions)


# Classe de Treinamento que orquestra o ambiente e o agente Q-Learning
class Trainer:
    def __init__(self, episodes=5000):
        self.episodes = episodes
        self.environment = Environment()
        self.agent = QLearningAgent(actions=list(Action))
        self.history_rewards = []
        self.history_steps = []

    # Método de treinamento estruturado do agente no ambiente
    def train(self):
        print(f"Iniciando treinamento estruturado por {self.episodes} episódios")
        for episode in range(self.episodes):
            state = self.environment.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.agent.choose_action(state)
                next_state, reward, done, info = self.environment.step(action)
                self.agent.update_q_value(state, action, reward, next_state)
                state = next_state
                total_reward += reward

            self.agent.decay_exploration()
            self.history_rewards.append(total_reward)
            self.history_steps.append(info["steps"])

            if (episode + 1) % 1000 == 0 or (episode + 1) == 200:
                print(f" -> Ep {episode+1:4d}/{self.episodes} | Recompensa: {total_reward:4.0f} | ε: {self.agent.exploration_rate:.3f} | Passos: {info['steps']}")

    def evaluate(self):
        print("\n" + "-"*31 + "\n   Avaliação da rota encontrada\n")
        state = self.environment.reset()
        
        # Desliga a exploração para testar a rota determinística pura
        old_epsilon = self.agent.exploration_rate
        self.agent.exploration_rate = 0.0
        
        done = False
        self.environment.render()

        while not done:
            action = self.agent.get_policy(state)
            print(f"\nAção escolhida: {action.name}")
            state, reward, done, info = self.environment.step(action)
            self.environment.render()

        print("\n" + "-"*31)
        print(f"Resultado Final da Rota:")
        print(f"Recompensa Total: {info['score']}")
        print(f"Passos Efetuados: {info['steps']}")
        self.agent.exploration_rate = old_epsilon

    def plot_metrics(self):
        # Gera o painel de métricas 
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Cálculo da média móvel para suavizar as curvas do gráfico
        janela = 100
        media_movel_rec = np.convolve(self.history_rewards, np.ones(janela)/janela, mode='valid')
        media_movel_pas = np.convolve(self.history_steps, np.ones(janela)/janela, mode='valid')

        axes[0].plot(self.history_rewards, alpha=0.2, color='dodgerblue')
        axes[0].plot(media_movel_rec, color='blue', linewidth=2, label='Média Móvel (100 eps)')
        axes[0].set_title('Evolução das Recompensas Acumuladas', fontweight='bold')
        axes[0].set_xlabel('Episódio')
        axes[0].set_ylabel('Recompensa Total')
        axes[0].legend()

        axes[1].plot(self.history_steps, alpha=0.2, color='tomato')
        axes[1].plot(media_movel_pas, color='red', linewidth=2, label='Média Móvel (100 eps)')
        axes[1].set_title('Otimização do Número de Passos', fontweight='bold')
        axes[1].set_xlabel('Episódio')
        axes[1].set_ylabel('Passos até o Alvo')
        axes[1].legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    orchestrator = Trainer(episodes=5000)
    orchestrator.train()
    orchestrator.evaluate()
    orchestrator.plot_metrics()