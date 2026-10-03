# Inteligência Artificial — UnB/FCTE · 2026.1

Repositório com os trabalhos práticos que desenvolvi na disciplina **Inteligência Artificial (FGA0221)** da Universidade de Brasília, na FCTE, no semestre **2026.1**.

## Sobre a disciplina

A disciplina faz uma introdução ampla aos fundamentos da Inteligência Artificial: **representação do conhecimento**, **resolução de problemas por busca**, **raciocínio sob incerteza** e **aprendizado de máquina**. Ela passa pelos paradigmas simbólico, conexionista e probabilístico e também discute o uso ético da IA.

Cada tema teve aulas expositivas e um **projeto prático em Python**, e a avaliação foi feita por **portfólio**. Cada projeto deste repositório vem acompanhado do **relatório em PDF** que entreguei, explicando o problema, a modelagem, os algoritmos e os resultados.

## Estrutura do repositório

```
.
├── Busca/                     # Busca, CSP e sistemas baseados em conhecimento
│   ├── labirinto/             # BFS x A* em labirintos gerados proceduralmente
│   ├── TSP/                   # Caixeiro Viajante com Hill Climbing e Algoritmo Genético
│   ├── CSP/                   # Coloração do mapa do Brasil com Backtracking + MRV/Grau
│   └── conhecimento/          # Sistema especialista para diagnóstico de computadores
│
├── incerteza/                 # Raciocínio probabilístico
│   ├── Redes Bayesianas/      # Tomada de decisão de um NPC de combate
│   ├── Modelo Oculto de Markov/  # Reconhecimento do perfil de um jogador (Forward)
│   └── Filtro de Kalman/      # Rastreamento 2D de um veículo com GPS ruidoso
│
└── Aprendizado De Maquina/    # Machine Learning
    ├── Supervisionado/        # Detecção de fraude: Random Forest + SMOTE + GridSearch
    ├── Nao Supervisionado/    # Segmentação de clientes: K-Means x DBSCAN
    ├── Reforco/               # Agente Q-Learning em um mapa com chave, porta e tesouro
    └── Deep Learning/         # CNN (PyTorch) para o alfabeto da Língua de Sinais Americana
```

| Módulo | Projeto | Técnicas | Interface |
|---|---|---|---|
| [Busca](./Busca) | Labirinto | BFS, A*, geração por DFS | Pygame |
| [Busca](./Busca) | TSP | Hill Climbing, Algoritmo Genético | Pygame |
| [Busca](./Busca) | Mapa do Brasil | CSP, Backtracking, MRV, Grau, Flood Fill | Pygame |
| [Busca](./Busca) | Diagnóstico de PC | Base de conhecimento, Forward Chaining | Pygame |
| [Incerteza](./incerteza) | NPC de combate | Rede Bayesiana, inferência por enumeração | Matplotlib |
| [Incerteza](./incerteza) | Perfil do jogador | HMM, algoritmo Forward | Matplotlib |
| [Incerteza](./incerteza) | Veículo em "S" | Filtro de Kalman 6D | Matplotlib |
| [Aprendizado de Máquina](./Aprendizado%20De%20Maquina) | Fraude em cartão | Random Forest, SMOTE, GridSearchCV | Matplotlib/Seaborn |
| [Aprendizado de Máquina](./Aprendizado%20De%20Maquina) | Clientes de shopping | K-Means, Silhouette, DBSCAN | Matplotlib/Seaborn |
| [Aprendizado de Máquina](./Aprendizado%20De%20Maquina) | Caça ao tesouro | Q-Learning, ε-greedy | Terminal + Matplotlib |
| [Aprendizado de Máquina](./Aprendizado%20De%20Maquina) | Alfabeto ASL | CNN, Data Augmentation, BatchNorm | Matplotlib/Seaborn |

Cada uma das três pastas principais tem um `README.md` próprio, com a explicação de cada projeto, como rodar e os resultados obtidos.

Algumas pastas têm uma subpasta `exemplos/` com **códigos de exemplo apresentados em aula** pelo professor. Eles servem de referência e não fazem parte dos trabalhos entregues.

## Como rodar

Pré-requisito: **Python 3.10+**.

```bash
# 1. Clone o repositório
git clone https://github.com/<seu-usuario>/Inteligencia-Artificial.git
cd Inteligencia-Artificial

# 2. (Opcional) Crie um ambiente virtual
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# 3. Entre na pasta do projeto e instale as dependências
cd "incerteza/Filtro de Kalman"
pip install -r requirements.txt   # quando houver
python main.py
```

Os projetos de **Busca** usam apenas o Pygame. Recomenda-se a versão `pygame-ce` (`pip install pygame-ce`). As instruções específicas de cada projeto estão no README da pasta correspondente.

> Os projetos foram desenvolvidos e testados no **Windows**. Em outros sistemas operacionais, alguns exemplos de aula podem precisar de ajuste nos caminhos de arquivo.

## Relatórios

Cada projeto tem o PDF do relatório na própria pasta:

- `Busca/labirinto/Busca Cega e Busca Informada.pdf`
- `Busca/TSP/Busca Complexa e Alg. Genético.pdf`
- `Busca/CSP/CSP.pdf`
- `Busca/conhecimento/Banco de Conhecimento.pdf`
- `incerteza/Redes Bayesianas/Redes Bayesianas.pdf`
- `incerteza/Modelo Oculto de Markov/Modelo Oculto de Markov.pdf`
- `incerteza/Filtro de Kalman/Filtro de Kalman.pdf`
- `Aprendizado De Maquina/Supervisionado/Aprendizado Supervisionado.pdf`
- `Aprendizado De Maquina/Nao Supervisionado/Aprendizado não Supervisionado.pdf`
- `Aprendizado De Maquina/Reforco/Aprendizado por Reforço.pdf`
- `Aprendizado De Maquina/Deep Learning/Deep Learning.pdf`

---
