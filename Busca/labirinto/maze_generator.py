import random

def create_maze(rows, cols):
    """
    Cria um labirinto usando DFS iterativo + backtracking.

    Parâmetros:
        rows -> quantidade de células na vertical
        cols -> quantidade de células na horizontal

    Retorna:
        maze -> matriz do labirinto
    """

    # Tamanho REAL do grid
    # (incluindo paredes)
    height = rows * 2 + 1
    width = cols * 2 + 1

    # Cria grid cheio de paredes
    maze = [['#' for _ in range(width)] for _ in range(height)]

    # Cria células válidas
    for row in range(1, height, 2):
        for col in range(1, width, 2):
            maze[row][col] = ' '

    # Movimentos possíveis
    directions = [
        (-2, 0),  # cima
        (2, 0),   # baixo
        (0, -2),  # esquerda
        (0, 2)    # direita
    ]

    # Estruturas do DFS
    stack = []
    visited = set()

    # Começa em (1,1)
    start = (1, 1)

    stack.append(start)
    visited.add(start)

    # DFS iterativo
    while stack:

        current_row, current_col = stack[-1]

        neighbors = []

        # Procura vizinhos válidos
        for dr, dc in directions:

            new_row = current_row + dr
            new_col = current_col + dc

            if (
                1 <= new_row < height - 1 and
                1 <= new_col < width - 1 and
                (new_row, new_col) not in visited
            ):
                neighbors.append((new_row, new_col))

        # Se encontrou vizinhos
        if neighbors:

            # Escolhe vizinho aleatório
            next_row, next_col = random.choice(neighbors)

            # Calcula parede entre células
            wall_row = (current_row + next_row) // 2
            wall_col = (current_col + next_col) // 2

            # Remove parede
            maze[wall_row][wall_col] = ' '

            # Marca visitado
            visited.add((next_row, next_col))

            # Continua DFS
            stack.append((next_row, next_col))

        else:
            # Backtracking
            stack.pop()

    # Entrada aleatória (lado esquerdo)
    entrance_row = random.randrange(1, height, 2)

    # Saída aleatória (lado direito)
    exit_row = random.randrange(1, height, 2)

    # Marca entrada e saída
    maze[entrance_row][0] = 'S'
    maze[exit_row][width - 1] = 'E'

    return maze

def add_loops(maze, loop_density):

    height = len(maze)
    width = len(maze[0])

    density = loop_density / 100.0  # chance de remover uma parede
    loop_count = int(height * width * density)

    for _ in range(loop_count):

        row = random.randint(1, height - 2)
        col = random.randint(1, width - 2)

        # Precisa ser parede
        if maze[row][col] != '#':
            continue

        # Parede vertical
        if (
            maze[row - 1][col] != '#' and
            maze[row + 1][col] != '#'
        ):
            maze[row][col] = ' '

        # Parede horizontal
        elif (
            maze[row][col - 1] != '#' and
            maze[row][col + 1] != '#'
        ):
            maze[row][col] = ' '

def create_maze_with_loops(rows, cols, loop_density):
    maze = create_maze(rows, cols)

    add_loops(maze, loop_density)

    return maze           