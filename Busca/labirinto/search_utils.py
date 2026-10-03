def find_symbol(maze, symbol):

    # Percorre todo o labirinto até encontrar o símbolo desejado, como o ponto inicial "S" ou objetivo "E".
    for row in range(len(maze)):
        for col in range(len(maze[row])):

            if maze[row][col] == symbol:
                return (row, col)

    return None

def is_valid_move(maze, row, col):

    height = len(maze)
    width = len(maze[0])

    # Verifica se a posição está dentro dos limites do labirinto.
    if not (0 <= row < height and 0 <= col < width):
        return False

    # Impede movimentação em paredes.
    if maze[row][col] == '#':
        return False

    return True

def get_neighbors(maze, position):
    row, col = position

    # Define os movimentos possíveis: cima, baixo, esquerda e direita.
    directions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1)
    ]

    neighbors = []

    for dr, dc in directions:
        new_row = row + dr
        new_col = col + dc

        # Adiciona apenas vizinhos válidos e acessíveis.
        if is_valid_move(maze, new_row, new_col):
            neighbors.append((new_row, new_col))

    return neighbors