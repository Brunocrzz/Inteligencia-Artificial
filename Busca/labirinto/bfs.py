from collections import deque
from search_utils import find_symbol, get_neighbors

#Algoritmo BFS
def bfs(maze):
    # Estado inicial e Objetivo
    start = find_symbol(maze, "S")
    end = find_symbol(maze, "E")

    # Inicializa listas e fila para guardar nós 
    queue = deque([start])
    visited = set()
    exploration_order = []

    # Adiciona o ponto inicial na fila e marca como visitado
    visited.add(start)
    exploration_order.append(start)

    parent = {}

    while queue:
        current = queue.popleft()

        # Para se encontrar o objetivo
        if current == end:
            break

        # Explora os vizinhos válidos ainda não visitados.
        for neighbor in get_neighbors(maze, current):
            if neighbor not in visited:
                visited.add(neighbor)
                exploration_order.append(neighbor)

                # Guarda de onde o nó veio para reconstruir o caminho final.
                parent[neighbor] = current
                queue.append(neighbor)
    
    path = []
    current = end

    # Reconstrói o caminho do objetivo até o início usando os pais.
    while current != start:
        path.append(current)
        current = parent[current]

    path.append(start)
    # Inverte o caminho para ficar do início até o fim.
    path.reverse()

    return {
        "path": path,
        "visited_nodes": len(visited),
        "path_length": len(path),
        "exploration_order": exploration_order
    }

def mark_path(maze, path):

    for row, col in path:
        # Marca visualmente o caminho encontrado no labirinto, usado para visualização
        if maze[row][col] not in ['S', 'E']:
            maze[row][col] = '.'        