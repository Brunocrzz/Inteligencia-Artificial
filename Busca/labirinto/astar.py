import heapq
from search_utils import find_symbol, get_neighbors

# Heuristica utilizando distância Manhattan
def heuristic(position, goal):

    row1, col1 = position
    row2, col2 = goal

    return abs(row1 - row2) + abs(col1 - col2)

# Algoritmo A*
def astar(maze):
    #Guarda o inicio e objetivo final do labirinto
    start = find_symbol(maze, "S")
    goal = find_symbol(maze, "E")

    # Inicializa listas para guardar nós visitados e ordem de exploração para visualização posterior
    open_set = []
    closed_set = set()
    exploration_order = []
    heapq.heappush(open_set, (0, start))

    g_score = {}
    g_score[start] = 0

    parent = {}

    while open_set:

        current_f, current = heapq.heappop(open_set)

        # Ignora duplicados
        if current in closed_set:
            continue

        closed_set.add(current)
        exploration_order.append(current)
        closed_set.add(current)

        # Objetivo encontrado
        if current == goal:
            break

        for neighbor in get_neighbors(maze, current):

            if neighbor in closed_set:
                continue

            # Cada movimento aumenta 1 no custo
            tentative_g = g_score[current] + 1

            # Se o vizinho ainda não foi visitado ou um caminho melhor foi encontrado, atualiza seus custos e adiciona novamente na fila de prioridade.
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                parent[neighbor] = current
                heapq.heappush(open_set, (f_score, neighbor))

    # Reconstrução do labirinto
    path = []
    current = goal

    while current != start:
        path.append(current)
        current = parent[current]

    path.append(start)
    path.reverse()

    return {
        "path": path,
        "visited_nodes": len(closed_set),
        "path_length": len(path),
        "exploration_order": exploration_order,
    }
