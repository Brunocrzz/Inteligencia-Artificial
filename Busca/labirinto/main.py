import time
from maze_generator import create_maze_with_loops
from bfs import bfs
from astar import astar
from pygame_visualizer import visualize_maze
import copy

#Cria um labirinto 
maze = create_maze_with_loops(30, 30, 3)

# Faz uma cópia de cada labirinto para ambos algoritmos rodarem no mesmo labirinto
astar_maze = copy.deepcopy(maze)
bfs_maze = copy.deepcopy(maze)

# Guarda o tempo de execução de cada
start_time = time.perf_counter()
bfs_result = bfs(bfs_maze) # Execução BFS
end_time = time.perf_counter()
bfs_execution_time = end_time - start_time

start_time = time.perf_counter()
astar_result = astar(astar_maze) # Execução A*
end_time = time.perf_counter()
astar_execution_time = end_time - start_time

# Printa os resultados obtidos
print("\nResultado BFS - Labirinto:")
print("-" * 25)

print(f"Tempo de Execução: {bfs_execution_time:.6f} seconds")
print(f"Nós Visitados: {bfs_result['visited_nodes']}")
print(f"Comprimento do Caminho: {bfs_result['path_length']}")

print("\nResultado Astar - Labirinto:")
print("-" * 25)

print(f"Tempo de Execução: {astar_execution_time:.6f} seconds")
print(f"Nós Visitados: {astar_result['visited_nodes']}")
print(f"Comprimento do Caminho: {astar_result['path_length']}")

#Visualização do labirinto resolvido por A*
visualize_maze(
    astar_maze,
    astar_result["exploration_order"],
    astar_result["path"]
)

#Visualização do labirinto resolvido por A*

'''

visualize_maze(
    bfs_maze,
    bfs_result["exploration_order"],
    bfs_result["path"]
)

'''
