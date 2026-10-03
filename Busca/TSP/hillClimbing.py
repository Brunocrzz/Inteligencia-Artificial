from pygame_visualizer import drawHillClimbing, initialize_screen, visualize
from city import route_distance, generate_neighbor
import pygame

#Algoritmo do Hill Climbing integrado com o Pygame para visualização
def hill_climbing(initial_route, max_iterations):
    screen = initialize_screen()

    # Inicializa com a rota inicial (embaralhada), guarda a distancia inicial e inicia a taxa de melhoria
    current_route = initial_route[:]
    current_distance = route_distance(current_route)
    initial_distance = current_distance
    improvement = 0.0

    for i in range(max_iterations):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Gera um vizinho (SWAP de cidade)
        neighbor = generate_neighbor(current_route)
        neighbor_distance = route_distance(neighbor)

        # Compara se a nova rota é mais curta que a anterior e atualiza os parametros
        if neighbor_distance < current_distance:
            current_route = neighbor
            current_distance = neighbor_distance
            improvement = ((initial_distance - current_distance) / initial_distance) * 100

            # Desenha na tela sempre que há atualização na rota
            drawHillClimbing(
                screen,
                current_route,
                current_distance,
                improvement,
                i
            )

    # Retorna dados de comparação
    print(f"Melhor distância antes do Hill Climbing: {route_distance(initial_route):.3f}")
    print(f"Melhor distância após o Hill Climbing: {current_distance:.3f}")
    print(f'Melhora: {((route_distance(initial_route) - current_distance) / route_distance(initial_route)) * 100:.2f}%')
    visualize("TSP - Hill Climbing")
    return current_route, current_distance