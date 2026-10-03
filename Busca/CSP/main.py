import pygame
from pygame_visualizer import initialize, draw
from brasilGraph import BRAZIL_STATES
from csp import backtracking

# Define os status iniciais de comparação como 0
stats = {"recursive_calls": 0, "backtracks": 0}
assignment = {}

app = initialize()

running = True
while running:
    # Inicializa ambos botões que levam o algoritmo ja desenhando a tela
    heuristic_button, normal_button = draw(app, stats, assignment)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Clique mouse
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()

            # Carrega backtracking com Heurísticas
            if heuristic_button.collidepoint(mouse_pos):
                stats = {"recursive_calls": 0, "backtracks": 0}
                solution = backtracking(
                    app,
                    {},
                    BRAZIL_STATES,
                    stats,
                    draw,
                    use_heuristics=True
                )
                assignment = solution

            # Carrega backtracking Sem heurísticas
            if normal_button.collidepoint(mouse_pos):
                stats = {"recursive_calls": 0, "backtracks": 0}
                solution = backtracking(
                    app,
                    {},
                    BRAZIL_STATES,
                    stats,
                    draw,
                    use_heuristics=False
                )
                assignment = solution

pygame.quit()
