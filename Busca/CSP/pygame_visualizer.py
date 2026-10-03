import os
import pygame
from brasilGraph import STATE_POSITIONS, COLORS, BRAZIL_STATES
from floodfill import generate_state_pixels

WIDTH = 1200
HEIGHT = 800

# Caminho do mapa relativo a este arquivo, funciona de qualquer pasta e em qualquer sistema
MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mapa.png")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

COLOR_MAP = {
    "red": (220, 50, 50),
    "green": (50, 180, 50),
    "blue": (50, 50, 220),
    "yellow": (230, 230, 50),
}

def initialize():
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    pygame.display.set_caption("Brazil CSP Map Coloring")

    font = pygame.font.SysFont(None, 28)

    map_image = pygame.image.load(MAP_PATH)
    map_image = pygame.transform.scale(map_image, (WIDTH, HEIGHT))

    state_pixels = generate_state_pixels(
        map_image,
        STATE_POSITIONS
    )
    
    return {
        "screen": screen,
        "font": font,
        "map_image": map_image,
        "state_pixels": state_pixels
    }

def draw(app,stats,assignment):
    screen = app["screen"]
    font = app["font"]
    map_image = app["map_image"]
    state_pixels = app["state_pixels"]

    screen.fill((255,255,255))

    colored_map = map_image.copy()

    # Estados
    for state, pixels in state_pixels.items():
        color = (180,180,180)
        if state in assignment:
            color = COLOR_MAP[assignment[state]]
        for pixel in pixels:
            colored_map.set_at(pixel, color)

    screen.blit(colored_map, (0,0))

    # Métricas
    backtrack_text = font.render(f"Backtracks: {stats['backtracks']}", True, BLACK)
    recursive_text = font.render(f"Recursive Calls: {stats['recursive_calls']}", True, BLACK)
    screen.blit(backtrack_text, (20, HEIGHT - 20))
    screen.blit(recursive_text, (20, HEIGHT - 60))

    # Botões
    heuristic_button = pygame.Rect(20, HEIGHT - 130, 250, 50)
    normal_button = pygame.Rect(20, HEIGHT - 200, 250, 50)
    pygame.draw.rect(screen, (180, 180, 180), heuristic_button)
    pygame.draw.rect(screen, (180, 180, 180), normal_button)

    # Textos nos botões
    heuristic_text = font.render("Rodar com Heurísticas", True, BLACK)
    normal_text = font.render("Rodar sem Heurísticas", True, BLACK)
    screen.blit(heuristic_text, (40, HEIGHT - 115))
    screen.blit(normal_text, (40, HEIGHT - 185))

    pygame.display.flip()

    return heuristic_button, normal_button
