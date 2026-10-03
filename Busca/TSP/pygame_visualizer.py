import pygame
import math
from city import distance

WIDTH = 1200
HEIGHT = 800

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

RED = (255, 0, 0)
BLUE = (50, 150, 255)
GREEN = (0, 255, 0)

pygame.init()
pygame.font.init()

def draw_route(screen, route):
    font = pygame.font.SysFont(None, 24)
    for i in range(len(route) - 1):
        city1 = route[i]
        city2 = route[i + 1]

        pygame.draw.line(screen, BLUE, city1, city2, 2)
        
        if len(route) <= 20:  # Exibe distâncias apenas para rotas curtas
            dist = int(distance(city1, city2))
            mid_x = (city1[0] + city2[0]) // 2
            mid_y = (city1[1] + city2[1]) // 2

            text = font.render(str(dist), True, BLACK)
            screen.blit(text, (mid_x, mid_y))

    if len(route) <= 20:
        dist = int(distance(route[0], route[-1]))
        mid_x = (route[0][0] + route[-1][0]) // 2
        mid_y = (route[0][1] + route[-1][1]) // 2

        text = font.render(str(dist), True, BLACK)
        screen.blit(text, (mid_x, mid_y))

    # Fecha ciclo
    pygame.draw.line(screen, BLUE, route[-1], route[0], 2)

def draw_cities(screen, cities):
    for city in cities[:]:
        pygame.draw.circle(screen, RED, city, 6)

def drawGenetics(screen,route,total_distance,generation, current_fitness, improvement):
    font = pygame.font.SysFont(None, 36)
    screen.fill(WHITE)

    if generation >= 50:
        pygame.time.delay(1)
    elif generation >= 30:
        pygame.time.delay(10)
    else:
        pygame.time.delay(30) 

    draw_route(screen, route)
    draw_cities(screen, route)

    text = font.render(f"Distância: {total_distance:.3f}",True,BLACK)
    screen.blit(text, (20, 20))

    generation_text = font.render(f"Generation: {generation+1}",True,BLACK)
    screen.blit(generation_text, (20, 60))

    fitness_text = font.render(f"Current Fitness: {current_fitness:.6f}",True,BLACK)
    screen.blit(fitness_text, (20, 100))

    improvement_text = font.render(f"Improvement: {improvement:.2f}%",True,BLACK)
    screen.blit(improvement_text, (20, 140))

    pygame.display.update()

def drawHillClimbing(screen, route, total_distance,improvement,interation):
    font = pygame.font.SysFont(None, 36)
    screen.fill(WHITE)
    if len(route) >= 60:
        pygame.time.delay(10)
    elif len(route) >= 30:
        pygame.time.delay(30)
    else:
        pygame.time.delay(50)    
    draw_route(screen, route)

    draw_cities(screen, route)

    text = font.render(f"Distância: {total_distance:.3f}",True,BLACK)
    screen.blit(text, (20, 20))

    improvement_text = font.render(f"Improvement: {improvement:.2f}%",True,BLACK)
    screen.blit(improvement_text, (20, 60))

    interation_text = font.render(f'Iteração: {interation}',True,BLACK)
    screen.blit(interation_text,(20,100))

    pygame.display.update()

def visualize(name):
    pygame.display.set_caption(name)

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()

def initialize_screen():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("TSP - Algoritmo Genético")

    return screen
