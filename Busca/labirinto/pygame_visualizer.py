import pygame

MAX_WIDTH = 1200
MAX_HEIGHT = 800

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

GREEN = (0, 255, 0)
RED = (255, 0, 0)

BLUE = (50, 150, 255)
YELLOW = (255, 255, 0)

pygame.init()

def draw_maze(screen, maze):

    rows = len(maze)
    cols = len(maze[0])

    # Define o tamanho de cada célula de acordo com o tamanho
    cell_size = min(MAX_WIDTH // cols, MAX_HEIGHT // rows)

    # Para cada símbolo do labirinto define a cor do pixel
    for row in range(rows):
        for col in range(cols):

            if maze[row][col] == "#":
                color = BLACK
            else:
                color = WHITE

            if maze[row][col] == "S":
                color = GREEN

            elif maze[row][col] == "E":
                color = RED

            elif maze[row][col] == "v":
                color = BLUE

            elif maze[row][col] == ".":
                color = YELLOW

            pygame.draw.rect(
                screen, color, (col * cell_size, row * cell_size, cell_size, cell_size)
            )

def visualize_maze(maze, exploration_order, path):

    rows = len(maze)
    cols = len(maze[0])

    cell_size = min(MAX_WIDTH // cols, MAX_HEIGHT // rows)

    width = cols * cell_size
    height = rows * cell_size

    # Cria a janela do Pygame proporcional ao tamanho do labirinto.
    screen = pygame.display.set_mode((width, height))

    pygame.display.set_caption("Maze Visualization")

    visual_maze = [row[:] for row in maze]

    # ? EXPLORATION ANIMATION
    # Mostra visualmente os nós sendo explorados pelo algoritmo.
    for row, col in exploration_order:
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()
                return

        if visual_maze[row][col] not in ["S", "E"]:
            visual_maze[row][col] = "v"

        draw_maze(screen, visual_maze)
        pygame.display.flip()

        # Ajusta automaticamente o delay conforme o tamanho do labirinto.
        area = rows * cols
        if area < 900:
            pygame.time.delay(5)

    # ^ FINAL PATH ANIMATION
    # Após encontrar a solução, desenha o caminho final.
    for row, col in path:
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()
                return

        if visual_maze[row][col] not in ["S", "E"]:
            visual_maze[row][col] = "."
        draw_maze(screen, visual_maze)

        pygame.display.flip()

        # Delay menor para labirintos grandes.
        area = rows * cols
        if area > 1600:
            pygame.time.delay(3)
        else:
            pygame.time.delay(5)

    # Mantém a janela aberta até o usuário fechá-la.
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    pygame.quit()