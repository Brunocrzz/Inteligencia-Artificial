from collections import deque

# Cores consideradas "borda"
BORDER_THRESHOLD = 40


def is_border(color):
    """
    Detecta se o pixel é preto/quase preto.
    """

    r, g, b, *_ = color

    return (
        r < BORDER_THRESHOLD
        and g < BORDER_THRESHOLD
        and b < BORDER_THRESHOLD
    )

# FloodFill utilizando DFS. Retorna os pixels pertencentes a regiao do estado.
def get_region_pixels(surface, start_position):
    width = surface.get_width()
    height = surface.get_height()

    visited = set()

    region_pixels = []

    queue = deque([start_position])

    while queue:

        x, y = queue.popleft()

        # Fora da tela
        if x < 0 or x >= width or y < 0 or y >= height:
            continue

        # Já visitado
        if (x, y) in visited:
            continue

        visited.add((x, y))

        current_color = surface.get_at((x, y))

        # Encontrou borda
        if is_border(current_color):
            continue

        region_pixels.append((x, y))

        # 4 vizinhos
        queue.append((x + 1, y))
        queue.append((x - 1, y))
        queue.append((x, y + 1))
        queue.append((x, y - 1))

    return region_pixels


#Gera os pixels de todos os estados para não ter que fazer o calculo todo momento
def generate_state_pixels(map_image, state_positions):
    state_pixels = {}

    for state, position in state_positions.items():

        pixels = get_region_pixels(
            map_image,
            position
        )

        state_pixels[state] = pixels

    return state_pixels