import random
from city import route_distance
from pygame_visualizer import drawGenetics, initialize_screen, visualize
import pygame

# Cria a populacao de tamanho population_size. Cada população é uma rota em ordem aleatória
def create_population(cities, population_size):
    population = []

    for _ in range(population_size):
        route = cities[:]
        random.shuffle(route)
        population.append(route)

    return population

# Calcula a funcao fitness
def fitness(route):
    return 1 / route_distance(route) # A menor distância tem maior fitness

#Seleciona as elite_size melhores cidades  
def select_best(population, elite_size):
    sorted_population = sorted(population, key=fitness, reverse=True) # Ordena a população com base no fitness (distância) em ordem decrescente 
    return sorted_population[:elite_size] # Retorna as melhores rotas (elite) com base na distância

#Funcao de embaralhamento genetico, dados dois pais
def crossover(parent1, parent2):
    size = len(parent1)

    # Gerar dois pontos de corte aleatórios para o crossover
    start = random.randint(0, size - 2)
    end = random.randint(start + 1, size - 1) 
    child = [None] * size

    # Copia trecho do pai1
    child[start:end] = parent1[start:end]

    # Completa com pai2
    pointer = 0

    # Itera sobre as cidades do pai2 e adiciona ao filho se não estiver presente
    for city in parent2:

        # Verifica se a cidade do pai2 já está presente no filho
        if city not in child:

            # Move o ponteiro para a próxima posição disponível no filho
            while child[pointer] is not None:
                pointer += 1

            # Adiciona a cidade do pai2 ao filho na posição do ponteiro
            child[pointer] = city

    return child

#Funcao de mutacao. Tem uma chance de mutation_rate 0-100 de mutacao (SWAP)
def mutate(route, mutation_rate):
    mutated = route[:]

    # Aleatoriamente faz um SWAP entre duas cidades com base na taxa de mutação
    if random.random() < mutation_rate:
        i = random.randint(0, len(route) - 1)
        j = random.randint(0, len(route) - 1)

        mutated[i], mutated[j] = mutated[j], mutated[i]

    return mutated

# Define a proxima geracao de acordo com a populacao atual, elite_size e taxa de mutacao
def create_next_generation(population, elite_size, mutation_rate):
    best = select_best(population, elite_size) # Seleciona os melhores da população atual (elite)

    # Gera filhos a partir dos melhores pais
    new_population = best[:]
    while len(new_population) < len(population):
        # Seleciona dois pais aleatoriamente da elite para o crossover
        parent1 = random.choice(best)
        parent2 = random.choice(best)

        # Realiza o crossover entre os dois pais para criar um filho e depois aplica mutação no filho
        child = crossover(parent1, parent2)
        child = mutate(child, mutation_rate)

        new_population.append(child)

    return new_population

#Algoritmo genetico principal. Integrado com o pygame para visualização do algoritmo. Ele cria populações, geracoes, mutacoes e etc.
def genetic_algorithm(cities,population_size,elite_size,mutation_rate,generations):
    population = create_population(cities,population_size) # Cria a população inicial de rotas aleatórias

    initial_route = max(population, key=fitness)
    initial_distance = route_distance(initial_route)

    screen = initialize_screen() # Inicializa a tela do Pygame para visualização

    best_route = None
    best_distance = float("inf")

    for generation in range(generations):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Gera a próxima geração de rotas a partir da população atual usando seleção, crossover e mutação
        population = create_next_generation(population,elite_size,mutation_rate) 

        current_best = max(population,key=fitness) # Seleciona a melhor rota da população atual com base no fitness (distância)

        current_distance = route_distance(current_best)
        current_fitness = fitness(current_best)

        # Atualiza a melhor rota e distância global se a melhor rota da população atual for melhor do que a melhor rota global
        if current_distance < best_distance:
            best_distance = current_distance
            best_route = current_best
            improvement = ((initial_distance - best_distance) / initial_distance) * 100 

        drawGenetics(
            screen,
            best_route,
            best_distance,
            generation,
            current_fitness,
            improvement
        )

        if generation == 0:
            print("\n" + "="* 25 + f"\nGeração {generation + 1} | "f"Distância: {best_distance:.3f}")
        elif generation == generations - 1:  
            print(f"Geração {generation + 1} | "f"Distância: {best_distance:.3f}")

    visualize("Algoritimo Genético - TSP")
    print(f"\nDistância da melhor rota: {best_distance:.3f}")
    print(f"Melhoria em relação à rota inicial: {improvement:.2f}%")
    return best_route, best_distance