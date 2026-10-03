from city import generate_cities
from geneticAlg import genetic_algorithm
from hillClimbing import hill_climbing
import random

# Variaveis utilizadas nos algoritmos
CITIES = 20
POPULATION_SIZE = 100
ELITE_SIZE = 5
MUTATION_RATE = 0.05
GENERATIONS = 30
ITERACOES = 1000

# Gera as cidades aleatoriamentes, cria uma rota e embaralha
cities = generate_cities(CITIES)
route = cities[:]
random.shuffle(route)

# Algoritmo genético

'''
best_route, best_distance = genetic_algorithm(cities,POPULATION_SIZE,ELITE_SIZE,MUTATION_RATE,GENERATIONS)
'''

# Algoritmo HillClimbing

best_route, best_distance = hill_climbing(route, ITERACOES)

