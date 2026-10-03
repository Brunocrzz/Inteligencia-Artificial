import math
import random

# Gera num_cities cidades em posições aleatórias na tela
def generate_cities(num_cities):
    width = 1200
    height = 800
    cities = []
    for _ in range(num_cities):
        x = random.randint(50, width-50)
        y = random.randint(50, height-50)
        cities.append((x, y))
    return cities

# Calcula a distancia Euclidiana entre duas cidades
def distance(city1, city2):
    x1, y1 = city1
    x2, y2 = city2

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Calcula a distancia total da rota, somando as distancias entre todas as cidades vizinhas
def route_distance(route): 
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance(route[i], route[i + 1])
    # Volta para cidade inicial
    total_distance += distance(route[-1], route[0])

    return total_distance

# Aleatoriamente troca duas cidades na ordem, alterando a rota 
def generate_neighbor(route):

    neighbor = route[:]

    i = random.randint(0, len(route) - 1)
    j = random.randint(0, len(route) - 1)

    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

    return neighbor