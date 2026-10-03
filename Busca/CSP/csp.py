import pygame
from brasilGraph import COLORS
import random

# Analisa se a cor é válida, ou seja, se é diferente da cor dos vizinhos já atribuídos
def is_consistent(state, color, assignment, graph):
    for neighbor in graph[state]:

        if neighbor in assignment:
            if assignment[neighbor] == color:
                return False

    return True


# Retorna as cores válidas para um estado, ou seja, as cores que não violam as restrições de cor dos vizinhos
def get_valid_colors(state, assignment, graph, colors):
    valid_colors = []

    for color in colors:

        if is_consistent(state, color, assignment, graph):
            valid_colors.append(color)

    return valid_colors


# MRV - Seleciona a variável não atribuída com o menor número de valores possíveis (cores válidas) para atribuir a seguir
def select_unassigned_variable(assignment, graph, colors):
    unassigned = [state for state in graph if state not in assignment]

    best_state = None
    # Inicializa o número mínimo como infinito para garantir que qualquer estado com um número de cores válidas menor será selecionado
    minimum_remaining = float("inf")

    # Armazena o grau mais alto encontrado para desempate, inicializado como -1 para garantir que qualquer estado com um grau maior será selecionado
    highest_degree = -1

    # para cada estado não atribuído, calcula o número de cores válidas e seleciona o estado com o menor número de cores válidas
    for state in unassigned:
        valid_colors = get_valid_colors(state, assignment, graph, colors)
        remaining = len(valid_colors)
        degree = len(graph[state])  # Grau do estado é o número de vizinhos

        # MRV
        if remaining < minimum_remaining:
            minimum_remaining = remaining
            highest_degree = degree
            best_state = state

        # Grau da Variavel - Desempate pelo grau mais alto
        elif remaining == minimum_remaining:
            if degree > highest_degree:
                highest_degree = degree
                best_state = state

    return best_state


def get_draw_frequency(recursive_calls):
    values = [
        (100, 10),
        (500, 20),
        (2000, 50),
        (10000, 100),
        (30000, 500),
        (50000, 1000),
        (100000, 10000),
    ]

    for limit, frequency in values:
        if recursive_calls < limit:
            return frequency

    return 5000


def backtracking(app, assignment, graph, stats, draw_callback, use_heuristics):
    stats["recursive_calls"] += 1
    colors = COLORS

    # Caso base: se todas as variáveis estão atribuídas, retorna a atribuição
    if len(assignment) == len(graph):
        return assignment

    # Lista de estados não coloridos
    unassigned = [state for state in graph if state not in assignment]
    if not use_heuristics:
        random.shuffle(unassigned) #Garante que a ordem de seleção dos estados seja diferente a cada execução

    if use_heuristics:
        # Seleciona o próximo estado a ser colorido usando o heurístico MRV
        current_state = select_unassigned_variable(assignment, graph, colors)
    else:
        # Seleciona o próximo estado a ser colorido aleatoriamente
        current_state = unassigned[0]

    # Gera uma lista de cores disponíveis para o estado atual, embaralhando a ordem para garantir que a solução encontrada seja diferente a cada execução
    available_colors = colors[:]
    random.shuffle(available_colors)

    # Para cada cor disponível, verifica se é consistente atribuir a cor ao estado atual
    for color in available_colors:

        # Se for consistente, atribui a cor ao estado atual e chama recursivamente o backtracking para tentar atribuir as próximas variáveis
        if is_consistent(current_state, color, assignment, graph):
            assignment[current_state] = color

            # Diminui a taxa de desenho caso não usar heutisticas para visualização mais fluida
            if not use_heuristics:
                draw_frequency = get_draw_frequency(stats["recursive_calls"])
                if stats["recursive_calls"] % draw_frequency == 0:
                    draw_callback(app, stats,assignment)
                    pygame.time.delay(10)
            else:
                draw_callback(app, stats,assignment)
                pygame.time.delay(30)

            result = backtracking(app, assignment, graph, stats, draw_callback, use_heuristics)
            if result:
                return result

            # Se a atribuição não levar a uma solução, remove a atribuição e tenta a próxima cor
            stats["backtracks"] += 1
            del assignment[current_state]

    return None
