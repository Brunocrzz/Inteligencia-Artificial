import pygame
from knowledge_base import (rules,questions,diagnosis_text,fact_labels)
from inference import forward_chaining
from pygame_visualizer import draw

facts = []
current_question = 0

running = True
while running:

    # VERIFICA SE AINDA EXISTEM PERGUNTAS
    if current_question < len(questions):
        question, fact_name = questions[current_question]
    else:
        question = "Diagnostico finalizado."

    # INFERÊNCIA
    if current_question >= len(questions):
        diagnosis_keys, activated_rules = forward_chaining(
            facts,
            rules
        )
    else:
        diagnosis_keys = []
        activated_rules = []

    diagnosis = []

    for rule in activated_rules:
        diagnosis.append({
            "name": diagnosis_text[rule["then"]],
            "confidence": rule["confidence"]
        })

    # DESENHA INTERFACE
    yes_button, no_button, reset_button, content_height = draw(
        question,
        facts,
        diagnosis,
        fact_labels,
    )

    # EVENTOS
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        # SCROLL
        if event.type == pygame.MOUSEWHEEL:
            visible_area = 700
            max_scroll = 0
            min_scroll = min(0,visible_area - content_height - 50)
            scroll_offset += event.y * 40

            # Impede subir demais
            if scroll_offset > 0:
                scroll_offset = 0

            # Impede descer demais
            if scroll_offset < min_scroll:
                scroll_offset = min_scroll

        # CLIQUES
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()

            # BOTÃO SIM
            if yes_button.collidepoint(mouse_pos):
                if current_question < len(questions):
                    if fact_name not in facts:
                        facts.append(fact_name)
                    current_question += 1

            # BOTÃO NÃO
            elif no_button.collidepoint(mouse_pos):
                if current_question < len(questions):
                    negative_fact = f"not_{fact_name}"
                    if negative_fact not in facts:
                        facts.append(negative_fact)
                    current_question += 1

            # RESET
            elif reset_button.collidepoint(mouse_pos):
                facts.clear()
                diagnosis.clear()
                activated_rules.clear()
                current_question = 0
                scroll_offset = 0

        

pygame.quit()