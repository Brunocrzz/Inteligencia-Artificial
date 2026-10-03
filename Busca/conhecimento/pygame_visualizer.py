import pygame

WIDTH = 1200
HEIGHT = 800

WHITE = (255,255,255)
BLACK = (0,0,0)

GREEN = (50,200,50)
RED = (200,50,50)
GRAY = (210,210,210)

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Sistema Especialista")

font = pygame.font.SysFont(None, 32)
small_font = pygame.font.SysFont(None, 26)

def draw(question,facts,diagnosis,fact_labels):
    screen.fill(WHITE)

    # TÍTULO
    title = font.render("Sistema Especialista - Diagnostico de Hardware",True,BLACK)
    screen.blit(title, (30, 20))

    # PERGUNTA
    question_text = font.render(question, True, BLACK)
    screen.blit(question_text, (30, 100))

    # BOTÕES
    yes_button = pygame.Rect(30, 170, 200, 60)
    no_button = pygame.Rect(260, 170, 200, 60)
    reset_button = pygame.Rect(950, 20, 200, 50)

    pygame.draw.rect(screen, GREEN, yes_button)
    pygame.draw.rect(screen, RED, no_button)
    pygame.draw.rect(screen, GRAY, reset_button)

    yes_text = font.render("SIM", True, BLACK)
    no_text = font.render("NAO", True, BLACK)
    reset_text = font.render("RESET", True, BLACK)

    screen.blit(yes_text, (100, 185))
    screen.blit(no_text, (330, 185))
    screen.blit(reset_text, (1000, 32))

    # SCROLL AREA
    facts_y = 250 
    diagnosis_y = 250

    # FATOS
    facts_title = font.render("Fatos Atuais:",True,BLACK)
    screen.blit(facts_title, (30, facts_y))

    facts_y += 45
    for fact in facts:
        color = (20,120,20)
        if fact.startswith("not_"):
            color = (180,50,50)

        fact_text = small_font.render(f"- {fact_labels.get(fact, fact)}",True,color)
        screen.blit(fact_text, (50, facts_y))

        facts_y += 30

    # DIAGNÓSTICOS
    diagnosis_title = font.render("Diagnósticos:",True,BLACK)
    screen.blit(diagnosis_title, (500, diagnosis_y))

    diagnosis_y += 45

    # Define a maior taxa de confiança como zero inicialmente
    highest_confidence = 0
    # Ordena os diagnósticos pela maior tx de confianca
    diagnosis.sort(key=lambda x: x["confidence"],reverse=True)

    for item in diagnosis:
        if item["confidence"] > highest_confidence:
            highest_confidence = item["confidence"]

    for item in diagnosis:
        diagnosis_name = item["name"]
        confidence = item["confidence"]

        color = BLACK
        if confidence == highest_confidence:
            color = (20,140,20)

        diag_text = small_font.render(f"- {diagnosis_name} ({confidence}%)",True,color)
        screen.blit(diag_text, (520, diagnosis_y))

        diagnosis_y += 35

    # ALTURA DO CONTEÚDO
    content_height = max(facts_y, diagnosis_y)

    pygame.display.flip()
    return yes_button, no_button, reset_button, content_height