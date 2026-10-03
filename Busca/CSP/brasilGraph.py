# Grafo com todos os estados e seus vizinhos
BRAZIL_STATES = {
    "AC": ["AM", "RO"],
    "AL": ["SE", "PE", "BA"],
    "AP": ["PA"],
    "AM": ["AC", "RO", "RR", "PA","MT"],
    "BA": ["AL", "SE", "PE", "PI", "GO", "MG","ES","TO"],
    "CE": ["PI", "PE","PB", "RN"],
    "DF": ["GO", "MG"],
    "ES": ["RJ", "MG", "BA"],
    "GO": ["DF", "MG", "BA", "MS", "MT","TO"],
    "MA": ["PI", "TO", "PA"],
    "MG": ["BA", "GO", "MS", "SP", "RJ", "ES","DF"],
    "MS": ["MT", "GO", "MG", "SP", "PR"],
    "MT": ["RO", "AM", "PA", "TO", "GO", "MS"],
    "PA": ["AP", "AM", "RR", "MT", "TO", "MA"],
    "PB": ["RN", "CE", "PE"],
    "PE": ["PB", "CE", "PI", "BA", "AL"],
    "PI": ["MA", "TO", "BA", "PE", "CE"],
    "PR": ["MS", "SP", "SC"],
    "RJ": ["ES", "MG", "SP"],
    "RN": ["CE", "PB"],
    "RO": ["AC", "AM", "MT"],
    "RR": ["AM", "PA"],
    "RS": ["SC"],
    "SC": ["PR", "RS"],
    "SE": ["BA", "AL"],
    "SP": ["MS", "MG", "RJ", "PR"],
    "TO": ["PA", "MA", "PI", "BA", "GO", "MT"]
}

# 4 Cores disponivel por conta do teorema das 4 cores
COLORS = [
    "red",
    "green",
    "blue",
    "yellow"
]

#Posicao de cada estado, baseado na imagem e tela de 800px x 1200px
STATE_POSITIONS = {
    "AC": (156, 320),
    "AM": (400, 200),
    "RR": (390, 54),
    "PA": (620, 170),
    "AP": (680, 60),
    "RO": (375, 360),
    "MT": (550, 330),
    "MS": (630, 525),
    "GO": (745, 415),
    "DF": (790, 433),
    "TO": (785, 280),
    "MA": (845, 232),
    "PI": (880, 297),
    "CE": (1039, 244),
    "RN": (1110, 238),
    "PB": (1093, 267),
    "PE": (1047, 288),
    "AL": (1125, 321),
    "SE": (1096, 342),
    "BA": (1010, 396),
    "MG": (903, 454),
    "ES": (989, 519),
    "RJ": (925, 571),
    "SP": (777, 580),
    "PR": (636, 609),
    "SC": (664, 656),
    "RS": (647, 733)
}