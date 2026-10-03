
# Conjunto de regras para inferência
rules = [
    {
        "if": ["not_liga"],
        "then": "problema_fonte",
        "confidence": 95
    },
    {
        "if": ["liga", "not_video", "not_beep"],
        "then": "problema_gpu",
        "confidence": 85
    },
    {
        "if": ["liga", "beep", "not_video"],
        "then": "problema_memoria",
        "confidence": 90
    },
    {
        "if": ["desliga_sozinho", "superaquecimento"],
        "then": "problema_temperatura",
        "confidence": 92
    },
    {
        "if": ["tela_azul"],
        "then": "problema_sistema",
        "confidence": 80
    },
    {
        "if": ["internet_lenta"],
        "then": "problema_rede",
        "confidence": 70
    },
    {
        "if": ["pc_lento", "barulho_hd"],
        "then": "problema_hd",
        "confidence": 88
    },
    {
        "if": ["superaquecimento", "not_cooler"],
        "then": "falha_cooler",
        "confidence": 93
    },
    {
        "if": ["not_hd_detectado"],
        "then": "problema_armazenamento",
        "confidence": 90
    },
    {
        "if": ["travamentos", "pc_lento"],
        "then": "problema_desempenho",
        "confidence": 75
    },
    {
        "if": ["not_usb"],
        "then": "problema_usb",
        "confidence": 70
    },
    {
        "if": [
            "liga",
            "video",
            "beep",
            "not_tela_azul",
            "not_superaquecimento",
            "not_pc_lento",
            "not_internet_lenta"
        ],
        "then": "sistema_normal",
        "confidence": 99
    },
    {
        "if": ["liga", "reinicia"],
        "then": "instabilidade_energia",
        "confidence": 78
    },
    {
        "if": ["not_hd_detectado"],
        "then": "problema_armazenamento",
        "confidence": 90
    }
]

# Perguntas a serem utilizadas para utilização de regras
questions = [
    ("O computador liga?", "liga"),
    ("Existe vídeo na tela?", "video"),
    ("O computador faz beep?", "beep"),
    ("O computador desliga sozinho?", "desliga_sozinho"),
    ("Existe tela azul?", "tela_azul"),
    ("A internet está lenta?", "internet_lenta"),
    ("O computador está lento?", "pc_lento"),
    ("O computador aquece muito?", "superaquecimento"),
    ("Existe barulho estranho no HD?", "barulho_hd"),
    ("O HD/SSD é detectado pela BIOS?", "hd_detectado"),
    ("O sistema trava frequentemente?", "travamentos"),
    ("Os coolers estão funcionando?", "cooler"),
    ("As portas USB funcionam?", "usb"),
    ("O computador reinicia sozinho?", "reinicia"),
]

# Diagnose final. Associa o problema retornado na frase a ser mostrada
diagnosis_text = {
    "problema_memoria":"Possível problema na memória RAM.",
    "problema_gpu":"Possível problema na placa de vídeo.",
    "problema_fonte":"Possível problema na fonte de alimentação.",
    "problema_temperatura":"Possível problema de temperatura/refrigeração.",
    "falha_cooler":"Possível falha no cooler do sistema.",
    "problema_sistema":"Possível falha no sistema operacional.",
    "problema_rede":"Possível problema de conexão de rede.",
    "problema_hd":"Possível falha no HD.",
    "problema_armazenamento":"Dispositivo de armazenamento não detectado.",
    "problema_desempenho":"Possível problema geral de desempenho.",
    "problema_usb":"Possível falha nas portas USB.",
    "instabilidade_energia":"Possível instabilidade de energia.",
    "problema_driver":"Possível problema de drivers.",
    "memoria_insuficiente":"Possível insuficiência de memória.",
    "problema_roteador":"Possível problema no roteador/conexão.",
    "problema_bios":"Possível problema relacionado à BIOS.",
    "problema_placa_mae":"Possível falha na placa-mãe.",
    "sistema_funcional":"Sistema aparentemente funcional.",
    "sistema_normal":"Nenhum problema relevante detectado no sistema.",
    "diagnostico_inconclusivo":"Nenhum problema identificado ou diagnóstico inconclusivo."
}

# Associa cada fato a um texto a ser mostrado na tela
fact_labels = {
    "liga": "Computador Liga",
    "not_liga": "Computador Não Liga",
    "video": "Existe Vídeo",
    "not_video": "Sem Vídeo",
    "beep": "Possui Beep",
    "not_beep": "Sem Beep",
    "tela_azul": "Tela Azul",
    "not_tela_azul": "Sem tela Azul",
    "pc_lento": "Computador Lento",
    "not_pc_lento": "Computador não está lento",
    "superaquecimento": "Superaquecimento",
    "not_superaquecimento": "Sem Superaquecimento",
    "internet_lenta": "Internet Lenta",
    "not_internet_lenta": "Internet não está lenta",
    "barulho_hd": "Barulho no HD",
    "not_barulho_hd": "Sem barulho no HD",
    "desliga_sozinho":"Desliga Sozinho",
    "not_desliga_sozinho":"Não Desliga Sozinho",
    "reinicia": "Reinicia Sozinho",
    "not_reinicia": "Não Reinicia Sozinho",
     "travamentos": "Sistema Travando",
    "not_travamentos": "Sem Travamentos",
    "hd_detectado": "HD/SSD Detectado",
    "not_hd_detectado": "HD/SSD Não Detectado",
    "cooler": "Coolers Funcionando",
    "not_cooler": "Coolers Não Funcionam",
    "usb": "USB Funcionando",
    "not_usb": "USB Não Funciona",
}