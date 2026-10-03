import matplotlib.pyplot as plt

#Variaveis utilizadas na rede
VARIAVEIS = ["Inimigo_Visivel", "Vida_Critica", "Estado_Alerta", "Atacar", "Fugir", "Reforco"]

# Rede Bayesiana, com seus nós, pais e Tabela de Probabilidade Condicional
rede_bayesiana = {
    "Inimigo_Visivel": {
        "pais": [],
        "cpt": {(): {"Sim": 0.4, "Nao": 0.6}},  # Sem pais: tupla vazia
    },
    "Vida_Critica": {
        "pais": [], 
        "cpt": {(): {"Sim": 0.3, "Nao": 0.7}}
    },
    "Estado_Alerta": {
        "pais": ["Inimigo_Visivel", "Vida_Critica"],
        "cpt": {
            # Para cada combinacao de estados dos pais, gera um numero
            ("Sim", "Sim"): {"Alto": 0.99, "Baixo": 0.01},
            ("Sim", "Nao"): {"Alto": 0.80, "Baixo": 0.20},
            ("Nao", "Sim"): {"Alto": 0.50, "Baixo": 0.50},
            ("Nao", "Nao"): {"Alto": 0.01, "Baixo": 0.99},
        },
    },
    "Atacar": {
        "pais": ["Estado_Alerta"],
        "cpt": {
            ("Alto",): {"Sim": 0.85, "Nao": 0.15},
            ("Baixo",): {"Sim": 0.15, "Nao": 0.85},
        },
    },
    "Fugir": {
        "pais": ["Estado_Alerta"],
        "cpt": {
            ("Alto",): {"Sim": 0.65, "Nao": 0.35},
            ("Baixo",): {"Sim": 0.05, "Nao": 0.95},
        },
    },
    "Reforco": {
        "pais": ["Estado_Alerta"],
        "cpt": {
            ("Alto",): {"Sim": 0.75, "Nao": 0.25},
            ("Baixo",): {"Sim": 0.02, "Nao": 0.98},
        },
    },
}


# Busca a probabilidade de um nó assumir um determinado valor, dados os valores atuais dos seus pais contidos no dicionário de evidências.
def obter_probabilidade_condicional(no, valor_no, evidencias):
    lista_pais = rede_bayesiana[no]["pais"]
    
    # Cria a tupla com os valores atuais dos pais com base nas evidências
    valores_pais = tuple(evidencias[pai] for pai in lista_pais)
    
    # Retorna o valor da probabilidade na CPT
    return rede_bayesiana[no]["cpt"][valores_pais][valor_no]


# Percorre a árvore de probabilidades e realiza a soma sobre as variáveis ocultas.
def enumerate_all(vars, e):
    # Caso base: se a lista de variáveis estiver vazia, chegamos ao fim de um ramo
    if not vars:
        return 1.0
    
    Y = vars[0]
    resto = vars[1:]
    
    # Descobre dinamicamente os estados possíveis da variável atual Y
    uma_chave_cpt = list(rede_bayesiana[Y]["cpt"].keys())[0] # Pega a primeira tupla como exemplo
    estados_possiveis = list(rede_bayesiana[Y]["cpt"][uma_chave_cpt].keys()) 

    if Y in e:
        # Se Y é uma evidência conhecida, apenas multiplicamos sua probabilidade e avançamos
        prob = obter_probabilidade_condicional(Y, e[Y], e)
        return prob * enumerate_all(resto, e)
    else:
        # Se Y é uma variável oculta, precisamos somar (marginalizar) sobre todas as suas possibilidades
        soma = 0.0
        for yi in estados_possiveis:
            e_copia = e.copy()
            e_copia[Y] = yi
            prob = obter_probabilidade_condicional(Y, yi, e_copia)
            soma += prob * enumerate_all(resto, e_copia)
        return soma


# Retorna a distribuição de probabilidade da variável de consulta X, dadas as evidências observadas 'e'.
# P(X|e)
def enumeration_ask(X, e):
    q = {}
    
    # Descobre dinamicamente os estados possíveis da variável de consulta
    uma_chave_cpt = list(rede_bayesiana[X]["cpt"].keys())[0] # Pega a primeira tupla como exemplo
    estados_possiveis = list(rede_bayesiana[X]["cpt"][uma_chave_cpt].keys()) 
    
    # Avalia a probabilidade conjunta para cada estado possível da nossa query X
    for xi in estados_possiveis:
        e_copia = e.copy()
        e_copia[X] = xi
        q[xi] = enumerate_all(VARIAVEIS, e_copia)
        
    # Normalização
    alfa = sum(q.values())
    for xi in q:
        q[xi] = round(q[xi] / alfa, 3)  # Arredonda para 3 casas decimais
        
    return q


# Gera uma única janela dividida em uma matriz de 2 linhas e 3 colunas (6 gráficos).
def dashboard(dados_cenarios):
    # Define o tamanho da janela da imagem (largura x altura)
    fig, eixos = plt.subplots(2, 3, figsize=(18, 10))
    fig.canvas.manager.set_window_title('Análise Probabilística de Incerteza - Comportamento de Agentes (NPC)')

    # Transforma a matriz de eixos de 2D para 1D para facilitar o loop (0 a 5)
    eixos_flat = eixos.flatten()

    for i, cenario in enumerate(dados_cenarios):
        ax = eixos_flat[i]
        resultado = cenario["resultado"]
        query = cenario["query"]
        evidencias = cenario["evidencias"]
        num_cenario = cenario["num"]

        estados = list(resultado.keys())
        probabilidades = [v * 100 for v in resultado.values()]
        
        # Mapeamento de cores estéticas baseado nas classes das respostas
        if 'Sim' in estados:
            cores = ['#2ca02c', '#d62728']  # Verde para Sim, Vermelho para Não
        elif 'Alto' in estados:
            cores = ['#e377c2', '#bcbd22']  # Cores contrastantes para Estados de Alerta
        else:
            cores = ['#1f77b4', '#ff7f0e']
            
        barras = ax.bar(estados, probabilidades, color=cores, width=0.4)
        
        # Configuração textual detalhada de cada Subplot (O que o gráfico está mostrando)
        texto_evidencias = str(evidencias) if evidencias else "Nenhuma (A Priori)"
        titulo_detalhado = (
            f"Cenário {num_cenario}\n"
            f"➔ Variável Medida (Query): {query}\n"
            f"➔ Evidências:\n {texto_evidencias}"
        )
        
        ax.set_title(titulo_detalhado, fontsize=9, fontweight='bold', loc='left', pad=10, color='#2c3e50')
        ax.set_ylabel('Probabilidade Calculada (%)', fontsize=9)
        ax.set_ylim(0, 115)  # Espaço extra no topo para as labels
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Insere os valores numéricos idênticos aos do terminal acima das barras
        for barra in barras:
            yval = barra.get_height()
            ax.text(barra.get_x() + barra.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.tight_layout()
    # Ajusta o espaçamento superior para não cortar o título geral
    plt.subplots_adjust(top=0.88, hspace=0.35, wspace=0.25)
    plt.show()


# Bateria de testes e graficos
if __name__ == "__main__":
    print("=" * 100)
    print(" Compilação de 6 cenários diferentes ")
    print("=" * 100)

    # Execução e armazenamento dos cálculos probabilísticos
    # Cenário 1 - Probabilidade de estar em alerta sem evidências a priori
    ev_1 = {}
    r_1 = enumeration_ask("Estado_Alerta", ev_1)
    
    # Cenário 2 - Probabilidade de atacar com um inimigo à vista e sem vida crítica
    ev_2 = {"Inimigo_Visivel": "Sim", "Vida_Critica": "Nao"}
    r_2 = enumeration_ask("Atacar", ev_2)
    
    # Cenário 3 - Probabilidade de Fuga dado vida crítica e inimigo visível
    ev_3 = {"Inimigo_Visivel": "Sim", "Vida_Critica": "Sim"}
    r_3 = enumeration_ask("Fugir", ev_3)
    
    # Cenário 4 - Probabilidade de estar com a vida crítica com inimigo visivel e sem ter atacado
    ev_4 = {"Inimigo_Visivel": "Sim", "Atacar": "Nao"}
    r_4 = enumeration_ask("Vida_Critica", ev_4)
    
    # Cenário 5 - Probabilidade de um inimigo estar visivil dado que tenha pedido reforco e sem a vida critica
    ev_5 = {"Reforco": "Sim", "Vida_Critica": "Nao"}
    r_5 = enumeration_ask("Inimigo_Visivel", ev_5)
    
    # Cenário 6 - Probabilidade de estar em alerta com evidencias de fuga e pedir reforco
    ev_6 = {"Fugir": "Sim", "Reforco": "Sim"}
    r_6 = enumeration_ask("Estado_Alerta", ev_6)

    # Mapeamento descritivo que alimenta o interpretador visual
    dashboard_data = [
        {"num": 1, "query": "Estado_Alerta", "evidencias": ev_1, "resultado": r_1, "titulo": "Probabilidade de estar em alerta sem evidências a priori"},
        {"num": 2, "query": "Atacar", "evidencias": ev_2, "resultado": r_2,"titulo": "Probabilidade de atacar com um inimigo à vista e sem vida crítica" },
        {"num": 3, "query": "Fugir", "evidencias": ev_3, "resultado": r_3,"titulo": "Probabilidade de Fuga dado vida crítica e inimigo visível"},
        {"num": 4, "query": "Vida_Critica", "evidencias": ev_4, "resultado": r_4,"titulo": "Probabilidade de estar com a vida crítica com inimigo visivel e sem ter atacado"},
        {"num": 5, "query": "Inimigo_Visivel", "evidencias": ev_5, "resultado": r_5,"titulo": "Probabilidade de um inimigo estar visivil dado que tenha pedido reforco e sem a vida critica"},
        {"num": 6, "query": "Estado_Alerta", "evidencias": ev_6, "resultado": r_6,"titulo": "Probabilidade de estar em alerta com evidencias de fuga e pedir reforco"}
    ]

    # Exibe no terminal os valores para cópia de texto segura no relatório
    for data in dashboard_data:
        print(f"Cenário {data['num']} - {data["titulo"]} | Resultados: {data['resultado']}\n")

    dashboard(dashboard_data)
    print("=" * 100)