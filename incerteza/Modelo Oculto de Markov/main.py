import matplotlib.pyplot as plt
import numpy as np

# Estados Ocultos (S) e Observações (V)
estados = [0,1,2]
estadosOcultos = {0: "Agressivo", 1: "Defensivo", 2: "Economico"}
observacoes = {0: "Comprou_Arma", 1: "Escondeu", 2: "Avancou", 3: "Curou"}

# Probabilidades Iniciais (Pi)
inicio = {"Agressivo": 0.3, "Defensivo": 0.3, "Economico": 0.4}

# Matriz de Transição
# Linha i = estado atual (t), Coluna j = próximo estado (t+1)
T = np.array([
    [0.7, 0.2, 0.1],  # Transições a partir do estado 0
    [0.15, 0.65, 0.2],  # Transições a partir do estado 1
    [0.2, 0.2, 0.6],  # Transições a partir do estado 2
])

# Matriz de Emissão (B) - Chance de gerar a ação dado o estado
# Linha= estado oculto, Coluna = observacao
B = np.array([
    [0.4, 0.05, 0.5, 0.05],  # Estado 0 (Agressivo)
    [0.1, 0.4, 0.1, 0.4],  # Estado 1 (Defensivo)
    [0.1, 0.3, 0.2, 0.4],  # Estado 2 (Economico)
])

# Vetor de Inicialização (Tempo t=0) - chute inicial
pi = np.array([0.3, 0.3, 0.4])


# Constrói dinamicamente a matriz de observação O SxS.
# A i-ésima entrada diagonal é P(et | Xt = i) e as outras entradas são 0.
def matrizObsDiagonal(observacao):
    # Coleta a coluna da ação observada para todos os estados
    valores_sensor = B[:, observacao]
    
    # np.diag cria uma matriz quadrada com esses valores na diagonal e zero no resto
    return np.diag(valores_sensor)

# Executa o algoritmo Forward de forma matricial
# Utiliza operadores nativos do NumPy para álgebra linear.
def forward_matricial(sequencia_observacoes):
    # f_atual começa como o vetor coluna pi
    f_atual = pi.copy()
    
    # Armazena o histórico e converte para lista para facilitar a plotagem
    historico_f = [f_atual.flatten().tolist()]
    
    # Transposta da matriz de transição (T^T) usando a propriedade .T do NumPy
    T_transposta = T.T
    
    # Avança no tempo passo a passo (t -> t+1)
    for acao in sequencia_observacoes:
        # Calcula f_{t+1} = O_{t+1} * T^T * f_t
        # O operador '@' no NumPy realiza a multiplicação de matrizes
        matriz_O = matrizObsDiagonal(acao)
        f_novo = matriz_O @ T_transposta @ f_atual
        
        # Normalização
        alfa = np.sum(f_novo)
        if alfa > 0:
            f_atual = f_novo / alfa
        else:
            f_atual = f_novo
            
        # Arredonda e salva o estado atualizado no histórico
        f_atual_arredondado = np.round(f_atual, 3)
        historico_f.append(f_atual_arredondado.flatten().tolist())
        
    return historico_f


# Gera um dashboard com 3 gráficos de acordo com os cenarios apresentados
def dashboard(dados_cenarios):
    fig, eixos = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.canvas.manager.set_window_title('Modelo Oculto de Markov - HMM')

    for i, cenario in enumerate(dados_cenarios):
        ax = eixos[i]
        historico = np.array(cenario["historico"])  # Transforma em array para fatiar colunas
        sequencia_nomes = [observacoes[act] for act in cenario["sequencia"]]

        # Cria o eixo X representando a linha do tempo (Começa em t=0)
        passos_tempo = list(range(len(historico)))
        rotulos_x = ["t=0\n(Início)"] + [f"t={t}\n({nome})" for t, nome in enumerate(sequencia_nomes, start=1)]
        
        # Plota a evolução da linha para cada um dos 3 estados ocultos
        ax.plot(passos_tempo, historico[:, 0] * 100, label="Agressivo", color='#d62728', marker='o', linewidth=2)
        ax.plot(passos_tempo, historico[:, 1] * 100, label="Defensivo", color='#2ca02c', marker='s', linewidth=2)
        ax.plot(passos_tempo, historico[:, 2] * 100, label="Econômico", color='#1f77b4', marker='^', linewidth=2)
        
        # Customizações detalhadas do gráfico (Especificando o que ele mostra)
        ax.set_title(f"Cenário {cenario['num']}: {cenario['desc']}", fontsize=9, fontweight='bold', color='#2c3e50', pad=12)
        ax.set_xlabel("Evolução Temporal / Ações Observadas", fontsize=9)
        ax.set_ylabel("Certeza do Perfil pela IA (%)", fontsize=9)
        ax.set_xticks(passos_tempo)
        ax.set_xticklabels(rotulos_x, fontsize=8)
        ax.set_ylim(-5, 105)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, wspace=0.25)
    plt.show()


if __name__ == "__main__":
    print("=" * 70)

    # Execução das simulações sequenciais
    seq_1 = [0,2,2]  # Comprou_Arma -> Avançou -> Avançou
    hist_1 = forward_matricial(seq_1)
    
    seq_2 = [1, 3, 1, 3]  # Escondeu -> Curou -> Escondeu -> Curou
    hist_2 = forward_matricial(seq_2)
    
    seq_3 = [1, 1, 0, 2, 2]  # Escondeu -> Escondeu -> Comprou_Arma -> Avançou -> Avançou
    hist_3 = forward_matricial(seq_3)

    # Estruturação do painel de dados
    dados_dashboard = [
        {"num": 1, "sequencia": seq_1, "historico": hist_1, "desc": "Predição Hostil\n(Foco em Avanço e Ataque)"},
        {"num": 2, "sequencia": seq_2, "historico": hist_2, "desc": "Predição Conservadora\n(Foco em Cobertura e Cura)"},
        {"num": 3, "sequencia": seq_3, "historico": hist_3, "desc": "Transição Dinâmica de Perfil\n(Mudança de Estratégia no Meio da Partida)"}
    ]

    # Exibição dos dados no terminal
    for cenario in dados_dashboard:
        print(f"\nValores Cenário {cenario['num']} - Histórico de Vetores de Estado:")
        for t, vetor in enumerate(cenario["historico"]):
            print(f" Instante t={t}: Agressivo={vetor[0]:.3f} | Defensivo={vetor[1]:.3f} | Econômico={vetor[2]:.3f}")

    print("\n"+"=" * 70)
    dashboard(dados_dashboard)