import numpy as np
import matplotlib.pyplot as plt

# Constantes 
dt = 0.1             # Intervalo de amostragem temporal - Delta T
passos = 200         # Duração do experimento - 100 iterações
tempo = np.linspace(0, passos * dt, passos)

# Variâncias do Ruído (Incertezas)
ruido_sensor_gps = 8.0  # Variância do erro do GPS de pista (Ruído de Medição - R)
ruido_processo_fisico = 0.2  # Incerteza na dinâmica do carro (Ruído de Processo - Q)

# Matriz de Transição de Estado A 6x6
# Mapeia: s = s0 + v0*dt + 0.5*a*dt^2  e  v = v0 + a*dt
A = np.array([
    [1.0,  dt, 0.5*dt**2,  0.0, 0.0,     0.0], # Equações para o Eixo X
    [0.0, 1.0,        dt,  0.0, 0.0,     0.0], 
    [0.0, 0.0,       1.0,  0.0, 0.0,     0.0], 
    [0.0, 0.0,       0.0,  1.0,  dt, 0.5*dt**2], # Equações para o Eixo Y
    [0.0, 0.0,       0.0,  0.0, 1.0,        dt], 
    [0.0, 0.0,       0.0,  0.0, 0.0,       1.0]  
])

# Matriz de Observação (H) 2x6
# Indica que o sensor (GPS) lê apenas as posições X (índice 0) e Y (índice 3)
H = np.array([
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
])

# Matriz de Covariância do Ruído de Medição (R) 2x2
# Define o ruido do GPS
R = np.eye(2) * ruido_sensor_gps

# Matriz de Covariância do Ruído de Processo (Q) 6x6
# Mede o ruido associado a incertezas na dinâmica do carro (aceleração, forças externas, etc.)
Q = np.eye(6) * ruido_processo_fisico

# Simula o cenário real do carro
x_real, y_real = 0.0, 0.0
vx_real, vy_real = 30.0, 0.0 # Velocidade inicial

trajetoria_real = []
medicoes_gps = []

# Semente fixa para garantir o mesmo padrão de ruído em todas as execuções
np.random.seed(42)

for t in tempo:
    # Aceleração dinâmica senoidal: força lateral oscilando esquerda/direita
    ax_real = 2.0 * np.cos(t)
    ay_real = 5.0 * np.sin(t)
    
    # Aplica o calculo de cinematica 
    # s = s0 + v0*dt + 0.5*a*dt^2  
    x_real = x_real + vx_real * dt + 0.5 * ax_real * dt**2
    y_real = y_real + vy_real * dt + 0.5 * ay_real * dt**2
    #v = v0 + a*dt
    vx_real = vx_real + ax_real * dt
    vy_real = vy_real + ay_real * dt
    
    trajetoria_real.append([x_real, vx_real, ax_real, y_real, vy_real, ay_real])
    
    # Geração da leitura do GPS adicionando ruído Gaussiano baseado na matriz R 
    gps_x = x_real + np.random.normal(0, np.sqrt(R[0, 0]))
    gps_y = y_real + np.random.normal(0, np.sqrt(R[1, 1]))
    medicoes_gps.append([gps_x, gps_y])

trajetoria_real = np.array(trajetoria_real)
medicoes_gps = np.array(medicoes_gps)


#Execução do filtro
# Inicialização das estimativas 
X_estimado = np.zeros((6, 1))
X_estimado[0, 0] = medicoes_gps[0, 0] # Posição X inicial baseada no sensor 
X_estimado[3, 0] = medicoes_gps[0, 1] # Posição Y inicial baseada no sensor

# Matriz de covariância inicial do erro (P)
P = np.eye(6) * 10.0

trajetoria_filtrada = []

for k in range(passos):
    # Predicao
    X_previsto = A @ X_estimado
    P_previsto = (A @ P @ A.T) + Q
    
    # Coleta do vetor de medição atual Z 2x1 (posição X e Y do GPS)
    Z = medicoes_gps[k].reshape(2, 1)
    
    # Atualizacao
    S_matriz = (H @ P_previsto @ H.T) + R
    K = P_previsto @ H.T @ np.linalg.inv(S_matriz)
    
    # Correção da estimativa usando o vetor de inovação Z - H*X_previsto
    X_estimado = X_previsto + K @ (Z - (H @ X_previsto))
    
    # Atualização da covariância do erro de estimação - matriz P
    P = P_previsto - K @ H @ P_previsto
    
    # Armazena as posições X (índice 0) e Y (índice 3) limpas pelo filtro
    trajetoria_filtrada.append(X_estimado.flatten().tolist())

trajetoria_filtrada = np.array(trajetoria_filtrada)


# Visualização dos resultados
fig, eixos = plt.subplots(3, 2, figsize=(15, 12))
fig.canvas.manager.set_window_title('Filtro de Kalman')

# Posição X
eixos[0, 0].plot(tempo, trajetoria_real[:, 0], 'b-', label='Real Oculta')
eixos[0, 0].plot(tempo, medicoes_gps[:, 0], 'r.', alpha=0.5, label='Medição GPS')
eixos[0, 0].plot(tempo, trajetoria_filtrada[:, 0], 'g--', linewidth=2, label='Filtro de Kalman')
eixos[0, 0].set_title("Telemetria de Posição - Eixo X", fontsize=10, fontweight='bold')
eixos[0, 0].set_ylabel("Posição (m)")
eixos[0, 0].grid(True, linestyle='--', alpha=0.5)
eixos[0, 0].legend()

# Posição Y
eixos[0, 1].plot(tempo, trajetoria_real[:, 3], 'b-', label='Real Oculta')
eixos[0, 1].plot(tempo, medicoes_gps[:, 1], 'r.', alpha=0.5, label='Medição GPS')
eixos[0, 1].plot(tempo, trajetoria_filtrada[:, 3], 'g--', linewidth=2, label='Filtro de Kalman')
eixos[0, 1].set_title("Telemetria de Posição - Eixo Y", fontsize=10, fontweight='bold')
eixos[0, 1].set_ylabel("Posição (m)")
eixos[0, 1].grid(True, linestyle='--', alpha=0.5)
eixos[0, 1].legend()

# Velocidade X
eixos[1, 0].plot(tempo, trajetoria_real[:, 1], 'b-', label='Velocidade X Real')
eixos[1, 0].plot(tempo, trajetoria_filtrada[:, 1], 'g--', linewidth=2, label='Estimada por Kalman')
eixos[1, 0].set_title("Estimação de Variável Oculta: Velocidade X", fontsize=10, fontweight='bold')
eixos[1, 0].set_ylabel("Velocidade (m/s)")
eixos[1, 0].grid(True, linestyle='--', alpha=0.5)
eixos[1, 0].legend()

# Velocidade Y
eixos[1, 1].plot(tempo, trajetoria_real[:, 4], 'b-', label='Velocidade Y Real')
eixos[1, 1].plot(tempo, trajetoria_filtrada[:, 4], 'g--', linewidth=2, label='Estimada por Kalman')
eixos[1, 1].set_title("Estimação de Variável Oculta: Velocidade Y", fontsize=10, fontweight='bold')
eixos[1, 1].set_ylabel("Velocidade (m/s)")
eixos[1, 1].grid(True, linestyle='--', alpha=0.5)
eixos[1, 1].legend()

# Aceleração X
eixos[2, 0].plot(tempo, trajetoria_real[:, 2], 'b-', label='Aceleração X Real')
eixos[2, 0].plot(tempo, trajetoria_filtrada[:, 2], 'g--', linewidth=2, label='Estimada por Kalman')
eixos[2, 0].set_title("Estimação de Variável Oculta: Aceleração X", fontsize=10, fontweight='bold')
eixos[2, 0].set_xlabel("Tempo (s)")
eixos[2, 0].set_ylabel("Aceleração (m/s²)")
eixos[2, 0].grid(True, linestyle='--', alpha=0.5)
eixos[2, 0].legend()

# Aceleração Y
eixos[2, 1].plot(tempo, trajetoria_real[:, 5], 'b-', label='Aceleração Y Real')
eixos[2, 1].plot(tempo, trajetoria_filtrada[:, 5], 'g--', linewidth=2, label='Estimada por Kalman')
eixos[2, 1].set_title("Estimação de Variável Oculta: Aceleração Y", fontsize=10, fontweight='bold')
eixos[2, 1].set_xlabel("Tempo (s)")
eixos[2, 1].set_ylabel("Aceleração (m/s²)")
eixos[2, 1].grid(True, linestyle='--', alpha=0.5)
eixos[2, 1].legend()

plt.tight_layout()
plt.show()