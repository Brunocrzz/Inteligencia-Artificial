import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# Geracao de dados sintéticos para simular clientes de um shopping
X,y = make_blobs(n_samples=600, n_features=2, centers=4, cluster_std=1.2, random_state=42)

# Ajuste do espaço de dados para simular um cenário financeiro realista
x_min, x_max = X[:, 0].min() - 2, X[:, 0].max() + 2
y_min, y_max = X[:, 1].min() - 2, X[:, 1].max() + 2

# Gera até 15 outliers automáticos distribuídos aleatoriamente no espaço de dados
np.random.seed(42)
outliers_automaticos = np.zeros((15, 2))
outliers_automaticos[:, 0] = np.random.uniform(x_min, x_max, 15)
outliers_automaticos[:, 1] = np.random.uniform(y_min, y_max, 15)
X = np.vstack([X, outliers_automaticos])

# Redimensionamento matemático para criar escalas financeiras reais
# X[:, 0] é a renda anual (em milhares de reais) e X[:, 1] é o score de gastos (1 a 100)
X[:, 0] = (X[:, 0] - X[:, 0].min()) / (X[:, 0].max() - X[:, 0].min()) * 100 + 15  # Renda Anual (k$)
X[:, 1] = (X[:, 1] - X[:, 1].min()) / (X[:, 1].max() - X[:, 1].min()) * 80 + 10   # Score de Gastos

# Padronização Z-Score para os algoritmos baseados em distância
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Otimização do número de clusters para K-Means usando o método do cotovelo e a métrica de Silhouette
wcss = []
melhor_k = 2
maior_silhouette = -1

# Descobrir o melhor K (número de clusters) entre 1 e 10
for i in range(1, 11):
    kmeans_test = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans_test.fit(X_scaled)
    wcss.append(kmeans_test.inertia_)
    if i > 1:
        score = silhouette_score(X_scaled, kmeans_test.labels_) # Cálculo da métrica de Silhouette
        if score > maior_silhouette: # Comparação para encontrar o melhor K de acordo com o score de Silhouette
            maior_silhouette = score
            melhor_k = i

# Definição do modelo final de K-Means com o melhor K encontrado
kmeans_final = KMeans(n_clusters=melhor_k, init='k-means++', random_state=42)
labels_kmeans = kmeans_final.fit_predict(X_scaled)


# Aplica o DBSCAN para identificar clusters de densidade e outliers
eps_val = 0.21  
dbscan = DBSCAN(eps=eps_val, min_samples=5) # Definição de eps e min_samples para o DBSCAN
labels_dbscan = dbscan.fit_predict(X_scaled)
n_outliers = np.sum(labels_dbscan == -1) # Contagem de outliers identificados pelo DBSCAN

# Funcoes auxiliares para reverter a escala dos eixos e plotar as fronteiras de decisão do K-Means
sns.set_theme(style="whitegrid")

# Função para reverter a escala dos eixos para os valores originais de renda e gastos
def reverter_escala_eixos(ax):
    # Fixa os limites visuais na escala padronizada equivalente aos valores reais desejados
    # Renda (X): De 0 a 120 | Score (Y): De 0 a 100
    limites_reais_x = np.array([0, 120])
    limites_reais_y = np.array([0, 100])
    
    # Converte os limites reais desejados para a escala Z-Score correspondente
    lim_z_x = (limites_reais_x - scaler.mean_[0]) / scaler.scale_[0]
    lim_z_y = (limites_reais_y - scaler.mean_[1]) / scaler.scale_[1]
    ax.set_xlim(lim_z_x[0], lim_z_x[1])
    ax.set_ylim(lim_z_y[0], lim_z_y[1])

    ticks_reais_x = np.arange(0, 121, 10)  
    ticks_reais_y = np.arange(0, 101, 10)  
    ticks_z_x = (ticks_reais_x - scaler.mean_[0]) / scaler.scale_[0]
    ticks_z_y = (ticks_reais_y - scaler.mean_[1]) / scaler.scale_[1]
    
    ax.set_xticks(ticks_z_x)
    ax.set_xticklabels([f"{val:.0f}" for val in ticks_reais_x])
    ax.set_yticks(ticks_z_y)
    ax.set_yticklabels([f"{val:.0f}" for val in ticks_reais_y])

# Função para plotar as fronteiras de decisão do K-Means
def plot_fronteiras_kmeans(ax, model, data):
    h = .02  
    x_min, x_max = data[:, 0].min() - 0.5, data[:, 0].max() + 0.5
    y_min, y_max = data[:, 1].min() - 0.5, data[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.04, cmap='Set1')


# Plota o gráfico inicial com blobs de clientes sem classificação e sem fronteiras de decisão
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], color='dimgray', s=45, alpha=0.7, edgecolors='none')
ax1.set_title("Cenário Inicial: Distribuição de Clientes sem Rótulos Predefinidos", fontweight='bold', fontsize=12)
ax1.set_xlabel("Renda Anual do Cliente (milhares de reais)")
ax1.set_ylabel("Score de Gastos no Shopping (1 a 100)")
reverter_escala_eixos(ax1)
plt.tight_layout()
plt.show()


# Plota o gráfico do método do cotovelo para determinar o número ideal de clusters para o K-Means
plt.figure(figsize=(7, 4.5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='darkviolet', linewidth=2)
plt.axvline(x=melhor_k, color='red', linestyle=':', label=f'K Ideal Sugerido ({melhor_k})')
plt.title('Método do Cotovelo (Elbow Method)', fontweight='bold')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('WCSS (Inércia) [Distância Quadrática Intra-cluster]')
plt.legend()
plt.tight_layout()
plt.show()


# Plota o gráfico dos dados brutos com os centroides do K-Means e as fronteiras de decisão de Voronoi
fig3, ax3 = plt.subplots(figsize=(8, 6))
plot_fronteiras_kmeans(ax3, kmeans_final, X_scaled)
ax3.scatter(X_scaled[:, 0], X_scaled[:, 1], color='gainsboro', s=45, edgecolors='silver', label='Clientes (Sem Classificação)')
ax3.scatter(kmeans_final.cluster_centers_[:, 0], kmeans_final.cluster_centers_[:, 1], 
            s=250, c='red', marker='*', edgecolors='black', zorder=5, label='Centroides Fundamentais')
ax3.set_title(f"Centroides e Fronteiras Reais de Voronoi (K={melhor_k})", fontweight='bold', fontsize=12)
ax3.set_xlabel("Renda Anual do Cliente (milhares de reais)")
ax3.set_ylabel("Score de Gastos no Shopping (1 a 100)")
reverter_escala_eixos(ax3)
ax3.legend(loc='upper right')
plt.tight_layout()
plt.show()


# Plota os resultados do K-Means e DBSCAN lado a lado para comparação visual
fig4, axes4 = plt.subplots(1, 2, figsize=(16, 7))


# Subgráfico 1: K-Means
# Definição de perfis de clientes com base na posição dos centroides do K-Means no espaço Z-Score
perfeis_kmeans_nomes = {}
for kid in range(melhor_k):
    centro_v = kmeans_final.cluster_centers_[kid]
    c_renda, c_gastos = centro_v[0], centro_v[1]
    
    # Classificação baseada no posicionamento quadrático do centroide no espaço Z-Score
    if c_renda > 0 and c_gastos > 0:
        perfeis_kmeans_nomes[kid] = "Clientes VIP"
    elif c_renda < 0 and c_gastos > 0:
        perfeis_kmeans_nomes[kid] = "Clientes Impulsivos"
    elif c_renda > 0 and c_gastos < 0:
        perfeis_kmeans_nomes[kid] = "Clientes Conservadores"
    elif c_renda < 0 and c_gastos < 0:
        perfeis_kmeans_nomes[kid] = "Clientes Ocasionais"
paleta_kmeans = plt.colormaps['viridis'].resampled(melhor_k)

for kid in range(melhor_k):
    membros_k = X_scaled[labels_kmeans == kid]
    axes4[0].scatter(membros_k[:, 0], membros_k[:, 1], color=paleta_kmeans(kid), 
                     s=45, alpha=0.8, edgecolors='k', label=perfeis_kmeans_nomes[kid])

# Desenha as estrelas dos centroides por cima dos pontos
axes4[0].scatter(kmeans_final.cluster_centers_[:, 0], kmeans_final.cluster_centers_[:, 1], 
                 s=200, c='red', marker='*', edgecolors='black', zorder=5, label='Centroides')

axes4[0].set_title(f"Segmentação Forçada via K-Means (K={melhor_k})", fontweight='bold', fontsize=12)
axes4[0].set_xlabel("Renda Anual (k$)")
axes4[0].set_ylabel("Score de Gastos (1-100)")
reverter_escala_eixos(axes4[0])
axes4[0].legend(loc='upper right', frameon=True, shadow=True, facecolor='white', framealpha=0.9)



# Subgráfico 2: DBSCAN
# Definição de perfis de clientes com base na média de renda e gastos de cada cluster identificado pelo DBSCAN
perfis_nomes = {}
cluster_ids_validos = [c for c in set(labels_dbscan) if c != -1]

# Para cada cluster válido, calcula a média de renda e gastos e classifica o perfil do cliente arbitrariamente
for cid in cluster_ids_validos:
    membros = X_scaled[labels_dbscan == cid]
    media_renda = membros[:, 0].mean()
    media_gastos = membros[:, 1].mean()
    
    if media_renda > 0 and media_gastos > 0:
        perfis_nomes[cid] = "Clientes VIP"
    elif media_renda < 0 and media_gastos > 0:
        perfis_nomes[cid] = "Clientes Impulsivos"
    elif media_renda > 0 and media_gastos < 0:
        perfis_nomes[cid] = "Clientes Conservadores"
    elif media_renda < 0 and media_gastos < 0:
        if media_renda > -0.5:
            perfis_nomes[cid] = "Clientes de Transição"
        else:
            perfis_nomes[cid] = "Clientes Ocasionais"
paleta_cores = plt.colormaps['rainbow'].resampled(len(cluster_ids_validos))

for idx, cid in enumerate(cluster_ids_validos):
    membros = X_scaled[labels_dbscan == cid]
    axes4[1].scatter(membros[:, 0], membros[:, 1], color=paleta_cores(idx), 
                     s=45, alpha=0.7, edgecolors='k', label=perfis_nomes[cid])

mascara_outliers = (labels_dbscan == -1)
axes4[1].scatter(X_scaled[mascara_outliers, 0], X_scaled[mascara_outliers, 1], c='red', marker='x', 
                  s=100, linewidths=2.5, label=f'Outliers Isolados [{n_outliers} pontos]')

for pt in X_scaled[mascara_outliers]:
    circulo_eps = plt.Circle((pt[0], pt[1]), radius=eps_val, color='red', fill=False, linestyle=':', alpha=0.4, linewidth=1.2)
    axes4[1].add_patch(circulo_eps)

axes4[1].set_title(f"Divisão por Densidade e Outliers via DBSCAN (eps={eps_val})", fontweight='bold', fontsize=12)
axes4[1].set_xlabel("Renda Anual (milhares de reais)")
axes4[1].set_ylabel("Score de Gastos (1-100)")
reverter_escala_eixos(axes4[1])
axes4[1].legend(loc='upper right', frameon=True, shadow=True, facecolor='white', framealpha=0.9)

plt.tight_layout()
plt.show()