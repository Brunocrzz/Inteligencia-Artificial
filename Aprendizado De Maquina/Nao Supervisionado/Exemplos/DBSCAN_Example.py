import matplotlib.pyplot as plt
import mglearn  # Biblioteca necessária para mglearn.cm2
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

# Padronizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# DBSCAN com parâmetros adequados para make_moons
dbscan = DBSCAN(eps=0.3, min_samples=5)  # Ajuste crucial!
clusters = dbscan.fit_predict(X_scaled)

# Plotar resultados
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm2, s=60)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.title("DBSCAN no dataset make_moons")
plt.show()  # Exibe o gráfico
