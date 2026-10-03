import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# Criando dados simples para duas classes
X = np.array([[2, 3], [3, 3], [2, 2], [7, 8], [8, 8], [7, 7]])
y = [0, 0, 0, 1, 1, 1]  # Classes correspondentes

# Criando o classificador SVM com kernel linear
clf = svm.SVC(kernel='linear')
clf.fit(X, y)

# Gerando um grid para visualização da fronteira de decisão
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotar pontos e fronteiras de decisão
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=50, edgecolors='k')
plt.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=['--', '-', '--'], colors='k')
plt.title("Exemplo simples de SVM com kernel linear")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
