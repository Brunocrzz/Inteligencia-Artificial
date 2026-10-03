import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_circles

# Gerar dados não linearmente separáveis (círculos concêntricos)
X, y = make_circles(n_samples=100, factor=0.3, noise=0.1, random_state=42)

# Criar classificador SVM com kernel RBF
clf = svm.SVC(kernel='rbf', gamma='scale')
clf.fit(X, y)

# Criar grid para visualização da fronteira
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotar pontos e fronteiras
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=50, edgecolors='k')
plt.contour(xx, yy, Z, levels=[0], colors='k')  # Fronteira de decisão
plt.title('SVM não linear com kernel RBF')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
