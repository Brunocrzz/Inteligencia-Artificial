# =============================================================================
# PCA SIMPLES E CLARO - Para Aula
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. CARREGAR DADOS IRIS
print("🔄 Carregando Iris Dataset...")
iris = load_iris()
X_original = iris.data  # 150 flores x 4 medidas
print(f"📊 Dados ORIGINAIS: {X_original.shape} (4 dimensões)")

# 2. PADRONIZAR (IMPORTANTE!)
scaler = StandardScaler()
X_padronizado = scaler.fit_transform(X_original)
print("✅ Dados PADRONIZADOS (média=0, desvio=1)")

# 3. PCA - REDUZIR DE 4D PARA 2D
pca = PCA(n_components=2)
X_pca_2d = pca.fit_transform(X_padronizado)
print(f"✅ PCA 2D: {X_pca_2d.shape}")

# =============================================================================
# PLOTS SIMPLES E CLAROS
# =============================================================================

plt.figure(figsize=(15, 10))

# PLOT 1: ANTES x DEPOIS (mais intuitivo)
plt.subplot(2, 3, 1)
plt.scatter(X_original[:, 0], X_original[:, 2], c=iris.target, cmap='viridis', s=50)
plt.title('ANTES do PCA\n(2 das 4 dimensões originais)', fontsize=12)
plt.xlabel('Comprimento Sépala')
plt.ylabel('Comprimento Pétala')

plt.subplot(2, 3, 2)
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=iris.target, cmap='viridis', s=50)
plt.title('DEPOIS do PCA\n(2 dimensões novas)', fontsize=12)
plt.xlabel('PC1')
plt.ylabel('PC2')

# PLOT 2: QUANTOS % DE INFORMAÇÃO CADA PC GUARDA
plt.subplot(2, 3, 3)
pc1 = pca.explained_variance_ratio_[0] * 100
pc2 = pca.explained_variance_ratio_[1] * 100
total = (pc1 + pc2)
plt.bar(['PC1', 'PC2'], [pc1, pc2], color=['orange', 'lightblue'])
plt.title(f' Quanto cada PC explica?\nPC1+PC2 = {total:.0f}%', fontsize=12)
plt.ylabel('% Informação')

# PLOT 3: COTOVELO (quantas dimensões precisamos?)
plt.subplot(2, 3, 4)
pca_full = PCA().fit(X_padronizado)
plt.plot([1,2,3,4], np.cumsum(pca_full.explained_variance_ratio_)*100, 
         'ro-', linewidth=3, markersize=10)
plt.axhline(y=95, color='red', linestyle='--', alpha=0.7, label='95%')
plt.title('Cotovelo: Onde parar?', fontsize=12)
plt.ylabel('Variância Acumulada (%)')
plt.xlabel('Número de PCs')
plt.legend()
plt.ylim(0, 105)
plt.grid(True, alpha=0.3)

# PLOT 4: RESUMO VISUAL
plt.subplot(2, 3, 5)
plt.text(0.1, 0.7, f'DADOS ORIGINAIS:\n• 4 dimensões\n• {X_original.shape[0]} amostras', 
         fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
plt.text(0.1, 0.3, f'DADOS PCA:\n• 2 dimensões\n• {total:.0f}% informação\n• Mesma separação!', 
         fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
plt.title('RESUMO', fontsize=12)
plt.axis('off')

# PLOT 6: Tabela de números
plt.subplot(2, 3, 6)
info = f"""
PC1: {pc1:.1f}%
PC2: {pc2:.1f}%  
TOTAL: {total:.1f}%

{chr(10004) if total >= 90 else chr(10006)} Excelente compressão!
"""
plt.text(0.05, 0.5, info, fontsize=16, fontweight='bold', 
         va='center', ha='left',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat"))
plt.title('NÚMEROS FINAIS', fontsize=12)
plt.axis('off')

plt.tight_layout()
plt.show()

# RESULTADO FINAL
print(f"\n RESULTADO:")
print(f"   PC1 guarda:  {pc1:.1f}% da informação")
print(f"   PC2 guarda:  {pc2:.1f}% da informação") 
print(f"   TOTAL 2D:    {total:.1f}% (de 4D para 2D!)")
print(f"\n PCA COMPRESSOU 4 DIMENSÕES EM 2, mantendo {total:.0f}% da informação!")
