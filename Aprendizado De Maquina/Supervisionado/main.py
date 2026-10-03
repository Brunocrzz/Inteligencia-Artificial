import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importações de módulos do Scikit-Learn para dados, amostragem e modelos
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Importação do SMOTE da biblioteca imbalanced-learn para engenharia de dados
from imblearn.over_sampling import SMOTE

def pipeline_deteccao_fraudes():
    # Gera 10.000 amostras com 20 variáveis numéricas independentes.
    # weights=[0.99, 0.01] replica o desbalanceamento de 1% fraude.
    # flip_y=0 impede que o scikit-learn introduza ruído aleatório trocando rótulos arbitrariamente.
    # random_state=42 garante a semente pseudoaleatória fixa para reprodutibilidade matemática.
    X, y = make_classification(n_samples=10000, n_features=20, n_classes=2, weights=[0.99, 0.01], flip_y=0, random_state=42)

    # Realiza o split clássico: 75% para Treinamento e 25% para Validação/Teste.
    # O parâmetro crítico 'stratify=y' obriga o algoritmo de partição a manter 
    # a mesma proporção exata de 1% de fraudes em ambos os blocos resultantes.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    # Criamos um classificador Random Forest simulando a parametrização inicial.
    # max_depth=5 atua como um limitador de crescimento para evitar overfitting.
    clf_desbalanceado = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    
    # O método .fit descobre as regras de divisão matemática a partir do treino enviesado.
    clf_desbalanceado.fit(X_train, y_train)
    
    # Classificação do conjunto de teste e computação da matriz de confusao
    y_pred_desb = clf_desbalanceado.predict(X_test)
    matriz_desb = confusion_matrix(y_test, y_pred_desb)


    # Balanceamento de dados com SMOTE (Synthetic Minority Over-sampling Technique)  
    smote = SMOTE(random_state=42)
    
    # O ajuste por reamostragem ocorre somente nos dados de treino. 
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    # Realiza a verificação da proporção de classes após o balanceamento
    clf_so_smote = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    clf_so_smote.fit(X_train_bal, y_train_bal)
    
    # Predição e matriz de confusão para o cenário intermediário
    y_pred_so_smote = clf_so_smote.predict(X_test)
    matriz_so_smote = confusion_matrix(y_test, y_pred_so_smote)


    # Modelo otimizado com GridSearchCV para encontrar a melhor combinação de hiperparâmetros
    # Definimos um dicionário (grade) contendo as combinações estruturais a testar.
    # Cruzaremos 3 limites de profundidade com 3 volumes de árvores (9 combinações).
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [3, 5, 10]
    }
    
    # Realiza Validação Cruzada em 3 dobras.
    # scoring='f1' define que o critério de seleção ignorará a acurácia geral, 
    # elegendo a combinação que maximizar a média harmônica entre Precision e Recall.
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, scoring='f1', cv=3, verbose=0) 
    grid_search.fit(X_train_bal, y_train_bal)
    melhor_clf = grid_search.best_estimator_
    
    # Predição final com o modelo calibrado
    y_pred_otimizado = melhor_clf.predict(X_test)
    matriz_otimizado = confusion_matrix(y_test, y_pred_otimizado)


    # Saída de métricas comparativas entre os três cenários
    print("\n" + "="*70)
    print("      PAINEL COMPARATIVO DE MÉTRICAS - APRENDIZADO SUPERVISIONADO")
    print("="*70)
    
    print("\n[CENÁRIO 1] Random Forest Sem Tratamento (Desbalanceado)")
    print(classification_report(y_test, y_pred_desb, target_names=['Legítima', 'Fraude']))
    
    print("-"*70)
    print("\n[CENÁRIO 2] Random Forest + Apenas SMOTE (max_depth=5)")
    print(classification_report(y_test, y_pred_so_smote, target_names=['Legítima', 'Fraude']))
    
    print("-"*70)
    print(f"\n[OTIMIZAÇÃO] Melhores parâmetros determinados pelo GridSearch:")
    print(f" -> max_depth: {grid_search.best_params_['max_depth']}")
    print(f" -> n_estimators: {grid_search.best_params_['n_estimators']}")
    print("-"*70)
    
    print("\n[CENÁRIO 3] Random Forest + SMOTE + GridSearchCV (Otimizado)")
    print(classification_report(y_test, y_pred_otimizado, target_names=['Legítima', 'Fraude']))
    print("="*70)


    # Plotagem comparativa das matrizes de confusão para os três cenários
    print("\n[-] Gerando interface gráfica comparativa...")
    sns.set_theme(style="whitegrid")
    
    # Inicializa uma janela contendo 1 linha e 3 eixos de desenho posicionados em paralelo
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Subgráfico 1: Modelo Inicial com gradiente de vermelho (Alerta de Viés)
    sns.heatmap(matriz_desb, annot=True, fmt='d', cmap='Reds', ax=axes[0],
                xticklabels=['Legítima', 'Fraude'], yticklabels=['Legítima', 'Fraude'])
    axes[0].set_title("1. Modelo Desbalanceado", fontweight='bold', pad=10)
    axes[0].set_ylabel("Classe Real (Gabarito)")
    axes[0].set_xlabel("Classe Predita")

    # Subgráfico 2: Modelo SMOTE Raso com gradiente de laranja (Transição Metodológica)
    sns.heatmap(matriz_so_smote, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
                xticklabels=['Legítima', 'Fraude'], yticklabels=['Legítima', 'Fraude'])
    axes[1].set_title("2. Apenas SMOTE (Raso)", fontweight='bold', pad=10)
    axes[1].set_ylabel("Classe Real (Gabarito)")
    axes[1].set_xlabel("Classe Predita")

    # Subgráfico 3: Nosso Modelo Otimizado com gradiente de azul (Configuração Alvo)
    sns.heatmap(matriz_otimizado, annot=True, fmt='d', cmap='Blues', ax=axes[2],
                xticklabels=['Legítima', 'Fraude'], yticklabels=['Legítima', 'Fraude'])
    axes[2].set_title("3. SMOTE + GridSearch (Ideal)", fontweight='bold', pad=10)
    axes[2].set_ylabel("Classe Real (Gabarito)")
    axes[2].set_xlabel("Classe Predita")

    # Ajusta as margens automaticamente para evitar sobreposição de textos nas bordas
    plt.tight_layout()
    
    # Interrompe o terminal abrindo a janela gráfica nativa do sistema operacional
    plt.show()


if __name__ == "__main__":
    pipeline_deteccao_fraudes()