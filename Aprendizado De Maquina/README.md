# Aprendizado de Máquina

Esta pasta reúne os projetos da última parte da disciplina, um para cada um dos quatro grandes paradigmas de **Machine Learning**:

| Paradigma | Pasta | Problema | Técnicas | Relatório |
|---|---|---|---|---|
| **Supervisionado** | [`Supervisionado/`](./Supervisionado) | Detecção de fraudes em cartão de crédito | Random Forest, SMOTE, GridSearchCV | `Aprendizado Supervisionado.pdf` |
| **Não Supervisionado** | [`Nao Supervisionado/`](./Nao%20Supervisionado) | Segmentação de clientes de shopping | K-Means, Método do Cotovelo, Silhouette, DBSCAN | `Aprendizado não Supervisionado.pdf` |
| **Por Reforço** | [`Reforco/`](./Reforco) | Agente que busca um tesouro em um mapa | Q-Learning, ε-greedy, equação de Bellman | `Aprendizado por Reforço.pdf` |
| **Deep Learning** | [`Deep Learning/`](./Deep%20Learning) | Classificação do alfabeto da língua de sinais (ASL) | CNN em PyTorch, Data Augmentation, BatchNorm, Dropout | `Deep Learning.pdf` |

### Qual a diferença entre eles?

- **Supervisionado:** o modelo aprende com exemplos **rotulados** (entrada `X` → resposta `Y`) e depois prevê o rótulo de dados novos. É usado em classificação e regressão.
- **Não supervisionado:** os dados **não têm rótulo**, então o algoritmo precisa descobrir sozinho a estrutura deles, como grupos (*clusters*), anomalias ou reduções de dimensionalidade.
- **Por reforço:** um **agente** interage com um **ambiente**, recebe **recompensas ou penalidades** e aprende por tentativa e erro uma política que maximize o retorno acumulado.
- **Deep Learning:** redes neurais com muitas camadas que aprendem **representações hierárquicas** direto dos dados brutos (como pixels), sem precisar extrair características manualmente.

### Exemplos de aula

As pastas `Exemplos/` têm **códigos apresentados pelo professor em aula**, usados como referência para os trabalhos. Alguns precisam de bibliotecas extras, como `mglearn`, `pandas`, `tensorflow`/`keras` e `pillow`.

---

## Aprendizado Supervisionado: detecção de fraudes

**Problema:** classificar transações como **Legítima (0)** ou **Fraude (1)**. O principal desafio é o **desbalanceamento**: só **1%** das transações são fraudes. Um modelo que chame tudo de "legítimo" teria 99% de acurácia sem detectar nenhuma fraude (a chamada "armadilha da acurácia").

**Dados:** 10.000 transações sintéticas com 20 atributos, geradas com `make_classification(weights=[0.99, 0.01])`.

**Técnicas**
- **Random Forest:** *ensemble* de árvores de decisão com amostragem *bootstrap*, seleção aleatória de atributos e votação por maioria.
- **SMOTE:** cria amostras **sintéticas** da classe minoritária interpolando vizinhos próximos. No treino, as classes passaram de 7.425 x 75 para 7.425 x 7.425.
- **GridSearchCV:** busca de hiperparâmetros (`max_depth` ∈ {3, 5, 10} e `n_estimators` ∈ {10, 50, 100}) com validação cruzada de 3 *folds*, otimizando o **F1-Score**.

**Resultados no conjunto de teste (fraude = classe positiva)**

| Cenário | VN | FP | FN | VP | Acurácia | Precisão | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| Desbalanceado | 2475 | 0 | 21 | 4 | 99,16% | 1,00 | 0,16 | 0,28 |
| SMOTE (max_depth=5) | 2261 | 214 | 5 | 20 | 91,24% | 0,09 | 0,80 | 0,15 |
| SMOTE + GridSearch | 2376 | 99 | 9 | 16 | 96,00% | 0,14 | 0,64 | 0,23 |

O primeiro cenário mostra a armadilha da acurácia: 99% de acerto, mas 84% das fraudes passam sem ser detectadas. Só com o SMOTE, o recall sobe para 80%, ao custo de 214 alarmes falsos. Com o GridSearch (`max_depth=10`), os alarmes falsos caem para 99 e o modelo chega a um equilíbrio melhor entre detectar fraudes e não incomodar clientes legítimos.

**Parâmetros que podem ser alterados** (em `main.py`)

| Parâmetro | Valor | O que significa |
|---|---|---|
| `n_samples` | 10000 | Quantidade de transações geradas |
| `n_features` | 20 | Quantidade de atributos de cada transação |
| `weights` | `[0.99, 0.01]` | Proporção de transações legítimas e de fraudes |
| `test_size` | 0.25 | Fração dos dados separada para teste (25%) |
| `n_estimators` | 50 | Número de árvores da floresta (cenários 1 e 2) |
| `max_depth` | 5 | Profundidade máxima de cada árvore (cenários 1 e 2) |
| `param_grid` | `{n_estimators: [10, 50, 100], max_depth: [3, 5, 10]}` | Combinações testadas pelo GridSearch (cenário 3) |
| `cv` | 3 | Número de partes (*folds*) da validação cruzada |
| `scoring` | `'f1'` | Métrica usada para escolher a melhor combinação |

- **`weights`:** é o que cria o desbalanceamento. Com `[0.999, 0.001]` (0,1% de fraudes), sobram pouquíssimos exemplos de fraude para aprender e o cenário 1 fica ainda pior. Com `[0.5, 0.5]`, o problema deixa de ser desbalanceado e o SMOTE perde o sentido, porque os três cenários ficam parecidos.
- **`n_samples`:** mais amostras dão mais exemplos de fraude (1% de 10.000 são só 100) e métricas mais estáveis, mas o GridSearch demora mais.
- **`n_estimators`:** mais árvores deixam a votação mais estável e reduzem a variância, até um ponto em que o ganho fica pequeno e só o tempo de treino aumenta.
- **`max_depth`:** árvores mais profundas aprendem fronteiras de decisão mais detalhadas. No cenário 2, a profundidade 5 deixou o modelo "raso" demais para os dados do SMOTE, gerando muitos alarmes falsos. Profundidade muito alta (ou `None`, sem limite) pode decorar o treino (*overfitting*).
- **`param_grid`:** cada valor adicionado multiplica o número de combinações testadas. Com 3 x 3 valores e `cv=3`, são 27 treinos. Adicionar, por exemplo, `max_depth: [3, 5, 10, 20, None]` amplia a busca, mas deixa a execução mais lenta.
- **`scoring`:** trocar `'f1'` por `'recall'` faz o GridSearch priorizar **pegar mais fraudes**, mesmo com mais alarmes falsos. Usar `'precision'` prioriza **errar menos ao acusar fraude**, mesmo deixando mais fraudes passarem. Já `'accuracy'` cai de novo na armadilha da acurácia.
- **`random_state=42`:** fixa a aleatoriedade para que os resultados sejam sempre os mesmos. Trocar o número gera outros dados e outras divisões de treino e teste.

**Como rodar**
```bash
cd "Aprendizado De Maquina/Supervisionado"
pip install -r requirements.txt   # numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn
python main.py
```
O terminal mostra o relatório de métricas dos três cenários e uma janela abre com as matrizes de confusão.

**Exemplos de aula (`Exemplos/`):** classificação do dataset Iris com KNN (`ML_Iris_*`), Naive Bayes com `Social_Network_Ads.csv` (deve ser executado a partir da pasta `Supervisionado/`), regressão logística e LinearSVC, SVM linear e com kernel, Random Forest e *underfitting*/*overfitting* com regressão polinomial. Inclui também o PDF de apoio *Beginners Guide to Naive Bayes Algorithm in Python*.

---

## Aprendizado Não Supervisionado: segmentação de clientes

**Problema:** agrupar **600 clientes** de um shopping pela **Renda Anual** e pelo **Score de Gastos** (de 1 a 100), sem nenhum rótulo prévio, e comparar um algoritmo **particional** (K-Means) com um **baseado em densidade** (DBSCAN), que detecta ruídos.

**Dados:** gerados com `make_blobs` (4 centros), com pontos aleatórios adicionados como ruído.

**Pipeline**
1. **Padronização Z-Score** (`StandardScaler`), para a renda não dominar a distância euclidiana.
2. **K-Means:** o **Método do Cotovelo** (inércia/WCSS de K=1 a 10) e o **Silhouette Score** (K ≥ 2) escolhem automaticamente **K = 4**. O gráfico mostra os centróides e as **fronteiras de Voronoi**, calculadas com `np.meshgrid` + `.predict()`.
3. **DBSCAN** com `eps = 0.21` e `min_samples = 5`. Pontos com rótulo `-1` são marcados como **outliers** (X vermelho).

**Perfis encontrados**

| Grupo | Renda | Gasto | Interpretação |
|---|---|---|---|
| VIP | Alta | Alto | Alto poder de compra e alta fidelidade |
| Impulsivos | Baixa | Alto | Consomem muito, com menos lastro financeiro |
| Conservadores | Alta | Baixo | Têm poder de compra, mas gastam pouco |
| Ocasionais | Baixa | Baixo | Pouca frequência e pouco gasto |
| Outliers | — | — | Anomalias isoladas (detectadas só pelo DBSCAN) |

**Conclusão:** o K-Means precisa colocar todos os pontos em algum cluster, então os outliers acabam deslocando os centróides. O DBSCAN contorna a forma real dos grupos e isola os ruídos sem que eles contaminem os perfis.

**Parâmetros que podem ser alterados** (em `main.py`)

| Parâmetro | Valor | O que significa |
|---|---|---|
| `n_samples` | 600 | Quantidade de clientes gerados |
| `centers` | 4 | Quantos grupos "reais" existem nos dados gerados |
| `cluster_std` | 1.2 | Espalhamento de cada grupo |
| Outliers | 15 | Pontos aleatórios adicionados como ruído |
| `range(1, 11)` | K de 1 a 10 | Valores de K testados no cotovelo e no Silhouette |
| `eps_val` | 0.21 | Raio da vizinhança do DBSCAN (na escala padronizada) |
| `min_samples` | 5 | Mínimo de vizinhos dentro do raio para um ponto ser "núcleo" |

- **`cluster_std`:** com valores menores (0.5, por exemplo), os grupos ficam compactos e bem separados, e K-Means e DBSCAN acertam com facilidade. Com valores maiores (2.0 ou mais), os grupos se sobrepõem, o Silhouette pode escolher um K diferente de 4 e o DBSCAN pode juntar grupos vizinhos.
- **`centers`:** muda quantos grupos existem de verdade. O K-Means escolhe o K pelo Silhouette, então deve acompanhar a mudança. Os nomes dos perfis (VIP, Impulsivos, Conservadores e Ocasionais) são dados pelo quadrante em que fica o centro de cada grupo (renda e gasto acima ou abaixo da média). Com mais de 4 grupos, dois grupos podem cair no mesmo quadrante e receber o mesmo nome.
- **Número de outliers (15):** mais ruído desloca ainda mais os centróides do K-Means, que precisa encaixar todo ponto em algum grupo, e deixa mais clara a vantagem do DBSCAN.
- **`eps_val`:** é o parâmetro mais sensível do DBSCAN.
  - **Menor (0.1):** quase nenhum ponto tem vizinhos suficientes, os grupos se fragmentam em vários pedaços e muitos clientes legítimos viram "outliers".
  - **Maior (0.5 ou mais):** o raio alcança pontos de outros grupos, os grupos se fundem e alguns outliers passam a ser absorvidos por eles.
- **`min_samples`:** aumentar exige regiões mais densas para formar um grupo, então mais pontos são marcados como ruído. Diminuir (2 ou 3) torna o DBSCAN mais permissivo, e até pequenos aglomerados de outliers podem virar um grupo.

**Como rodar**
```bash
cd "Aprendizado De Maquina/Nao Supervisionado"
pip install numpy matplotlib seaborn scikit-learn
python main.py
```

**Exemplos de aula (`Exemplos/`):** K-Means, DBSCAN no `make_moons`, PCA no dataset Iris e um Autoencoder no dataset `digits`. Alguns usam `mglearn`. A pasta `cache/` é gerada automaticamente pelo `joblib`/`mglearn` e é ignorada pelo Git.

---

## Aprendizado por Reforço: caça ao tesouro com Q-Learning

**Problema:** um agente começa na posição (1, 1) de um mapa **10x10** com paredes, moedas, armadilhas, uma chave, uma porta trancada e um **tesouro** (estado terminal). Ele precisa aprender uma rota que maximize a recompensa acumulada em até `MAX_STEPS = 200` passos.

**Modelagem como MDP**
- **Estados:** `(linha, coluna, tem_chave)`. Ter ou não a chave muda o estado, porque a porta só abre com ela.
- **Ações:** `UP`, `DOWN`, `LEFT`, `RIGHT`.
- **Transição:** determinística. Bater em uma parede, ou na porta sem a chave, mantém o agente no lugar.
- **Recompensas:**

| Evento | Recompensa |
|---|---|
| Movimento | −1 |
| Bater na parede | −10 |
| Armadilha | −50 |
| Moeda | +30 |
| Chave | +35 |
| Abrir a porta | +50 |
| Tesouro | +200 |

**Algoritmo:** **Q-Learning** tabular (Tabela Q em um dicionário) treinado por **5.000 episódios**, com:
- política **ε-greedy** e **desempate aleatório** entre as ações de mesmo valor Q;
- atualização pela **equação de Bellman** com `α = 0.1` e `γ = 0.9`;
- **decaimento de ε** de 1.0 até 0.01, com fator 0.999 por episódio.

**Resultado:** na avaliação determinística (ε = 0), o agente fez **249 pontos em 16 passos**. Ele pegou uma moeda, atravessou de propósito uma armadilha para pegar a chave e contornou a porta por uma abertura lateral até o tesouro. A rota funciona, mas é **subótima** (um máximo local). Explorando o mapa todo, daria para passar de 300 pontos. Melhorias possíveis seriam a **inicialização otimista** da Tabela Q ou punir mais as armadilhas (*reward shaping*).

**Estrutura**
- `main.py`: **versão final**, com ambiente, agente, treinamento, avaliação e gráficos em um único arquivo.
- `config.py`, `map_generator.py`, `environment.py`, `agent.py`, `trainer.py`, `renderer.py`: versão **modular** anterior do mesmo projeto. Ela salva a Tabela Q em `q_table.pkl` e é executada por `renderer.py`.
- `exemplo/`: exemplo de Q-Learning apresentado em aula.

**Como rodar**
```bash
cd "Aprendizado De Maquina/Reforco"
pip install numpy matplotlib seaborn
python main.py
```
O terminal mostra o progresso do treino e depois a rota aprendida passo a passo, com o mapa renderizado em texto. Para ver os gráficos de **recompensa e passos por episódio**, adicione `orchestrator.plot_metrics()` ao final do bloco `if __name__ == "__main__":`.

**Parâmetros que podem ser alterados** (em `main.py`)

*Recompensas e limite de passos (constantes no início do arquivo):*

- **`REWARD_MOVE` (−1):** é o custo de cada passo. Um custo maior (−5, por exemplo) faz o agente priorizar caminhos curtos e ignorar moedas mais distantes. Com 0, ele não tem pressa nenhuma e pode passear pelo mapa.
- **`REWARD_TRAP` (−50):** com o valor atual, ainda compensa atravessar a armadilha para chegar mais rápido à chave. Com uma penalidade bem maior (−300), o atalho deixa de valer a pena e o agente é forçado a procurar outra rota.
- **`REWARD_COIN`, `REWARD_KEY`, `REWARD_DOOR` (+30, +35, +50):** recompensas maiores tornam os desvios para coletar esses itens mais atrativos. Se forem pequenas em relação ao custo de andar até eles, o agente simplesmente as ignora.
- **`REWARD_TREASURE` (+200):** é o objetivo final. Se for pequena demais em relação às outras recompensas, o agente pode preferir ficar coletando itens a terminar o episódio.
- **`MAX_STEPS` (200):** limite de passos por episódio. Um limite menor encerra cedo os episódios de exploração aleatória, e no começo do treino o agente pode quase nunca chegar ao tesouro.

*Agente e treinamento:*

| Parâmetro | Onde | Valor | O que significa |
|---|---|---|---|
| `episodes` | `Trainer(episodes=5000)` | 5000 | Quantidade de episódios de treino |
| `learning_rate` (α) | `QLearningAgent` | 0.1 | Quanto cada nova experiência altera a Tabela Q |
| `discount_factor` (γ) | `QLearningAgent` | 0.9 | Peso das recompensas futuras |
| `exploration_rate` (ε) | `QLearningAgent` | 1.0 | Chance inicial de escolher uma ação aleatória |
| `exploration_decay` | `QLearningAgent` | 0.999 | Fator que multiplica ε ao fim de cada episódio |
| `min_exploration_rate` | `QLearningAgent` | 0.01 | Valor mínimo de ε |

- **α (taxa de aprendizado):** perto de 0, o agente aprende devagar, mas de forma estável. Perto de 1, cada experiência nova substitui quase todo o conhecimento anterior, e o aprendizado oscila.
- **γ (fator de desconto):** perto de 0, o agente é "imediatista" e só se importa com a próxima recompensa, então dificilmente planeja o caminho chave → porta → tesouro. Perto de 1, ele valoriza recompensas distantes e planeja rotas mais longas.
- **ε e o decaimento:** com fator 0,999, ε leva cerca de **4.600 episódios** para cair de 1,0 até 0,01. Se o número de episódios for reduzido para 1.000 sem mudar o fator, o treino termina com ε ≈ 0,37, ou seja, o agente ainda age aleatoriamente em 37% das vezes. Um decaimento mais rápido (0,99) para de explorar cedo e aumenta a chance de ficar preso no primeiro caminho bom que encontrar. Um mais lento explora mais e tem mais chance de achar a rota de mais de 300 pontos, mas exige mais episódios.
- **Mapa:** o layout fica na função `create_fixed_map()`. Cada linha da lista é uma linha do mapa, com os tipos `EMPTY`, `WALL`, `TRAP`, `COIN`, `KEY`, `DOOR` e `TREASURE`. Dá para mover itens, criar paredes ou mudar o tamanho (atualizando `MAP_WIDTH` e `MAP_HEIGHT`). O agente sempre começa em (1, 1), então essa posição precisa ser livre.

---

## Deep Learning: classificação do alfabeto ASL com CNN

**Problema:** classificar imagens de gestos do **alfabeto da Língua de Sinais Americana (ASL)** em **29 classes**: as letras A–Z e mais `SPACE`, `DELETE` e `NOTHING`. As imagens variam em rotação, iluminação, sombras, tom de pele e fundo.

**Dados:** dataset público [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet), baixado automaticamente via `kagglehub`. A divisão é **80% treino / 20% validação**.

**Pré-processamento e Data Augmentation**
- Redimensionamento de 200x200 para **64x64** RGB.
- No treino: espelhamento horizontal, rotação de até 15° e variação de 20% no brilho e no contraste.
- Normalização com média e desvio padrão da ImageNet.

**Arquitetura `AslCNN` (PyTorch)**
```
Entrada 3x64x64
→ [Conv 3x3 (32)  → BatchNorm → ReLU → MaxPool 2x2 → Dropout(0.25)]   → 32x32x32
→ [Conv 3x3 (64)  → BatchNorm → ReLU → MaxPool 2x2 → Dropout(0.25)]   → 64x16x16
→ [Conv 3x3 (128) → BatchNorm → ReLU → MaxPool 2x2 → Dropout(0.25)]   → 128x8x8
→ Flatten (8192) → Linear(256) → ReLU → Dropout(0.5) → Linear(29)
```
Função de perda `CrossEntropyLoss`, otimizador **Adam** (`lr = 1e-3`), `batch = 32`, **10 épocas**, com GPU (CUDA) se disponível.

**Resultados (10ª época)**

| Métrica | Valor |
|---|---|
| Acurácia de treino | 73,38% |
| **Acurácia de validação** | **97,84%** |
| Loss de treino | 0,7218 |
| Loss de validação | 0,1226 |

A validação ficou acima do treino porque o Dropout e o Data Augmentation só atuam durante o treino. A matriz de confusão mostra que os erros se concentram em sinais muito parecidos: **W → V** (109 erros) e **R → U** (49 erros). Nesses pares, a diferença está em detalhes finos dos dedos, que se perdem em parte na resolução de 64x64.

**Parâmetros que podem ser alterados** (no início de `main.py`)

| Parâmetro | Valor | O que significa |
|---|---|---|
| `IMAGE_SIZE` | 64 | Lado (em pixels) para o qual as imagens são redimensionadas |
| `BATCH_SIZE` | 32 | Quantas imagens são processadas antes de cada atualização dos pesos |
| `NUM_EPOCHS` | 10 | Quantas vezes a rede percorre todo o conjunto de treino |
| `LEARNING_RATE` | 1e-3 | Tamanho do passo do otimizador Adam |
| `VAL_RATIO` | 0.2 | Fração das imagens separada para validação |
| `SEED` | 42 | Semente aleatória, para os resultados serem reproduzíveis |

- **`IMAGE_SIZE`:** imagens maiores (128, por exemplo) preservam detalhes finos dos dedos e podem reduzir confusões como W/V e R/U, mas o treino fica bem mais lento e usa mais memória. Imagens menores (32) treinam rápido, mas perdem detalhes. **Importante:** a primeira camada `Linear` foi escrita para 64x64 (`128 * 8 * 8`). Se mudar o tamanho, troque para `128 * (IMAGE_SIZE // 8) * (IMAGE_SIZE // 8)`, senão o código dá erro de dimensão.
- **`BATCH_SIZE`:** lotes maiores (64, 128) aproveitam melhor a GPU e deixam cada época mais rápida, mas usam mais memória. Se aparecer erro de "out of memory", diminua o valor. Lotes menores deixam o treino mais "ruidoso", o que às vezes até ajuda a generalizar.
- **`NUM_EPOCHS`:** mais épocas costumam aumentar a acurácia até a curva estabilizar. Nos gráficos, se a loss de validação começar a **subir** enquanto a de treino continua caindo, é sinal de *overfitting* e não vale a pena treinar mais.
- **`LEARNING_RATE`:** um valor alto demais (1e-2) faz a loss oscilar ou não cair. Um valor baixo demais (1e-5) faz o aprendizado ser muito lento, e 10 épocas não seriam suficientes.
- **`VAL_RATIO`:** separar mais imagens para validação deixa a avaliação mais confiável, mas sobra menos para treinar.
- **Data Augmentation (`train_transform`):** aumentar os ângulos de `RandomRotation` ou a intensidade de `ColorJitter` deixa o modelo mais robusto a fotos variadas, mas dificulta o treino (a acurácia de treino fica ainda mais baixa). Remover essas transformações faz a acurácia de treino subir, com mais risco de *overfitting*.
- **Dropout (0.25 nos blocos e 0.5 no classificador):** valores maiores regularizam mais e explicam por que a acurácia de treino fica abaixo da de validação. Com Dropout 0, o treino converge mais rápido, mas a rede tende a decorar as imagens.

**Como rodar**
```bash
cd "Aprendizado De Maquina/Deep Learning"
pip install -r requirements.txt   # torch, torchvision, kagglehub, scikit-learn, seaborn...
python main.py
```
> **Atenção:** na primeira execução o dataset (cerca de 1 GB) é baixado pelo `kagglehub`. Recomenda-se usar uma **GPU**, porque o treino em CPU é bem mais lento. Para instalar o PyTorch com suporte a CUDA, veja [pytorch.org](https://pytorch.org/get-started/locally/).

Ao final, abrem as curvas de **loss/acurácia** e a **matriz de confusão**. O arquivo `asl_cnn_model.pth` contém os pesos do modelo treinado.

**Exemplos de aula (`Exemplos/`):** CNN e Autoencoder de remoção de ruído no MNIST (PyTorch), classificador Cats vs Dogs com interface em Tkinter (TensorFlow/Keras) e scripts de teste de imagem e desempenho.
