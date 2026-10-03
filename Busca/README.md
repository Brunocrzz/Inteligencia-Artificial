# Busca, CSP e Conhecimento

Esta pasta reúne os projetos da primeira parte da disciplina, sobre **resolução de problemas por busca** e **representação do conhecimento**. Todos têm interface gráfica feita com **Pygame**, para acompanhar a execução dos algoritmos passo a passo.

| Projeto | Problema | Algoritmos | Relatório |
|---|---|---|---|
| [`labirinto/`](./labirinto) | Encontrar a saída de um labirinto | BFS (busca cega) e A* (busca informada) | `Busca Cega e Busca Informada.pdf` |
| [`TSP/`](./TSP) | Problema do Caixeiro Viajante | Hill Climbing e Algoritmo Genético | `Busca Complexa e Alg. Genético.pdf` |
| [`CSP/`](./CSP) | Colorir o mapa do Brasil | Backtracking com MRV e Grau da Variável | `CSP.pdf` |
| [`conhecimento/`](./conhecimento) | Diagnosticar problemas em um computador | Base de conhecimento e Forward Chaining | `Banco de Conhecimento.pdf` |

## Requisitos

```bash
pip install pygame
# ou
pip install pygame-ce
```

Os projetos não usam nenhuma outra biblioteca externa.

---

## Labirinto: busca cega x busca informada

O programa gera labirintos proceduralmente e compara um algoritmo de **busca cega** (BFS) com um de **busca informada** (A*) quanto a tempo de execução, número de nós visitados e tamanho do caminho encontrado.

**Modelagem do problema**
- **Estado inicial:** a célula `S` (Start) do labirinto.
- **Ações:** mover para cima, baixo, esquerda ou direita, desde que o destino não seja parede.
- **Objetivo:** chegar à célula `E` (End).
- **Custo:** cada movimento custa 1.

**Como funciona**
- `maze_generator.py` cria uma matriz cheia de paredes (`#`) e "escava" os caminhos com uma **DFS** de vizinhos embaralhados, o que garante que sempre exista um caminho. Depois, `add_loops(maze, loop_density)` remove algumas paredes para criar ciclos e rotas alternativas. Nos testes, uma densidade de **3% a 5%** gerou os labirintos mais equilibrados.
- `bfs.py` implementa a **Busca em Largura**, que explora em camadas e garante o menor caminho.
- `astar.py` implementa o **A\*** com fila de prioridade (`heapq`) e heurística de **distância Manhattan**.
- `search_utils.py` traz funções auxiliares (`find_symbol`, `is_valid_move`, `get_neighbors`).
- `pygame_visualizer.py` anima a ordem de exploração dos nós e desenha o caminho final.

**Como rodar**
```bash
cd Busca/labirinto
python main.py
```
O terminal mostra o tempo, os nós visitados e o tamanho do caminho dos dois algoritmos, e a janela do Pygame mostra a solução do A\*. Para ver a animação do BFS, descomente o bloco `visualize_maze(bfs_maze, ...)` no final de `main.py`.

**Parâmetros que podem ser alterados**

O labirinto é criado em `main.py` pela chamada:

```python
maze = create_maze_with_loops(30, 30, 3)   # (rows, cols, loop_density)
```

| Parâmetro | Valor atual | O que significa |
|---|---|---|
| `rows` | 30 | Número de células na **vertical**. As paredes também ocupam posições na matriz, então a matriz real tem `2 * rows + 1` linhas (30 vira 61). |
| `cols` | 30 | Número de células na **horizontal**. A matriz real tem `2 * cols + 1` colunas. |
| `loop_density` | 3 | Percentual (de 0 a 100) que controla quantas paredes o programa **tenta** remover para criar atalhos e ciclos. |

- **`rows` e `cols` maiores:** o labirinto fica maior, com mais estados para explorar. O BFS visita muito mais nós, a diferença de tempo e de nós visitados entre BFS e A\* fica mais visível e a animação demora mais. Valores diferentes geram labirintos retangulares (por exemplo, `create_maze_with_loops(20, 50, 3)`).
- **`rows` e `cols` menores:** o labirinto fica pequeno e os dois algoritmos terminam quase iguais, o que dificulta a comparação.
- **Tamanho na tela:** a janela tem no máximo 1200x800 pixels, e o tamanho de cada célula é calculado para o labirinto inteiro caber nela. Quanto maior o labirinto, menores as células. Com algumas centenas de células por lado, cada uma fica com 1 ou 2 pixels e a visualização deixa de ser útil.
- **`loop_density = 0`:** labirinto "perfeito", sem ciclos. Existe exatamente um caminho entre a entrada e a saída, então BFS e A\* encontram a mesma rota e o A\* tem pouca vantagem.
- **`loop_density` entre 3 e 5:** alguns atalhos e caminhos alternativos. Foi a faixa que gerou os labirintos mais equilibrados nos testes.
- **`loop_density` alto (20 ou mais):** muitas paredes somem e o labirinto vira um espaço quase aberto. Nesse cenário a heurística de Manhattan guia bem o A\*, que passa a visitar muito menos nós que o BFS.

O valor de `loop_density` é uma quantidade de **tentativas** (`linhas * colunas * densidade / 100`). Uma parede só é removida se separar duas células livres, então o número real de paredes removidas é menor que o de tentativas. A entrada `S` e a saída `E` são sorteadas nas bordas esquerda e direita, e o labirinto muda a cada execução.

**Conclusão:** em labirintos fechados, com poucos caminhos alternativos, o A\* se comporta de forma parecida com o BFS, porque a heurística tem pouco poder de direcionamento. Em labirintos mais abertos, o A\* explora bem menos estados.

---

## TSP: Hill Climbing x Algoritmo Genético

O **Problema do Caixeiro Viajante** consiste em visitar todas as cidades exatamente uma vez e voltar à origem, percorrendo a menor distância possível. É um problema **NP-difícil**, com cerca de N! rotas possíveis, por isso foram usadas duas abordagens heurísticas.

**Arquivos**
- `city.py`: gera cidades aleatórias (`generate_cities`), calcula a distância euclidiana (`distance`), o custo total de uma rota fechada (`route_distance`) e gera vizinhos trocando duas cidades de posição (`generate_neighbor`, um *swap*).
- `hillClimbing.py`: **Hill Climbing** simples. A partir de uma rota aleatória, gera um vizinho a cada iteração e só aceita se ele for melhor. É rápido, mas pode ficar preso em ótimos locais.
- `geneticAlg.py`: **Algoritmo Genético** com população de rotas, fitness inverso à distância, elitismo, *crossover* por segmento (mantendo a ordem das cidades) e mutação por troca.
- `pygame_visualizer.py`: desenha as cidades e a evolução das rotas.

**Como rodar**
```bash
cd Busca/TSP
python main.py
```
Por padrão, o `main.py` executa o **Hill Climbing**. Para rodar o **Algoritmo Genético**, descomente a chamada `genetic_algorithm(...)` e comente a do `hill_climbing(...)`. Os parâmetros ficam no topo do arquivo:

| Parâmetro | Valor | Usado em | Descrição |
|---|---|---|---|
| `CITIES` | 20 | Ambos | Número de cidades |
| `ITERACOES` | 1000 | Hill Climbing | Quantos vizinhos são testados |
| `POPULATION_SIZE` | 100 | Genético | Quantas rotas existem em cada geração |
| `ELITE_SIZE` | 5 | Genético | Quantas das melhores rotas passam direto para a próxima geração |
| `MUTATION_RATE` | 0.05 | Genético | Probabilidade de cada filho sofrer mutação |
| `GENERATIONS` | 30 | Genético | Quantas gerações são criadas |

**O que acontece ao alterar cada parâmetro**

- **`CITIES`:** as cidades são sorteadas em posições aleatórias dentro da janela de 1200x800. Com mais cidades, o número de rotas possíveis cresce de forma fatorial (20 cidades já dão cerca de 10¹⁸ rotas). O Hill Climbing fica preso em ótimos locais com mais facilidade e o Algoritmo Genético precisa de mais gerações e de uma população maior para chegar a uma boa rota. Com poucas cidades (5 a 8), os dois algoritmos costumam encontrar a rota ótima.
- **`ITERACOES`:** a cada iteração o Hill Climbing troca duas cidades de lugar e só aceita a troca se a rota ficar menor. Mais iterações dão mais chances de melhorar, mas, depois que o algoritmo chega a um ótimo local, as iterações extras não mudam nada. Com poucas iterações (100, por exemplo), a rota final ainda fica bem longe de uma boa solução.
- **`POPULATION_SIZE`:** uma população maior tem mais diversidade de rotas e explora melhor o espaço de soluções, mas cada geração demora mais. Uma população pequena converge rápido para uma rota parecida em todos os indivíduos, que pode ser ruim.
- **`ELITE_SIZE`:** além de serem mantidas, as rotas da elite são as **únicas usadas como pais** dos filhos. Uma elite pequena (2 a 5) gera pressão de seleção forte: a convergência é rápida, mas a diversidade se perde. Uma elite grande mantém mais variedade e converge mais devagar. O valor precisa ser pelo menos 1 e menor que `POPULATION_SIZE`. Com 1, todos os filhos vêm do mesmo pai e só a mutação traz variação.
- **`MUTATION_RATE`:** valor entre 0 e 1 (0.05 = 5%). A mutação troca duas cidades de lugar no filho. Com 0, a população para de evoluir assim que todos os indivíduos ficam parecidos. Com valores muito altos (0.5 ou mais), boas rotas são destruídas o tempo todo e a busca fica quase aleatória.
- **`GENERATIONS`:** 30 gerações é pouco para 20 cidades. Aumentar para 200 ou 500 costuma dar rotas bem menores, e o gráfico de melhoria mostra em que ponto a evolução estabiliza.

As cidades mudam a cada execução. Para comparar os dois algoritmos sobre as mesmas cidades, adicione `random.seed(42)` (ou outro número) logo depois dos imports em `main.py`.

---

## CSP: coloração do mapa do Brasil

Problema de **Satisfação de Restrições**: colorir os **27 estados brasileiros** com **4 cores** (Teorema das Quatro Cores), sem que dois estados vizinhos tenham a mesma cor.

**Modelagem**
- **Variáveis:** os estados.
- **Domínio:** {vermelho, verde, azul, amarelo}.
- **Restrições:** estados adjacentes não podem ter a mesma cor.

**Arquivos**
- `brasilGraph.py`: grafo de adjacência dos estados, posições no mapa e cores.
- `csp.py`: **Backtracking** com `is_consistent()`. Com heurísticas, `select_unassigned_variable()` usa **MRV** (estado com menos cores válidas restantes) e desempata pelo **Grau da Variável** (estado com mais vizinhos). Sem heurísticas, estados e cores são escolhidos aleatoriamente.
- `floodfill.py`: algoritmo **Flood Fill** que encontra os pixels de cada estado no `mapa.png`, parando nas bordas escuras (`is_border()`). Os pixels ficam em memória (`generate_state_pixels()`), o que deixa a pintura rápida.
- `pygame_visualizer.py`: desenha o mapa, as estatísticas e os botões.

**Como rodar**
```bash
cd Busca
python CSP/main.py
```
> **Atenção:** a imagem é carregada com o caminho relativo `"CSP\mapa.png"`, então o script deve ser executado **a partir da pasta `Busca/`**. Esse caminho com `\` funciona no Windows. No Linux/macOS, troque para `os.path.join("CSP", "mapa.png")` em `pygame_visualizer.py`.

Na janela, clique em **"Rodar com Heurísticas"** ou **"Rodar sem Heurísticas"** para comparar o número de **chamadas recursivas** e de **backtracks**. Sem heurísticas, esses números variam muito entre as execuções.

**Parâmetros que podem ser alterados**

- **Cores (`COLORS` em `brasilGraph.py`):** é o domínio de cada variável.
  - **Com 3 cores** o problema **não tem solução**. Goiás faz fronteira com MT, TO, BA, MG e MS, que formam um ciclo de 5 estados vizinhos entre si. Um ciclo ímpar precisa de 3 cores, e Goiás, vizinho de todos eles, precisa de uma quarta. O backtracking testa todas as combinações antes de desistir, e sem heurísticas isso pode demorar bastante.
  - **Com 5 ou mais cores** a solução aparece mais rápido e com menos backtracks, porque sobram mais cores válidas para cada estado. Uma cor nova também precisa ser cadastrada no dicionário `COLOR_MAP` de `pygame_visualizer.py`, que converte o nome da cor em RGB.
- **Velocidade da animação (`csp.py`):** com heurísticas, a tela é redesenhada a cada atribuição, com uma pausa de `pygame.time.delay(30)` milissegundos. Um valor maior deixa a animação mais lenta e fácil de acompanhar, e um menor deixa mais rápida. Sem heurísticas, `get_draw_frequency()` redesenha a tela só a cada N chamadas, e esse N cresce conforme o número de chamadas recursivas aumenta, para a animação não travar quando passa de dezenas de milhares de chamadas.
- **Detecção de bordas (`BORDER_THRESHOLD = 40` em `floodfill.py`):** um pixel é considerado borda quando os três canais R, G e B estão abaixo desse valor. Com um valor maior, mais pixels cinzas viram borda e podem ficar falhas sem pintar perto das divisas. Com um valor menor, as bordas mais claras (suavizadas) deixam de ser reconhecidas e a pintura pode "vazar" para o estado vizinho.

---

## Banco de Conhecimento: sistema especialista

Um **sistema especialista** simples para diagnosticar problemas em computadores. Ele faz perguntas ao usuário, guarda as respostas como **fatos** e aplica **regras de inferência** para chegar a possíveis diagnósticos, cada um com um **nível de confiança**.

**Arquivos**
- `knowledge_base.py`: a base de conhecimento, com as perguntas, as regras (`if` → `then`, `confidence`), os textos dos diagnósticos e os rótulos dos fatos (`fact_labels`). Há regras para fonte, GPU, memória, temperatura, sistema, rede, HD, cooler, armazenamento, desempenho, USB e energia.
- `inference.py`: motor de inferência por **encadeamento progressivo (Forward Chaining)**. Ele percorre as regras e dispara as que têm todas as condições satisfeitas pelos fatos atuais.
- `pygame_visualizer.py`: interface com as perguntas, os fatos conhecidos e os diagnósticos.

**Como rodar**
```bash
cd Busca/conhecimento
python main.py
```
Responda às perguntas pelos botões. Os diagnósticos aparecem ordenados por confiança e podem ser vários ao mesmo tempo. Se nenhuma regra for satisfeita, o resultado é inconclusivo. Use a roda do mouse para rolar a lista.

> Exemplo de regra: se o computador **liga**, **não tem vídeo** e **não emite beep**, o sistema infere um possível **problema na placa de vídeo**.

**Como alterar ou ampliar a base de conhecimento**

Tudo fica em `knowledge_base.py`:

- **Perguntas (`questions`):** cada pergunta é uma tupla `("texto da pergunta", "nome_do_fato")`. Responder **Sim** adiciona o fato `nome_do_fato` e responder **Não** adiciona `not_nome_do_fato`. As perguntas aparecem na ordem da lista.
- **Regras (`rules`):** cada regra é um dicionário como este:

  ```python
  {
      "if": ["liga", "not_video", "not_beep"],   # TODOS os fatos precisam ser verdadeiros
      "then": "problema_gpu",                    # diagnóstico gerado
      "confidence": 85                           # confiança de 0 a 100
  }
  ```

  Quanto mais condições no `if`, mais específica é a regra e menos vezes ela dispara.
- **`confidence`:** serve para **ordenar** os diagnósticos na tela, do mais provável para o menos provável. O valor não é combinado nem recalculado entre regras: se duas regras disparam, as duas aparecem, cada uma com o próprio valor.
- **Novo diagnóstico:** além da regra, adicione o texto exibido em `diagnosis_text` e, se criar um fato novo, o rótulo dele em `fact_labels`.
