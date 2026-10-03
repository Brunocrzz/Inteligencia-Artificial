# Raciocínio sob Incerteza

Esta pasta reúne os projetos sobre **raciocínio probabilístico**, usado quando o agente não observa o mundo por completo e precisa estimar estados ocultos a partir de evidências ruidosas ou incompletas. Os três projetos foram implementados do zero com **NumPy** (ou Python puro) e geram painéis de gráficos com **Matplotlib**.

| Projeto | Problema | Técnica | Relatório |
|---|---|---|---|
| [`Redes Bayesianas/`](./Redes%20Bayesianas) | Decisão de um NPC de combate | Rede Bayesiana e inferência por enumeração | `Redes Bayesianas.pdf` |
| [`Modelo Oculto de Markov/`](./Modelo%20Oculto%20de%20Markov) | Reconhecer o perfil de um jogador | HMM e algoritmo Forward (filtragem) | `Modelo Oculto de Markov.pdf` |
| [`Filtro de Kalman/`](./Filtro%20de%20Kalman) | Rastrear um veículo com GPS ruidoso | Filtro de Kalman linear 6D | `Filtro de Kalman.pdf` |

## Como rodar

Todos os projetos seguem o mesmo padrão:

```bash
cd "incerteza/<nome do projeto>"
pip install -r requirements.txt
python main.py
```

O terminal mostra os valores calculados e uma janela do Matplotlib abre com os gráficos.

---

## Redes Bayesianas: NPC de combate

O projeto modela como um **NPC de jogo** decide o que fazer sob incerteza. A percepção do ambiente (há um inimigo visível?) e o estado interno (a vida está crítica?) influenciam o **estado de alerta** do NPC, que por sua vez define a probabilidade de suas ações: **Atacar**, **Fugir** ou **Pedir Reforço**. A topologia segue o modelo clássico de Russell & Norvig.

**Estrutura da rede (DAG com 6 nós)**

```
Inimigo_Visivel    Vida_Critica
          \          /
          Estado_Alerta
         /      |      \
    Atacar    Fugir    Reforço
```

A distribuição conjunta é fatorada como:

`P(IV, VC, EA, A, F, R) = P(IV) · P(VC) · P(EA | IV, VC) · P(A | EA) · P(F | EA) · P(R | EA)`

**Funções principais**
- `obter_probabilidade_condicional`: consulta as **CPTs** (tabelas de probabilidade condicional) a partir dos valores dos pais.
- `enumerate_all`: **inferência por enumeração**, recursiva, na ordem topológica, somando (marginalizando) as variáveis ocultas.
- `enumeration_ask`: faz a consulta e **normaliza** (α) o resultado.

**Cenários testados (6):** inferência preditiva (causa → efeito) e diagnóstica (efeito → causa). Dois destaques:
- **Sem evidências (a priori):** P(Alerta Alto) = **21,6%** e P(Alerta Baixo) = **78,4%**.
- **Inimigo visível e o NPC não atacou:** a probabilidade de a **vida estar crítica** sobe para **81,2%**. A rede "deduz" que a hesitação indica que o agente está fraco.

**Parâmetros que podem ser alterados**

- **Probabilidades a priori:** `Inimigo_Visivel` (40% de "Sim") e `Vida_Critica` (30% de "Sim"). São as chances de cada situação antes de qualquer evidência. Aumentar a chance de inimigo visível, por exemplo, faz o NPC ficar em alerta alto com mais frequência no cenário 1, que não tem evidências.
- **CPTs (tabelas `cpt` em `rede_bayesiana`):** cada linha diz a probabilidade do nó para uma combinação dos pais, e os valores de cada linha precisam **somar 1**. Alguns exemplos do que muda:
  - Em `Estado_Alerta`, a linha `("Sim", "Nao")` vale 0,80 de alerta alto. Aumentar esse valor deixa o NPC mais reativo quando vê um inimigo, mesmo com a vida cheia.
  - Em `Atacar`, a diferença entre `("Alto",)` (0,85) e `("Baixo",)` (0,15) define o quanto "atacar ou não" revela sobre o estado de alerta. Quanto mais distantes esses valores, mais forte é a conclusão do cenário 4 (não atacou → vida crítica). Se os dois valores forem iguais, observar o ataque não traz nenhuma informação.
- **Cenários (no final de `main.py`):** cada consulta é uma chamada do tipo

  ```python
  ev = {"Inimigo_Visivel": "Sim", "Atacar": "Nao"}   # evidências observadas
  r = enumeration_ask("Vida_Critica", ev)            # variável consultada
  ```

  Os valores possíveis são `"Sim"`/`"Nao"` para todos os nós, exceto `Estado_Alerta`, que usa `"Alto"`/`"Baixo"`. Quanto mais evidências, menos variáveis ocultas sobram para somar e mais "certa" fica a resposta. Para o novo cenário aparecer no gráfico, adicione-o também à lista que alimenta o `dashboard`.

**Dependências:** `matplotlib`

---

## Modelo Oculto de Markov: perfil do jogador

O objetivo é estimar, em tempo real, o **perfil estratégico oculto** de um adversário a partir das ações que ele faz.

- **Estados ocultos:** Agressivo, Defensivo, Econômico.
- **Observações:** Comprou Arma, Escondeu, Avançou, Curou.

A implementação usa a forma matricial do **algoritmo Forward** de um HMM de primeira ordem:

`f(t+1) = α · O(t+1) · Tᵀ · f(t)`

- `f(0)`: distribuição inicial (vetor 3×1).
- `T`: matriz de transição 3×3 entre os perfis.
- `O(t+1)`: matriz diagonal 3×3 com P(observação | estado) da ação observada.
- `α`: normalização para a soma voltar a 1.

**Cenários:** (1) predição hostil, (2) predição conservadora e (3) transição dinâmica de perfil. No cenário 3, com a sequência `[Escondeu, Escondeu, Comprou Arma, Avançou, Avançou]`:

| t | Ação | Agressivo | Defensivo | Econômico |
|---|---|---|---|---|
| 0 | — | 30,0% | 30,0% | 40,0% |
| 1 | Escondeu | 6,7% | 53,7% | 39,6% |
| 2 | Escondeu | 3,5% | 60,4% | 36,1% |
| 3 | Comprou Arma | 48,0% | 30,2% | 21,8% |
| 4 | Avançou | 72,3% | 11,4% | 16,3% |
| 5 | Avançou | 81,3% | 7,4% | 11,3% |

O modelo percebe a mudança de estratégia logo que ela acontece. Uma limitação é que as matrizes são **estacionárias**: se o estilo do jogador mudar ao longo de uma partida longa, elas podem deixar de representá-lo bem.

**Parâmetros que podem ser alterados**

- **Distribuição inicial (`pi`):** o "chute" antes da primeira ação (30% Agressivo, 30% Defensivo, 40% Econômico). Os valores precisam somar 1. Ela influencia principalmente os primeiros passos, e depois de algumas observações o efeito praticamente desaparece. O dicionário `inicio` é só descritivo, e o vetor que o algoritmo realmente usa é o `pi`.
- **Matriz de transição (`T`):** a linha é o perfil atual e a coluna é o próximo perfil, e cada linha soma 1. A diagonal (0,7, 0,65 e 0,6) é a chance de o jogador **manter** o perfil.
  - **Diagonal maior (perto de 1):** o modelo considera os perfis "estáveis" e demora mais para aceitar uma mudança de estratégia. A curva do cenário 3 sobe mais devagar depois que o jogador compra a arma.
  - **Diagonal menor:** o modelo acha que o jogador troca de perfil o tempo todo, reage mais rápido a cada ação e as curvas oscilam mais.
- **Matriz de emissão (`B`):** a linha é o perfil e a coluna é a ação (Comprou Arma, Escondeu, Avançou, Curou), e cada linha soma 1. Quanto mais diferentes as linhas, mais cada ação "denuncia" o perfil. Por exemplo, o Agressivo avança com 50% de chance e se esconde com só 5%. Se duas linhas forem iguais, o modelo não consegue diferenciar esses dois perfis.
- **Sequências observadas (`seq_1`, `seq_2`, `seq_3`):** listas de índices das ações, com `0` = Comprou Arma, `1` = Escondeu, `2` = Avançou e `3` = Curou. Sequências mais longas mostram melhor a convergência, e trocar a ordem das ações mostra como o modelo reage a mudanças de comportamento.

**Dependências:** `numpy`, `matplotlib`

---

## Filtro de Kalman: rastreamento de um veículo

O filtro estima a trajetória de um veículo que faz curvas em **"S"**, combinando um **GPS de baixa precisão** com o modelo físico do **MRUV** nos dois eixos. Além de suavizar a posição, ele estima variáveis que **não são medidas**, como a **velocidade** e a **aceleração**.

**Modelagem**
- **Vetor de estado 6×1:** `[x, vx, ax, y, vy, ay]`.
- **Matriz de transição A (6×6):** em blocos, com `s = s0 + v·Δt + ½·a·Δt²` e `v = v0 + a·Δt` para cada eixo.
- **Matriz de observação H (2×6):** o sensor mede apenas a posição (x, y).
- **Inicialização:** posição tirada da primeira leitura, velocidade e aceleração em zero, `P0 = 10·I`.

**Ciclo do filtro**
1. **Predição:** `x̂ = A·x` e `P = A·P·Aᵀ + Q`
2. **Ganho de Kalman:** `K = P·Hᵀ·(H·P·Hᵀ + R)⁻¹`
3. **Correção:** `x = x̂ + K·(z − H·x̂)` e `P = (I − K·H)·P`

**Parâmetros da simulação**

| Parâmetro | Valor |
|---|---|
| Δt | 0,1 s |
| Passos | 200 (20 s) |
| Variância do GPS (R) | 8,0 (cerca de ±2,8 m) |
| Ruído de processo (Q) | 0,2 |

**Resultados:** o filtro remove o ruído de alta frequência do GPS e reconstrói uma curva suave, próxima da trajetória real. Ele também recupera a **aceleração no eixo X** sem medi-la diretamente. Depois de um pico inicial de convergência (o filtro começa em repouso com o veículo já em movimento), passa a acompanhar a amplitude e a frequência da cossenoide real.

**Parâmetros que podem ser alterados**

Ficam no início de `main.py`:

| Variável | Valor | O que significa |
|---|---|---|
| `dt` | 0.1 | Intervalo entre duas leituras do GPS, em segundos |
| `passos` | 200 | Quantidade de leituras. A duração total é `passos * dt` (20 s) |
| `ruido_sensor_gps` | 8.0 | Variância do erro do GPS (matriz R) |
| `ruido_processo_fisico` | 0.2 | Incerteza do modelo físico (matriz Q) |
| `P = np.eye(6) * 10.0` | 10 | Incerteza inicial da estimativa (P0) |
| `vx_real, vy_real` | 30, 0 | Velocidade inicial do veículo simulado |
| `2.0 * cos(t)`, `5.0 * sin(t)` | — | Aceleração real nos eixos X e Y, que gera as curvas em "S" |

- **`ruido_sensor_gps` (R):** o mesmo valor é usado para **gerar o ruído do GPS simulado** e como a matriz **R do filtro**. Aumentar deixa os pontos do GPS mais espalhados e faz o filtro confiar menos neles: a curva estimada continua suave, mas reage mais devagar às curvas. Diminuir aproxima o GPS da trajetória real, e o filtro passa a seguir as medições mais de perto. Para testar um filtro "mal calibrado", troque `R = np.eye(2) * ruido_sensor_gps` por um valor diferente do usado na simulação.
- **`ruido_processo_fisico` (Q):** representa o quanto o filtro desconfia do próprio modelo de MRUV. Com Q **maior**, o filtro confia mais no GPS e a estimativa fica mais ruidosa, mas acompanha melhor as mudanças bruscas. Com Q **menor**, a estimativa fica muito suave, mas pode ficar atrasada nas curvas, porque o modelo assume aceleração constante entre um passo e outro.
- **`dt`:** com intervalos maiores, o GPS lê menos vezes por segundo e o modelo precisa prever mais longe entre uma leitura e outra, o que aumenta o erro nas curvas. Com intervalos menores, há mais leituras e o rastreamento melhora.
- **`passos`:** só muda a duração da simulação. Mais passos mostram mais curvas do "S".
- **P0:** um valor alto diz ao filtro que a estimativa inicial é pouco confiável, então ele se corrige rápido com as primeiras leituras. Um valor baixo faz o filtro "acreditar" no estado inicial (que começa com velocidade e aceleração zero), e o pico de convergência no começo do gráfico de aceleração fica maior e mais longo.
- **`np.random.seed(42)`:** fixa o ruído para que toda execução gere o mesmo resultado. Trocar o número gera outro padrão de ruído, e remover a linha gera um ruído diferente a cada execução.

**Dependências:** `numpy`, `matplotlib`
