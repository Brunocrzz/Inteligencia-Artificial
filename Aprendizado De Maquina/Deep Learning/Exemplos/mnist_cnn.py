'''
Exemplo de CNN utilizando o PyTorch explicado passo a passo

Prof. Fabiano A. Soares
'''

# -*- coding: utf-8 -*-

## ---------------------------------------------------------------------------------------- ##
# 0) Importando Bibliotecas
'''
- numpy: operações numéricas gerais.
- matplotlib: para visualização de imagens e gráficos de treino.
- torch, nn, DataLoader: núcleo do PyTorch para tensores, redes neurais e carregamento de dados.
- torchvision: fornece o dataset MNIST e transformações de imagem.
- sklearn.metrics: para calcular e plotar a matriz de confusão.
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# garante que o diretório para salvar figuras existe
os.makedirs("output", exist_ok=True)

## ---------------------------------------------------------------------------------------- ##

# 1) hiperparâmetros e device
'''
- seed: garante reprodutibilidade (mesmo treino, mesmos resultados).
- device: escolhe automaticamente GPU (cuda) se disponível, senão CPU.
- batch_size: número de exemplos processados em cada atualização de gradiente.
- num_epochs: quantas vezes percorremos completamente o conjunto de treino.
- learning_rate: passo da descida de gradiente.
- val_ratio: fração do conjunto de treino usada para validação.
'''
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 128
num_epochs = 5
learning_rate = 1e-3
val_ratio = 0.1

# 2) Dataset, split e exemplo de imagem

# carrega os dados brutos para calcular estatísticas de média e desvio-padrão
raw_train = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)
'''
all_pixels = torch.cat([img.view(-1) for img, _ in raw_train]):
- Percorre todo o conjunto raw_train, imagem por imagem.
- Para cada imagem (tensor 1x28x28), img.view(-1) "achata" em um vetor 1D
  com todos os 784 pixels daquela imagem.
- A list comprehension [img.view(-1) for img, _ in raw_train] cria uma lista
  desses vetores (um por imagem).
- torch.cat(...) concatena todos esses vetores em um único grande tensor 1D,
  contendo TODOS os pixels de TODAS as imagens de treino.
- Assim, quando calculamos mean = all_pixels.mean() e std = all_pixels.std(),
  estamos obtendo a média e o desvio-padrão globais dos pixels do MNIST,
  que usamos para normalizar o dataset.
'''
all_pixels = torch.cat([img.view(-1) for img, _ in raw_train])
mean = all_pixels.mean().item()
std = all_pixels.std().item()
print("MNIST mean:", mean, "std:", std)

# transforma cada imagem em tensor e normaliza com média/desvio-padrão do MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])

"""
Se preferir, pode usar valores de média e desvio-padrão do MNIST pré-calculados,
usados em vários exemplos da documentação/comunidade, como:

transforms.Normalize((0.1307,), (0.3081,))
"""

# carrega o MNIST já com a transformação definida
train_full = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)
test_set = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

# separa o conjunto de treino em treino + validação
n_val = int(len(train_full) * val_ratio)
n_train = len(train_full) - n_val
train_set, val_set = random_split(
    train_full,
    [n_train, n_val],
    generator=torch.Generator().manual_seed(seed)
)

# DataLoaders: fazem o batching e o embaralhamento (shuffle) dos dados
# OBS: num_workers=0 para evitar problemas de multiprocessing no Windows/Spyder.
num_workers = 0

train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available()
)
val_loader = DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available()
)
test_loader = DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available()
)

# exemplo de imagem para visualização
example_img, example_label = train_full[0]
plt.figure(figsize=(3, 3))
plt.imshow(example_img.squeeze(), cmap="gray")
plt.title(f"Classe: {example_label}")
plt.axis("off")
plt.tight_layout()
plt.savefig("output/mnist_exemplo.png", dpi=150)
plt.show()        # mostra no Spyder
plt.close()

## ---------------------------------------------------------------------------------------- ##

# 3) Modelo CNN
class CNNMnist(nn.Module):
    def __init__(self):
        super().__init__()

        # Bloco de extração de características (convoluções + pooling + dropout)
        self.features = nn.Sequential(
            # Camada de convolução:
            # - 1 canal de entrada (imagem em tons de cinza)
            # - 32 mapas de características de saída
            # - kernel_size=3 -> filtro 3x3 varre a imagem
            # - padding=1 -> acrescenta 1 pixel de borda com zeros em torno da figura
            nn.Conv2d(
                in_channels=1,   # número de canais da imagem de entrada (MNIST: 1 canal)
                out_channels=32, # número de filtros / mapas de características
                kernel_size=3,   # tamanho do filtro: aqui 3x3 pixels
                padding=1        # adiciona 1 pixel de zeros em cada borda
                                 # para manter largura/altura após a convolução
            ),
            nn.ReLU(),          # função de ativação não linear
            nn.MaxPool2d(2),    # redução de dimensão (pooling) com janela 2x2
            nn.Dropout(0.25),   # regularização por dropout desliga aleatóriamente 25% dos neurônios

            # segunda convolução: aumenta número de canais para 64
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        # Bloco de classificação (camadas totalmente conectadas)
        
        '''
        Dimensionamento das features antes da camada totalmente conectada:

        - Entrada MNIST: cada imagem começa com forma (1, 28, 28)
          (1 canal, 28x28 pixels).

        - Após a primeira convolução (Conv2d(1, 32, kernel_size=3, padding=1))
          seguida de MaxPool2d(2):
          * a convolução com padding=1 mantém altura/largura em 28x28;
          * o MaxPool2d(2) reduz para 14x14;
          * resultado: tensor com forma (32, 14, 14).

        - Após a segunda convolução (Conv2d(32, 64, kernel_size=3, padding=1))
          seguida de outro MaxPool2d(2):
          * a convolução mantém altura/largura em 14x14;
          * o MaxPool2d(2) reduz para 7x7;
          * resultado final do bloco convolucional: (64, 7, 7).

        - Isso significa que, para cada imagem, antes de entrar na parte
          totalmente conectada, temos 64 canais, cada um com um mapa 7x7.

        - A camada nn.Flatten() transforma esse tensor (64, 7, 7) em um vetor
          de tamanho 64 * 7 * 7, e é exatamente por isso que usamos
          64 * 7 * 7 como número de entradas (in_features) da primeira
          camada nn.Linear.
        '''
        
        self.classifier = nn.Sequential(
            nn.Flatten(),             # transforma (batch, canais, H, W) em (batch, features)

            nn.Linear(
                64 * 7 * 7,           # número de entradas da camada totalmente conectada
                                      # 64 canais 7 × linhas × 7 colunas 
                128                   # número de neurônios de saída
            ),

            nn.ReLU(),                # função de ativação ReLU

            nn.Dropout(0.5),          # desativa aleatoriamente 50% dos neurônios (regularização)

            nn.Linear(
                128,
                10                    # última camada: 10 neurônios (classes 0..9)
            )
        )
        
    '''
    Em PyTorch, quando criamos uma rede como uma subclasse de nn.Module,
    precisamos sobrescrever (override) o método forward(self, x).

    - O nn.Module já define toda a infraestrutura de módulo: registro de
      parâmetros, envio para CPU/GPU com .to(device), salvamento/carregamento
      de estados etc.

    - O que nós definimos no forward é o "caminho direto" dos dados:
      como o tensor de entrada x passa pelas camadas (features, classifier, ...)
      até gerar a saída.

    - É o forward que será chamado internamente quando usamos model(x),
      tanto no treino quanto na inferência.
    '''
    
    def forward(self, x):
        # primeiro, extrai características com as convoluções/pooling
        x = self.features(x)
        # depois, classifica com as camadas totalmente conectadas
        x = self.classifier(x)
        # retorna logits (pontuações) para cada classe
        return x


# instancia o modelo e envia para o device (CPU/GPU)
model = CNNMnist().to(device)

## ---------------------------------------------------------------------------------------- ##

# 4) Loss, otimizador e acurácia
'''
 Usamos CrossEntropyLoss porque:
 - O problema é de classificação multiclasse (10 dígitos: 0 a 9).
 - A rede retorna um vetor de logits de tamanho 10 para cada imagem
   (saída da camada Linear(128, 10)), sem softmax explícito na última camada.
 - CrossEntropyLoss combina internamente LogSoftmax + entropia cruzada,
   sendo a escolha padrão para "uma classe correta por exemplo"
   com rótulos inteiros (0..9) em PyTorch.
'''
criterion = nn.CrossEntropyLoss()
'''
 Usamos o otimizador Adam porque:
 - É um método de descida de gradiente adaptativo, que ajusta a taxa de
   aprendizagem para cada parâmetro com base em momentos do gradiente.
 - Em prática, costuma convergir bem e de forma estável para redes
   como esta CNN em problemas de visão (como MNIST), sem exigir
   muito ajuste fino da taxa de aprendizado.
 - É uma boa escolha "default" para exemplos didáticos de deep learning.
'''
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)


def batch_accuracy(logits, y):
    # logits: saída da rede antes do softmax, com forma (batch_size, num_classes)
    # y: rótulos verdadeiros da classe, como inteiros no intervalo [0, num_classes-1]

    # torch.argmax(logits, dim=1):
    # - para cada exemplo no batch, escolhe o índice da classe com maior logit
    # - isso corresponde à "classe prevista" pela rede (predição discreta)
    preds = torch.argmax(logits, dim=1)
    
    # (preds == y):
    # - cria um tensor booleano indicando se a predição está correta (True) ou não (False)
    # .float():
    # - converte True/False para 1.0/0.0
    # .mean():
    # - calcula a média desses valores no batch (fração de acertos)
    # .item():
    # - converte o escalar do tensor para um número Python
    return (preds == y).float().mean().item()

## ---------------------------------------------------------------------------------------- ##
# 5) Training loop
train_losses, val_losses, train_accs, val_accs = [], [], [], []

for epoch in range(num_epochs):
    # modo treinamento: ativa comportamentos de treino (Dropout, BatchNorm, etc.)
    model.train()
    tloss, tacc, n_batches = 0.0, 0.0, 0

    # Barra de progresso textual simples:
    # Vamos imprimir a cada N batches a fração concluída.
    num_batches = len(train_loader)
    print(f"\n=== Época {epoch+1}/{num_epochs} (treino) ===")
    for i, (x, y) in enumerate(train_loader, start=1):
        x, y = x.to(device), y.to(device)
        '''
         optimizer.zero_grad():
         - Zera (coloca em 0) todos os gradientes acumulados nos parâmetros
           do modelo.
         - Por padrão, o PyTorch ACUMULA gradientes a cada chamada de backward,
           ou seja, gradientes de vários batches se somam.
         - Em um loop de treino típico, queremos que cada atualização de pesos
           use APENAS o gradiente do batch atual.
         - Portanto, antes de calcular o gradiente de um novo batch
           (loss.backward()), zeramos os gradientes antigos com zero_grad().
        '''
        optimizer.zero_grad() # zera gradientes antigos
        out = model(x) # Passa a entrada X pela rede (forward: produz logits)
        loss = criterion(out, y) # Calcula a função loss comparando a saída da rede com o alvo
        loss.backward() # Executa o backpropagation (computa gradientes em cada parâmetro)
        optimizer.step() # usa esses gradientes para atualizar os parâmetros

        batch_acc = batch_accuracy(out, y)  # calcula acurácia deste batch
        tloss += loss.item() # acumula o valor de loss do batch
        tacc += batch_acc # acumula a acurácia do batch
        n_batches += 1 # soma 1 no número de batches explorados

        # atualiza barra simples a cada 10 batches (para não inundar o console)
        if i % 10 == 0 or i == num_batches:
            frac = i / num_batches
            percent = int(frac * 100)
            print(
                f"Batch {i}/{num_batches} "
                f"({percent:3d}%) - loss {loss.item():.4f} acc {batch_acc:.4f}",
                end="\r"
            )

    print()  # quebra de linha após a barra textual
    train_losses.append(tloss / n_batches)
    train_accs.append(tacc / n_batches)
    # modo avaliação: desativa comportamentos de treino em certas camadas
    model.eval()
    vloss, vacc, vn = 0.0, 0.0, 0
    # Durante validação/teste podemos usar:
    # - torch.no_grad(): desativa gradientes, mais flexível.
    # - torch.inference_mode(): desativa gradientes e parte do bookkeeping interno,
    #   sendo mais rápido/leve, porém mais restrito.
    with torch.inference_mode():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            vloss += loss.item()
            vacc += batch_accuracy(out, y)
            vn += 1

    val_losses.append(vloss / vn)
    val_accs.append(vacc / vn)

    print(
        f"Epoch {epoch+1}/{num_epochs} | "
        f"train loss {train_losses[-1]:.4f} acc {train_accs[-1]:.4f} | "
        f"val loss {val_losses[-1]:.4f} acc {val_accs[-1]:.4f}"
    )

# curvas de treinamento
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Treino")
plt.plot(val_losses, label="Validação")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label="Treino")
plt.plot(val_accs, label="Validação")
plt.xlabel("Época")
plt.ylabel("Acurácia")
plt.legend()

plt.tight_layout()
plt.savefig("output/curvas_treinamento.png", dpi=150)
plt.show()        # mostra as curvas no Spyder
plt.close()

## ---------------------------------------------------------------------------------------- ##

# 6) Teste, matriz de confusão e exemplo final

model.eval()
test_loss, test_acc, nt = 0.0, 0.0, 0
all_true, all_pred = [], []
example_saved = False

with torch.inference_mode():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        test_loss += loss.item()
        test_acc += batch_accuracy(out, y)
        nt += 1

        pred = torch.argmax(out, dim=1)  # obtém a classe prevista (índice do maior logit)
        all_true.extend(y.cpu().numpy().tolist()) # adiciona os rótulos verdadeiros deste batch
        all_pred.extend(pred.cpu().numpy().tolist()) # adiciona as predições deste batch
                                                     # (para depois montar a matriz de confusão)                                             
        '''
        .cpu(): garante que o tensor está na CPU.
        A maioria das funções externas (como sklearn.metrics) espera arrays na CPU;
        por isso, movemos os tensores da GPU para CPU antes de usar numpy() e montar
        a matriz de confusão. Assim evitamos erro
        '''

        if not example_saved:
            img = x[0].cpu().squeeze().numpy()
            y_true = y[0].cpu().item()
            y_pred = pred[0].cpu().item()

            plt.figure(figsize=(3, 3))
            plt.imshow(img, cmap="gray")
            plt.title(f"Real: {y_true} | CNN: {y_pred}")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig("output/exemplo_predicao.png", dpi=150)
            plt.show()        # mostra um exemplo de predição
            plt.close()
            example_saved = True

test_loss /= nt
test_acc /= nt
print(f"\nTeste - loss: {test_loss:.4f} | acc: {test_acc:.4f}")

cm = confusion_matrix(all_true, all_pred)
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=list(range(10))
).plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Matriz de confusão - teste")
plt.tight_layout()
plt.savefig("output/matriz_confusao.png", dpi=150)
plt.show()        # mostra a matriz de confusão
plt.close()