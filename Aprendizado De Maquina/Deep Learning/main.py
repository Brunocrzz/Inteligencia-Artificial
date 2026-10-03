import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import kagglehub


# Faz o download do dataset público real de LIBRAS (Alfabeto A-Z em pastas)
path = kagglehub.dataset_download("grassknoted/asl-alphabet")
train_path = os.path.join(path, "asl_alphabet_train", "asl_alphabet_train")

# Configuração de Hiperparâmetros estruturais
IMAGE_SIZE = 64  # Redimensionamento ideal para processamento em CPU/GPU comum
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
VAL_RATIO = 0.2  # 20% para validação isolada
SEED = 42

# Configuração de Semente Aleatória para Reprodutibilidade
torch.manual_seed(SEED)
np.random.seed(SEED)

# Configuração de Dispositivo (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Transformações de Treino: injeta perturbações para evitar Overfitting
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),          # Inversão sutil da mão
    transforms.RandomRotation(degrees=15),           # Pequenas inclinações do gesto
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Variações de iluminação da foto
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Escala ImageNet
])

# Transformações de Validação/Teste: sem perturbações aleatórias
eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Carrega o dataset bruto mapeando as pastas nativamente
full_dataset = datasets.ImageFolder(root=train_path)
num_classes = len(full_dataset.classes)
class_names = full_dataset.classes

# Divisão de Treino e Validação
n_val = int(len(full_dataset) * VAL_RATIO)
n_train = len(full_dataset) - n_val

train_set, val_set = random_split(full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))

# Sobrescreve os transforms específicos para cada partição de dados
train_set.dataset.transform = train_transform
val_set.dataset.transform = eval_transform

# Criação dos DataLoaders operacionais
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# Arquitetura CNN 
class AslCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # Bloco Convolucional 1: Entrada (3, 64, 64) -> Saída (32, 32, 32)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        
        # Bloco Convolucional 2: Entrada (32, 32, 32) -> Saída (64, 16, 16)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        
        # Bloco Convolucional 3: Entrada (64, 16, 16) -> Saída (128, 8, 8)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        
        # Classificação Final: 128 canais * 8 altura * 8 largura = 8192 entradas
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    # Forward Pass
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        logits = self.classifier(x)
        return logits

# Instanciação do modelo e envio obrigatório para o hardware ativo
model = AslCNN(num_classes=num_classes).to(device)

# Configuração de Critério de Perda e Otimizador
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Histórico de Métricas para Visualização
history_train_loss, history_val_loss = [], []
history_train_acc, history_val_acc = [], []

print(f"\nIniciando Treinamento do Modelo CNN para LIBRAS com {NUM_EPOCHS} épocas...")
# Treinamento do Modelo com Validação Isolada
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss, correct_train, total_train = 0.0, 0, 0
    
    for x_batch, y_batch in train_loader:
        # Envia os dados do lote atual explicitamente para a GPU/CPU
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * x_batch.size(0)
        _, preds = torch.max(outputs, 1)
        correct_train += (preds == y_batch).sum().item()
        total_train += y_batch.size(0)
        
    epoch_train_loss = running_loss / total_train
    epoch_train_acc = correct_train / total_train
    
    # Fase de Validação Isolada
    model.eval()
    val_loss, correct_val, total_val = 0.0, 0, 0
    all_preds, all_labels = [], []
    
    with torch.inference_mode():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            outputs_val = model(x_val)
            loss_v = criterion(outputs_val, y_val)
            
            val_loss += loss_v.item() * x_val.size(0)
            _, preds_v = torch.max(outputs_val, 1)
            correct_val += (preds_v == y_val).sum().item()
            total_val += y_val.size(0)
            
            all_preds.extend(preds_v.cpu().numpy())
            all_labels.extend(y_val.cpu().numpy())
            
    epoch_val_loss = val_loss / total_val
    epoch_val_acc = correct_val / total_val
    
    history_train_loss.append(epoch_train_loss)
    history_val_loss.append(epoch_val_loss)
    history_train_acc.append(epoch_train_acc)
    history_val_acc.append(epoch_val_acc)
    
    print(f" -> Época [{epoch+1:02d}/{NUM_EPOCHS:02d}] | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%")


# Plotagem dos graficos
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: Evolução da Loss (Entropia Cruzada)
axes[0].plot(history_train_loss, label='Treinamento', color='royalblue', linewidth=2)
axes[0].plot(history_val_loss, label='Validação', color='darkorange', linewidth=2, linestyle='--')
axes[0].set_title('Curvas de Custo Operacional (Loss Decay)', fontweight='bold')
axes[0].set_xlabel('Época de Treino')
axes[0].set_ylabel('Loss (Cross Entropy)')
axes[0].legend()

# Gráfico 2: Evolução da Acurácia
axes[1].plot(history_train_acc, label='Treinamento', color='royalblue', linewidth=2)
axes[1].plot(history_val_acc, label='Validação', color='darkorange', linewidth=2, linestyle='--')
axes[1].set_title('Evolução da Taxa de Acerto (Accuracy)', fontweight='bold')
axes[1].set_xlabel('Época de Treino')
axes[1].set_ylabel('Acurácia (%)')
axes[1].legend()

plt.tight_layout()
plt.show()

# Gráfico 3: Matriz de Confusão do Alfabeto
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
cmd.plot(cmap="Blues", values_format='d', ax=plt.gca())
plt.title("Matriz de Confusão Final - Classificação de LIBRAS", fontweight='bold', pad=15)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()