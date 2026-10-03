# =============================================================================
# AUTOENCODER DEFINITIVO
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

print("Autoencoder")
print("="*60)

# 1. DADOS
digits = load_digits()
X = digits.data.astype(np.float32)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. AUTOENCODER SIMPLES 
class SimpleAutoencoder:
    def __init__(self, input_size=64):
        np.random.seed(42)  # Reprodutível
        self.W_enc1 = np.random.randn(input_size, 32) * 0.01
        self.b_enc1 = np.zeros(32)
        self.W_enc2 = np.random.randn(32, 16) * 0.01
        self.b_enc2 = np.zeros(16)
        self.W_dec1 = np.random.randn(16, 32) * 0.01
        self.b_dec1 = np.zeros(32)
        self.W_dec2 = np.random.randn(32, input_size) * 0.01
        self.b_dec2 = np.zeros(input_size)
    
    def relu(self, z): 
        return np.maximum(0, z)
    
    def relu_deriv(self, z): 
        return (z > 0).astype(float)
    
    def sigmoid(self, z): 
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def forward(self, x):
        # ENCODER - SALVAR TODAS variáveis
        h1 = self.relu(x @ self.W_enc1 + self.b_enc1)
        z = np.tanh(h1 @ self.W_enc2 + self.b_enc2)
        
        # DECODER - SALVAR TODAS variáveis  
        h2 = self.relu(z @ self.W_dec1 + self.b_dec1)
        x_recon = self.sigmoid(h2 @ self.W_dec2 + self.b_dec2)
        
        return x_recon, z, (h1, z, h2)  # Cache para backprop
    
    def train_step(self, x, lr=0.001):
        x_recon, z, cache = self.forward(x)
        h1, z, h2 = cache
        
        loss = np.mean((x - x_recon)**2)
        
        # BACKPROP
        d_output = 2 * (x_recon - x) * x_recon * (1 - x_recon)
        dW_dec2 = h2.T @ d_output
        db_dec2 = np.sum(d_output, axis=0)
        
        d_h2 = d_output @ self.W_dec2.T
        d_hidden_dec = d_h2 * self.relu_deriv(h2)
        dW_dec1 = z.T @ d_hidden_dec
        db_dec1 = np.sum(d_hidden_dec, axis=0)
        
        d_z = d_hidden_dec @ self.W_dec1.T
        d_tanh = d_z * (1 - z**2)
        dW_enc2 = h1.T @ d_tanh
        db_enc2 = np.sum(d_tanh, axis=0)
        
        d_h1 = d_tanh @ self.W_enc2.T
        d_hidden_enc = d_h1 * self.relu_deriv(h1)
        dW_enc1 = x.T @ d_hidden_enc
        db_enc1 = np.sum(d_hidden_enc, axis=0)
        
        # ATUALIZAR PESOS
        self.W_dec2 -= lr * dW_dec2
        self.b_dec2 -= lr * db_dec2
        self.W_dec1 -= lr * dW_dec1
        self.b_dec1 -= lr * db_dec1
        self.W_enc2 -= lr * dW_enc2
        self.b_enc2 -= lr * db_enc2
        self.W_enc1 -= lr * dW_enc1
        self.b_enc1 -= lr * db_enc1
        
        return loss

# 3. TREINAMENTO
print("TREINANDO (200 epochs)...")
ae = SimpleAutoencoder()
losses = []

for epoch in range(200):
    loss = ae.train_step(X_scaled, lr=0.001)
    losses.append(loss)
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

print("Treinamento concluído!")

# 4. PLOTS PERFEITOS
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
fig.suptitle('ORIGINAL vs RECONSTRUÍDO', fontsize=16, fontweight='bold')

for i in range(8):
    # ORIGINAL
    orig = X[i].reshape(8, 8)
    axes[0, i].imshow(orig, cmap='gray')
    axes[0, i].set_title(f'Orig {i+1}', color='blue', fontsize=12)
    axes[0, i].axis('off')
    
    # RECONSTRUIDO
    x_recon, _, _ = ae.forward(X_scaled[i:i+1])
    recon_img = scaler.inverse_transform(x_recon.reshape(1, -1)).reshape(8, 8)
    axes[1, i].imshow(np.clip(recon_img, 0, 16), cmap='gray')  # Clip para melhor visual
    axes[1, i].set_title(f'Recon {i+1}', color='red', fontsize=12)
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()

# 5. GRÁFICO DE LOSS + COMPRESSÃO
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Loss
ax1.plot(losses)
ax1.set_title('Evolução da Perda')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('MSE')
ax1.grid(True, alpha=0.3)

# Compressão
ax2.bar(['Original', 'Comprimido'], [64, 16], color=['red', 'green'])
ax2.set_title('Compressão 4x')
ax2.text(0, 65, '64D', ha='center', va='bottom', fontsize=16, fontweight='bold')
ax2.text(1, 17, '16D', ha='center', va='bottom', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.show()

# RESULTADO FINAL
x_recon_final, _, _ = ae.forward(X_scaled)
mse_final = mean_squared_error(X_scaled, x_recon_final)
print(f"\n RESULTADO FINAL:")
print(f"   MSE Final: {mse_final:.5f}")
print(f"   Compressão: 64D → 16D (4x menor)")
