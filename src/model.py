import torch
import torch.nn as nn
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            
            nn.Linear(64, latent_dim) # Latent Bottleneck (Linear)
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.decoder(x)

class MultiModalAutoencoder(nn.Module):
    """
    Symmetric Deep Autoencoder for Equity Embeddings.
    """
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super(MultiModalAutoencoder, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

if __name__ == "__main__":
    # Test with dummy input
    input_dim = 85
    model = MultiModalAutoencoder(input_dim=input_dim)
    dummy_input = torch.randn(32, input_dim)
    reconstructed, latent = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
