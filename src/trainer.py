import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
import os

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    ModelTrainer handles the training process for the MultiModalAutoencoder.
    """
    def __init__(self, model: nn.Module, lr: float = 1e-3, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        for batch_x in dataloader:
            batch_x = batch_x[0].to(self.device)
            
            self.optimizer.zero_grad()
            reconstructed, _ = self.model(batch_x)
            loss = self.criterion(reconstructed, batch_x)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)

    def validate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_x in dataloader:
                batch_x = batch_x[0].to(self.device)
                reconstructed, _ = self.model(batch_x)
                loss = self.criterion(reconstructed, batch_x)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)

    def fit(self, train_data: torch.Tensor, val_data: torch.Tensor = None, epochs: int = 100, batch_size: int = 32):
        train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)
        if val_data is not None:
            val_loader = DataLoader(TensorDataset(val_data), batch_size=batch_size)
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            if val_data is not None:
                val_loss = self.validate(val_loader)
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}")

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        logger.info(f"Model loaded from {path}")

if __name__ == "__main__":
    from src.model import MultiModalAutoencoder
    
    input_dim = 85
    model = MultiModalAutoencoder(input_dim)
    trainer = ModelTrainer(model)
    
    # Dummy data
    train_x = torch.randn(100, input_dim)
    trainer.fit(train_x, epochs=5)
