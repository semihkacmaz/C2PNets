import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import matplotlib.pyplot as plt
from .losses import PenalizedMSELoss

class Trainer:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.model.to(device)

        lr = float(config["training"]["learning_rate"])
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        self.criterion = PenalizedMSELoss(penalty_factor=config["training"].get("penalty_factor", 0))

        self.train_losses = []
        self.val_losses = []
        self.model_save_path = Path(config["paths"]["models"])
        self.model_save_path.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, train_loader, output_scaler):
        self.model.train()
        total_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets, output_scaler.mean_[0], output_scaler.scale_[0])

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * inputs.size(0)

        return total_loss / len(train_loader.dataset)

    def validate(self, val_loader, output_scaler):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets, output_scaler.mean_[0], output_scaler.scale_[0])
                total_loss += loss.item() * inputs.size(0)

        return total_loss / len(val_loader.dataset)

    def save_model(self, suffix=""):
        model_name = self.config["model"]["name"]
        save_name = f"{model_name}{suffix}.pth"
        scripted_model = torch.jit.script(self.model)
        torch.jit.save(scripted_model, self.model_save_path / save_name)

    def train(self, train_loader, val_loader, output_scaler):
        checkpoint_interval = self.config["training"].get("checkpoint_interval", None)

        if checkpoint_interval:
            checkpoint_path = self.model_save_path / "checkpoints"
            checkpoint_path.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.config["training"]["num_epochs"]):
            train_loss = self.train_epoch(train_loader, output_scaler)
            val_loss = self.validate(val_loader, output_scaler)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.scheduler.step(val_loss)

            print(f'Epoch {epoch+1}/{self.config["training"]["num_epochs"]} '
                  f'Train Loss: {train_loss:.6f} Val Loss: {val_loss:.6f}')

            # Save checkpoint if interval is specified and epoch matches
            if checkpoint_interval and (epoch + 1) % checkpoint_interval == 0:
                self.save_model(f"_checkpoint_{epoch+1}")

        # Save final model
        self.save_model()
        self.plot_training_history(self.config["model"]["name"])

    def plot_training_history(self, model_name):
        plt.figure(figsize=(10, 5))
        plt.semilogy(range(1, len(self.train_losses) + 1), self.train_losses, label='Training Loss')
        plt.semilogy(range(1, len(self.val_losses) + 1), self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss vs. Epochs')
        plt.savefig(Path(self.config["paths"]["images"]) / f'{model_name}_training_history.png', dpi=600)
        plt.close()
