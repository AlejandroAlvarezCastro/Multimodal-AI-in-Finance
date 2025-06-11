import torch
from torch.utils.data import DataLoader
import optuna
from glob import glob

# from EmbeddingsConstruction.Training.Node.NodeContrastiveDataset import NodeContrastiveDataset
# from EmbeddingsConstruction.Embedding.SentenceAttentionEncoder import SentenceAttentionEncoder
# from EmbeddingsConstruction.Training.aux import nt_xent_loss

from multimodal_fin.embeddings.NodeEncoder import NodeContrastiveDataset
from multimodal_fin.embeddings.SentenceAttentionEncoder import SentenceAttentionEncoder
from multimodal_fin.training.nt_xent_loss import nt_xent_loss


class NodeEncoderTrainer:
    """
    Trainer for contrastive sentence encoder using Optuna optimization.
    """

    def __init__(
        self,
        json_paths,
        input_dim,
        save_path="best_encoder.pt",
        device=None,
        batch_size=16,
        optuna_epochs=5,
        final_epochs=100,
    ):
        # Data and model settings
        self.json_paths = json_paths
        self.input_dim = input_dim
        self.save_path = save_path

        # Device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Training settings
        self.batch_size = batch_size
        self.optuna_epochs = optuna_epochs
        self.final_epochs = final_epochs

        # To store best hyperparameters
        self.best_params = None

    def _objective(self, trial: optuna.trial.Trial):
        # Hyperparameters to tune
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512, 1024])
        n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
        dropout = trial.suggest_float("dropout", 0.1, 0.3)
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

        # Initialize dataset and loader
        dataset = NodeContrastiveDataset(self.json_paths)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        # Initialize model
        model = SentenceAttentionEncoder(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Training loop
        for epoch in range(self.optuna_epochs):
            model.train()
            total_loss = 0.0
            for view1, view2 in loader:
                view1, view2 = view1.to(self.device), view2.to(self.device)
                z1 = model(view1)
                z2 = model(view2)
                loss = nt_xent_loss(z1, z2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            trial.report(avg_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return avg_loss

    def optimize(self, n_trials=30, direction="minimize"):
        """
        Run hyperparameter search and store best parameters.
        """
        study = optuna.create_study(direction=direction)
        study.optimize(self._objective, n_trials=n_trials)
        self.best_params = study.best_params
        print(f"🏆 Best params: {self.best_params}")
        return self.best_params

    def train(self):
        """
        Train model with best parameters and save the encoder.
        """
        if not self.best_params:
            raise RuntimeError("Call optimize() before train_final().")

        # Prepare data loader
        dataset = NodeContrastiveDataset(self.json_paths)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        # Initialize model with best params
        model = SentenceAttentionEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.best_params["hidden_dim"],
            n_heads=self.best_params["n_heads"],
            dropout=self.best_params["dropout"]
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.best_params["lr"])

        # Training
        for epoch in range(self.final_epochs):
            model.train()
            total_loss = 0.0
            for view1, view2 in loader:
                view1, view2 = view1.to(self.device), view2.to(self.device)
                z1 = model(view1)
                z2 = model(view2)
                loss = nt_xent_loss(z1, z2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1}/{self.final_epochs} - Loss: {avg_loss:.4f}")

        # Save model state
        torch.save(model.state_dict(), self.save_path)
        print(f"✅ Model saved to {self.save_path}")
        return model