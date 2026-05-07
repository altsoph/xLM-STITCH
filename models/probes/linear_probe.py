"""Linear, QDA, and small-MLP probes for hidden-state analysis.

Supported probe tasks:
  - layer prediction
  - token identity prediction
  - position prediction
  - token-family classification
  - cross-family transfer (train on family A, test on family B)
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class LinearProbe:
    """Scikit-learn logistic regression probe."""

    def __init__(self, max_iter: int = 5000, C: float = 1.0):
        self.clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=max_iter, C=C, solver="lbfgs", random_state=42),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearProbe":
        self.clf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))

    def report(self, X: np.ndarray, y: np.ndarray, label_names: Optional[List[str]] = None) -> str:
        preds = self.predict(X)
        return classification_report(y, preds, target_names=label_names, zero_division=0)


class QDAProbe:
    """Quadratic Discriminant Analysis probe."""

    def __init__(self, reg_param: float = 0.1):
        self.clf = QuadraticDiscriminantAnalysis(reg_param=reg_param)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QDAProbe":
        self.clf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))


class MLPProbe(nn.Module):
    """Small MLP probe (PyTorch)."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TorchLinearProbe(nn.Module):
    """Single linear layer probe (PyTorch)."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLPProbeTrainer:
    """Training harness for MLPProbe."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 256,
        device: str = "cpu",
        seed: int = 42,
        progress_callback: Optional[Any] = None,
    ):
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.progress_callback = progress_callback
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.model = MLPProbe(input_dim, num_classes, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self._loader_generator = torch.Generator()
        self._loader_generator.manual_seed(seed)

    def _make_loader(self, ds: Any, shuffle: bool) -> DataLoader:
        use_cuda = isinstance(self.device, str) and self.device.startswith("cuda")
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=use_cuda,
            generator=self._loader_generator,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> List[float]:
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )
        loader = self._make_loader(ds, shuffle=True)
        losses = []
        self.model.train()
        for _ in range(self.epochs):
            epoch_idx = len(losses) + 1
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * xb.shape[0]
            epoch_loss = epoch_loss / len(ds)
            losses.append(epoch_loss)
            if self.progress_callback is not None:
                self.progress_callback(epoch_idx, self.epochs, epoch_loss)
        return losses

    def fit_dataset(self, ds: Any) -> List[float]:
        loader = self._make_loader(ds, shuffle=True)
        losses = []
        self.model.train()
        for _ in range(self.epochs):
            epoch_idx = len(losses) + 1
            epoch_loss = 0.0
            total_seen = 0
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * xb.shape[0]
                total_seen += xb.shape[0]
            epoch_loss = epoch_loss / max(total_seen, 1)
            losses.append(epoch_loss)
            if self.progress_callback is not None:
                self.progress_callback(epoch_idx, self.epochs, epoch_loss)
        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(t)
        return logits.argmax(dim=-1).cpu().numpy()

    def predict_dataset(self, ds: Any) -> np.ndarray:
        loader = self._make_loader(ds, shuffle=False)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(self.device, non_blocking=True)
                logits = self.model(xb)
                preds.append(logits.argmax(dim=-1).cpu().numpy())
        if not preds:
            return np.array([], dtype=np.int64)
        return np.concatenate(preds)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))

    def score_dataset(self, ds: Any) -> float:
        loader = self._make_loader(ds, shuffle=False)
        self.model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                logits = self.model(xb)
                preds = logits.argmax(dim=-1)
                correct += int((preds == yb).sum().item())
                total += int(yb.numel())
        return correct / max(total, 1)


class TorchLinearProbeTrainer:
    """Training harness for TorchLinearProbe."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        lr: float = 3e-3,
        weight_decay: float = 1e-4,
        epochs: int = 12,
        batch_size: int = 1024,
        device: str = "cpu",
        seed: int = 42,
        progress_callback: Optional[Any] = None,
    ):
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.progress_callback = progress_callback
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.model = TorchLinearProbe(input_dim, num_classes).to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self._loader_generator = torch.Generator()
        self._loader_generator.manual_seed(seed)

    def _make_loader(self, ds: Any, shuffle: bool) -> DataLoader:
        use_cuda = isinstance(self.device, str) and self.device.startswith("cuda")
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=use_cuda,
            generator=self._loader_generator,
        )

    def fit_dataset(self, ds: Any) -> List[float]:
        loader = self._make_loader(ds, shuffle=True)
        losses = []
        self.model.train()
        for _ in range(self.epochs):
            epoch_idx = len(losses) + 1
            epoch_loss = 0.0
            total_seen = 0
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * xb.shape[0]
                total_seen += xb.shape[0]
            epoch_loss = epoch_loss / max(total_seen, 1)
            losses.append(epoch_loss)
            if self.progress_callback is not None:
                self.progress_callback(epoch_idx, self.epochs, epoch_loss)
        return losses

    def score_dataset(self, ds: Any) -> float:
        loader = self._make_loader(ds, shuffle=False)
        self.model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                logits = self.model(xb)
                preds = logits.argmax(dim=-1)
                correct += int((preds == yb).sum().item())
                total += int(yb.numel())
        return correct / max(total, 1)


def prepare_layer_prediction_data(
    hidden_states: Dict[int, torch.Tensor],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (X, y) for layer-identity prediction from per-layer hidden states."""
    X_parts, y_parts = [], []
    for layer_idx, hs in sorted(hidden_states.items()):
        flat = hs.reshape(-1, hs.shape[-1]).numpy()
        X_parts.append(flat)
        y_parts.append(np.full(flat.shape[0], layer_idx))
    return np.concatenate(X_parts), np.concatenate(y_parts)


def prepare_token_family_data(
    hidden_states: torch.Tensor,
    families: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (X, y) for token-family classification."""
    X = hidden_states.reshape(-1, hidden_states.shape[-1]).numpy()
    y = np.array(families)
    return X, y
