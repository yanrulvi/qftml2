# ml/train.py
"""
Цикл обучения нейросети (с поддержкой GPU).

Точно воспроизводит обучение из оригинального DataGenerator.py:
- Функция потерь:
  - Классификация: CrossEntropy + L2_reg
  - Регрессия: MSE + L2_reg
- Оптимизатор: SGD (GradientDescent) с learning_rate=0.01
- Мини-батчи со случайным перемешиванием
- Валидация на каждой эпохе
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List, Optional


# Определяем устройство (GPU если доступно)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"ml/train.py: Используется устройство: {device}")


def l2_regularization(model: nn.Module) -> torch.Tensor:
    """
    Вычисляет L2-регуляризацию для всех весов модели.
    tf.nn.l2_loss = sum(t**2) / 2
    """
    l2_loss = torch.tensor(0.0, device=device)
    for param in model.parameters():
        if param.requires_grad:
            l2_loss += torch.sum(param**2) / 2.0
    return l2_loss


def train_epoch(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    l2_lambda: float = 0.001,
    batch_size: int = 100,
    is_regression: bool = False,
) -> float:
    """Одна эпоха обучения с мини-батчами."""
    model.train()
    n_samples = X_train.shape[0]

    # Данные уже должны быть на device
    perm = torch.randperm(n_samples, device=device)
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    total_loss = 0.0
    n_batches = 0

    for b in range(0, n_samples, batch_size):
        X_batch = X_shuffled[b : b + batch_size]
        y_batch = y_shuffled[b : b + batch_size]

        optimizer.zero_grad()
        outputs = model(X_batch)

        if is_regression:
            loss = loss_fn(outputs.squeeze(), y_batch)
        else:
            loss = loss_fn(outputs, y_batch.long())

        l2_loss = l2_regularization(model)
        total = loss + l2_lambda * l2_loss

        total.backward()
        optimizer.step()

        total_loss += total.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    loss_fn: nn.Module,
    l2_lambda: float = 0.001,
    is_regression: bool = False,
    y_min: float = None,
    y_max: float = None,
    n_classes: int = None,
) -> Tuple[float, float]:
    """Оценка модели на валидационных/тестовых данных."""
    model.eval()

    with torch.no_grad():
        outputs = model(X)

        if is_regression:
            loss = loss_fn(outputs.squeeze(), y)
            l2_loss = l2_regularization(model)
            cost = (loss + l2_lambda * l2_loss).item()

            preds = outputs.squeeze().cpu().numpy()
            targets = y.cpu().numpy()

            # Денормализация
            if y_min is not None and y_max is not None:
                preds = (preds - 0.25) * 2.0 * (y_max - y_min) + y_min
                targets = (targets - 0.25) * 2.0 * (y_max - y_min) + y_min

            # КАК В СТАТЬЕ: попадание в ±1% от истинной температуры
            accurate = np.abs(preds - targets) < 0.01 * np.abs(targets)
            accuracy = np.mean(accurate)

        else:
            loss = loss_fn(outputs, y.long())
            l2_loss = l2_regularization(model)
            cost = (loss + l2_lambda * l2_loss).item()

            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y).float().mean().item()

    return cost, accuracy


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    is_regression: bool = False,
    n_epochs: int = 1000,
    batch_size: int = 100,
    learning_rate: float = 0.01,
    l2_lambda: float = 0.001,
    y_min: float = None,
    y_max: float = None,
    n_classes: int = None,
    verbose: bool = True,
    print_every: int = 10,
) -> dict:
    """Полный цикл обучения."""

    # Переносим модель на GPU
    model = model.to(device)

    # Конвертируем данные и переносим на GPU
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_valid_t = torch.FloatTensor(X_valid).to(device)
    y_valid_t = torch.FloatTensor(y_valid).to(device)

    # Функция потерь
    if is_regression:
        loss_fn = nn.MSELoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    # Оптимизатор
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # История
    history = {
        "epoch": [],
        "train_loss": [],
        "valid_loss": [],
        "train_acc": [],
        "valid_acc": [],
    }

    if is_regression:
        init_acc = 0.0
    else:
        init_acc = 1.0 / max(model.fc2.out_features, 2)

    history["epoch"].append(-1)
    history["train_loss"].append(np.nan)
    history["valid_loss"].append(np.nan)
    history["train_acc"].append(init_acc)
    history["valid_acc"].append(init_acc)

    for epoch in range(n_epochs):
        train_loss = train_epoch(
            model,
            X_train_t,
            y_train_t,
            optimizer,
            loss_fn,
            l2_lambda=l2_lambda,
            batch_size=batch_size,
            is_regression=is_regression,
        )

        train_cost, train_acc = evaluate(
            model,
            X_train_t,
            y_train_t,
            loss_fn,
            l2_lambda=l2_lambda,
            is_regression=is_regression,
            y_min=y_min,
            y_max=y_max,
            n_classes=n_classes,
        )
        valid_cost, valid_acc = evaluate(
            model,
            X_valid_t,
            y_valid_t,
            loss_fn,
            l2_lambda=l2_lambda,
            is_regression=is_regression,
            y_min=y_min,
            y_max=y_max,
            n_classes=n_classes,
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_cost)
        history["valid_loss"].append(valid_cost)
        history["train_acc"].append(train_acc)
        history["valid_acc"].append(valid_acc)

        if verbose and epoch % print_every == 0:
            print(
                f"  Epoch {epoch:4d}: "
                f"Train loss={train_cost:.6f}, Valid loss={valid_cost:.6f}, "
                f"Train acc={train_acc:.4f}, Valid acc={valid_acc:.4f}"
            )

    return history
