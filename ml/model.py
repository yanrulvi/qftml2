# ml/model.py
"""
Нейросеть на PyTorch — точное воспроизведение архитектуры из статьи.

Архитектура из оригинального кода (TensorFlow 1.x) и Appendix A:
- Input(n_features) → Dense(n_hidden=30, ReLU) → Dense(n_output)
- Для классификации: выходной слой → softmax
- Для регрессии: выходной слой → sigmoid
- L2-регуляризация на весах обоих слоёв
- Оптимизатор: градиентный спуск (learning_rate=0.01)

Инициализация весов (из кода авторов):
- tf.random_normal(mean=0.0, stddev=sqrt(2/(n_in + n_out)))
  Это близко к Xavier/Glorot инициализации
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QFTClassifier(nn.Module):
    """
    Нейросеть для классификации (boundary sensing).
    
    Архитектура: 90 → 30 → n_classes
    (90 — размерность после PCA, в общем случае n_features)
    """
    
    def __init__(self, n_features: int, n_classes: int, n_hidden: int = 30):
        super().__init__()
        
        # Инициализация весов как в оригинале: N(0, sqrt(2/(n_in + n_out)))
        self.fc1 = nn.Linear(n_features, n_hidden)
        std_fc1 = np.sqrt(2.0 / (n_features + n_hidden))
        nn.init.normal_(self.fc1.weight, mean=0.0, std=std_fc1)
        nn.init.zeros_(self.fc1.bias)
        
        self.fc2 = nn.Linear(n_hidden, n_classes)
        std_fc2 = np.sqrt(2.0 / (n_hidden + n_classes))
        nn.init.normal_(self.fc2.weight, mean=0.0, std=std_fc2)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # softmax применяется в функции потерь (CrossEntropyLoss)
        return x


class QFTRegressor(nn.Module):
    """
    Нейросеть для регрессии (thermometry).
    
    Архитектура: n_features → 30 → 1
    Выход: сигмоида (предсказания в диапазоне [0, 1])
    """
    
    def __init__(self, n_features: int, n_hidden: int = 30):
        super().__init__()
        
        self.fc1 = nn.Linear(n_features, n_hidden)
        std_fc1 = np.sqrt(2.0 / (n_features + n_hidden))
        nn.init.normal_(self.fc1.weight, mean=0.0, std=std_fc1)
        nn.init.zeros_(self.fc1.bias)
        
        self.fc2 = nn.Linear(n_hidden, 1)
        std_fc2 = np.sqrt(2.0 / (n_hidden + 1))
        nn.init.normal_(self.fc2.weight, mean=0.0, std=std_fc2)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Сигмоида (как в оригинальном коде)
        return x


def create_model(
    n_features: int,
    n_output: int,
    is_regression: bool = False,
    n_hidden: int = 30
) -> nn.Module:
    """
    Фабрика для создания модели нужного типа.
    
    Args:
        n_features: размерность входа
        n_output: размерность выхода (число классов или 1 для регрессии)
        is_regression: True для регрессии, False для классификации
        n_hidden: число нейронов в скрытом слое
    
    Returns:
        Модель PyTorch
    """
    if is_regression:
        return QFTRegressor(n_features, n_hidden)
    else:
        return QFTClassifier(n_features, n_output, n_hidden)