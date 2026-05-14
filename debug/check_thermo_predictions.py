# debug/check_thermo_predictions.py
"""
Диагностика предсказаний термометрии.
Запускать после того, как эксперимент thermo уже выполнен.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.config import create_thermometry_config
from physics.evolution import FieldEvolution
from data.preprocess import apply_pca, preprocess_pipeline, split_and_shuffle_data, normalize_labels_for_regression
from ml.model import create_model

# Загружаем конфиг
config = create_thermometry_config()
config.measurement.N_tom = 1e6
config.ml.N_samples = 500

print("=" * 60)
print("ДИАГНОСТИКА ПРЕДСКАЗАНИЙ ТЕРМОМЕТРИИ")
print("=" * 60)

# Генерируем данные для одного окна
evolution = FieldEvolution(config)
windows = config.measurement.measurement_windows

# Берём окно с Tmax ≈ 13 (последнее, где точность максимальна)
Tmin = windows[-2]
Tmax = windows[-1]
print(f"\nАнализируем окно: Tmin={Tmin:.1f}, Tmax={Tmax:.1f}")

X = evolution.generate_data_for_window(Tmin, Tmax)
N_features = X.shape[1] - 1
X_data = X[:, :N_features]
y_data = X[:, N_features]

print(f"Данные: {X_data.shape}")
print(f"y_data range: [{y_data.min():.6f}, {y_data.max():.6f}]")
print(f"y_data unique: {len(np.unique(y_data))} (из {len(y_data)} образцов)")

# Разделение
X_train, y_train, X_valid, y_valid = split_and_shuffle_data(
    X_data, y_data, f_train=0.75, f_valid=0.25, f_test=0.0, random_seed=42
)

# Нормализация
y_train_norm, y_min, y_max = normalize_labels_for_regression(y_train)
y_valid_norm, _, _ = normalize_labels_for_regression(y_valid, y_train=y_train)

print(f"\nНормализация:")
print(f"  y_train: [{y_train.min():.6f}, {y_train.max():.6f}]")
print(f"  y_train_norm: [{y_train_norm.min():.4f}, {y_train_norm.max():.4f}]")
print(f"  y_min={y_min:.6f}, y_max={y_max:.6f}")
print(f"  y_valid: [{y_valid.min():.6f}, {y_valid.max():.6f}]")
print(f"  y_valid_norm: [{y_valid_norm.min():.4f}, {y_valid_norm.max():.4f}]")

# Проверка: восстановление из нормализации
y_valid_recovered = (y_valid_norm - 0.25) * 2.0 * (y_max - y_min) + y_min
recovery_error = np.abs(y_valid_recovered - y_valid).max()
print(f"  Ошибка восстановления: {recovery_error:.2e}")

# PCA
X_train_pca, y_train_final, X_valid_pca, y_valid_final, pca_params = preprocess_pipeline(
    X_train, y_train_norm, X_valid, y_valid_norm,
    var_keep=1.0, is_regression=True
)
print(f"\nPCA: {X_train.shape[1]} -> {X_train_pca.shape[1]} компонент")

# Модель
model = create_model(
    n_features=X_train_pca.shape[1],
    n_output=1, is_regression=True, n_hidden=30
)

# Обучаем быстро (50 эпох для диагностики)
from ml.train import train_model

history = train_model(
    model, X_train_pca, y_train_final,
    X_valid_pca, y_valid_final,
    is_regression=True, n_epochs=100, batch_size=100,
    learning_rate=0.01, l2_lambda=0.001, verbose=False
)

# Предсказания
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

X_valid_t = torch.FloatTensor(X_valid_pca).to(device)
with torch.no_grad():
    preds_norm = model(X_valid_t).squeeze().cpu().numpy()

# Денормализация
preds = (preds_norm - 0.25) * 2.0 * (y_max - y_min) + y_min
targets = (y_valid_final - 0.25) * 2.0 * (y_max - y_min) + y_min

print(f"\nПредсказания (денормализованные):")
print(f"  preds: [{preds.min():.6f}, {preds.max():.6f}], mean={preds.mean():.6f}")
print(f"  targets: [{targets.min():.6f}, {targets.max():.6f}], mean={targets.mean():.6f}")

bias = np.mean(preds - targets)
print(f"  BIAS (pred - true): {bias:.6f}")
print(f"  MAE: {np.abs(preds - targets).mean():.6f}")
print(f"  RMSE: {np.sqrt(np.mean((preds - targets)**2)):.6f}")

# Смотрим на распределение предсказаний по бинам
print(f"\nРаспределение по бинам:")
unique_targets = np.sort(np.unique(targets))
for t in unique_targets:
    mask = np.abs(targets - t) < 1e-6
    if mask.sum() > 0:
        print(f"  T={t:.6f}: pred_mean={preds[mask].mean():.6f}, pred_std={preds[mask].std():.6f}, n={mask.sum()}")

# График
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Scatter
ax1.scatter(targets, preds, alpha=0.5, s=5)
ax1.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', linewidth=2)
ax1.set_xlabel('True T')
ax1.set_ylabel('Predicted T')
ax1.set_title(f'Predicted vs True (bias={bias:.6f})')
ax1.grid(True, alpha=0.3)

# Ошибки
errors = preds - targets
ax2.hist(errors, bins=50, edgecolor='black', alpha=0.7)
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Error (pred - true)')
ax2.set_ylabel('Count')
ax2.set_title(f'Error Distribution (bias={bias:.6f}, MAE={np.abs(errors).mean():.6f})')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('debug/diagnostic_thermo.png', dpi=150)
print(f"\nГрафик сохранён в debug/diagnostic_thermo.png")