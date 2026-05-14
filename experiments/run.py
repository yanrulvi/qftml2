# experiments/run.py
"""
Главный файл для запуска экспериментов из статьи:
'Decoding Quantum Field Theory with Machine Learning'

Воспроизводит:
- Секция IV: Remote Boundary Sensing (Fig. 2)
- Секция V: Thermometry (Fig. 3)

Запуск:
    python experiments/run.py boundary   # Boundary sensing
    python experiments/run.py thermo     # Thermometry
    python experiments/run.py test       # Быстрый тест
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.config import (
    ExperimentConfig,
    ExperimentMode,
    create_boundary_config,
    create_thermometry_config,
    create_test_config,
)
from physics.evolution import FieldEvolution
from data.preprocess import (
    split_and_shuffle_data,
    preprocess_pipeline,
    normalize_labels_for_regression,
    denormalize_predictions,
)
from ml.model import create_model
from ml.train import train_model


###############################################################################
# Утилиты
###############################################################################

def setup_directories(base_dir: str, experiment_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    return exp_dir


def save_fig(fig, exp_dir: str, name: str):
    path = os.path.join(exp_dir, "plots", name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Сохранён: plots/{name}")


###############################################################################
# Генерация и обработка окна
###############################################################################

def generate_window_data(evolution, config, Tmin, Tmax, window_idx, exp_dir):
    print(f"  Window {window_idx}: Tmin={Tmin:.3f}, Tmax={Tmax:.3f}")
    X = evolution.generate_data_for_window(Tmin, Tmax)
    N_features = X.shape[1] - 1
    X_data = X[:, :N_features]
    y_data = X[:, N_features]
    window_df = pd.DataFrame(X)
    window_df.to_csv(
        os.path.join(exp_dir, "data", f"window_{window_idx:03d}.csv"),
        header=False, index=False,
    )
    return X_data, y_data


def process_window(X, y, config, window_Tmax):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, y_train, X_valid, y_valid = split_and_shuffle_data(
        X, y,
        f_train=config.ml.f_train,
        f_valid=config.ml.f_valid,
        f_test=config.ml.f_test,
        random_seed=42,
    )

    print(f"    Train: {X_train.shape[0]}, Valid: {X_valid.shape[0]}")
    print(f"    Train y unique: {len(np.unique(y_train))}, Valid y unique: {len(np.unique(y_valid))}")

    y_min, y_max = None, None
    if config.is_regression:
        y_train_norm, y_min, y_max = normalize_labels_for_regression(y_train)
        y_valid_norm, _, _ = normalize_labels_for_regression(y_valid, y_train=y_train)
        print(f"    y_train_norm: [{y_train_norm.min():.3f}, {y_train_norm.max():.3f}]")
        print(f"    y_valid_norm: [{y_valid_norm.min():.3f}, {y_valid_norm.max():.3f}]")
    else:
        y_train_norm = y_train.copy()
        y_valid_norm = y_valid.copy()

    X_train_pca, y_train_final, X_valid_pca, y_valid_final, pca_params = preprocess_pipeline(
        X_train, y_train_norm, X_valid, y_valid_norm,
        var_keep=config.ml.pca_var_keep,
        is_regression=config.is_regression,
    )

    n_features = X_train_pca.shape[1]

    model = create_model(
        n_features=n_features,
        n_output=config.n_classes,
        is_regression=config.is_regression,
        n_hidden=config.ml.n_hidden,
    )

    batch_size = min(config.ml.minibatch_size, X_train_pca.shape[0])
    history = train_model(
        model=model,
        X_train=X_train_pca,
        y_train=y_train_final,
        X_valid=X_valid_pca,
        y_valid=y_valid_final,
        is_regression=config.is_regression,
        n_epochs=config.ml.n_epochs,
        batch_size=batch_size,
        learning_rate=config.ml.learning_rate,
        l2_lambda=config.ml.L2_reg,
        y_min=y_min,
        y_max=y_max,
        verbose=False,
    )

    # Предсказания на валидации
    preds_denorm = None
    targets_denorm = None
    if config.is_regression:
        model.eval()
        X_valid_t = torch.FloatTensor(X_valid_pca).to(device)
        with torch.no_grad():
            preds = model(X_valid_t).squeeze().cpu().numpy()

        print(f"    [DEBUG] Raw preds: min={preds.min():.4f}, max={preds.max():.4f}, mean={preds.mean():.4f}")
        print(f"    [DEBUG] Raw targets: min={y_valid_final.min():.4f}, max={y_valid_final.max():.4f}, mean={y_valid_final.mean():.4f}")

        if y_min is not None and y_max is not None:
            preds_denorm = (preds - 0.25) * 2.0 * (y_max - y_min) + y_min
            targets_denorm = (y_valid_final - 0.25) * 2.0 * (y_max - y_min) + y_min
            mae = np.abs(preds_denorm - targets_denorm).mean()
            print(f"    [DEBUG] MAE (денорм): {mae:.6f}")

    valid_accs = history["valid_acc"][1:]
    best_acc = max(valid_accs) if valid_accs else 0.0
    best_epoch = valid_accs.index(best_acc) + 1 if valid_accs else 0
    print(f"    Best valid_acc={best_acc:.4f} at epoch {best_epoch}")

    return {
        "Tmax": window_Tmax,
        "best_valid_acc": best_acc,
        "best_epoch": best_epoch,
        "n_features": n_features,
        "train_size": X_train_pca.shape[0],
        "valid_size": X_valid_pca.shape[0],
        "history": history,
        "model_state": model.state_dict(),
        "pca_params": pca_params,
        "y_min": y_min,
        "y_max": y_max,
        "preds_denorm": preds_denorm,
        "targets_denorm": targets_denorm,
    }


###############################################################################
# Графики
###############################################################################

def plot_accuracy_vs_time(T_list, acc_list, config, exp_dir):
    """График 1: Accuracy vs время (рис. 2 / рис. 3 из статьи)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(T_list, [a * 100 for a in acc_list],
            "o-", lw=2, ms=7, color="#2271b3", label="Нейросеть")

    if config.mode == ExperimentMode.BOUNDARY_SENSING:
        light_time = config.physics.LatticeLength
        ax.axvline(light_time, color="red", ls="--", alpha=0.7,
                   label=f"Световое время ≈ {light_time:.0f}")
        ax.axhline(33, color="gray", ls=":", alpha=0.6, label="Случайное угадывание (33%)")
        ax.set_title("Fig. 2 — Boundary Sensing: точность vs время измерения", fontsize=13)
        ax.set_ylabel("Точность валидации (%)")
    else:
        heis = 1.0
        ax.axvline(heis, color="red", ls="--", alpha=0.7,
                   label=f"Время Гейзенберга = {heis:.2f}")
        ax.axhline(10, color="gray", ls=":", alpha=0.6, label="Случайное угадывание (10%)")
        ax.set_title("Fig. 3 — Thermometry: точность ±1% vs время измерения", fontsize=13)
        ax.set_ylabel("Точность валидации ±1% (%)")

    ax.set_xlabel("Время взаимодействия Tmax")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, exp_dir, "fig_accuracy_vs_time.pdf")


def plot_training_curves(results_by_window, config, exp_dir):
    """График 2: Кривые обучения для всех окон на одном полотне."""
    n = len(results_by_window)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()

    label_y = "Accuracy (%)" if not config.is_regression else "Accuracy ±1% (%)"

    for i, r in enumerate(results_by_window):
        ax = axes[i]
        hist = r["history"]
        epochs = hist["epoch"][1:]
        train_acc = [v * 100 for v in hist["train_acc"][1:]]
        valid_acc = [v * 100 for v in hist["valid_acc"][1:]]

        ax.plot(epochs, train_acc, lw=1.2, color="#e07b39", label="Train", alpha=0.85)
        ax.plot(epochs, valid_acc, lw=1.5, color="#2271b3", label="Valid")
        ax.axhline(r["best_valid_acc"] * 100, color="green", ls=":", lw=1, alpha=0.7)

        ax.set_title(f"Tmax={r['Tmax']:.2f}  best={r['best_valid_acc']*100:.1f}%",
                     fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel(label_y, fontsize=8)
        ax.set_ylim(0, 105)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Кривые обучения по временным окнам", fontsize=13, y=1.01)
    plt.tight_layout()
    save_fig(fig, exp_dir, "fig_training_curves.pdf")


def plot_loss_curves(results_by_window, config, exp_dir):
    """График 3: Train/Valid loss для каждого окна."""
    n = len(results_by_window)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()

    for i, r in enumerate(results_by_window):
        ax = axes[i]
        hist = r["history"]
        epochs = hist["epoch"][1:]
        train_loss = hist["train_loss"][1:]
        valid_loss = hist["valid_loss"][1:]

        ax.plot(epochs, train_loss, lw=1.2, color="#e07b39", label="Train loss", alpha=0.85)
        ax.plot(epochs, valid_loss, lw=1.5, color="#2271b3", label="Valid loss")
        ax.set_title(f"Tmax={r['Tmax']:.2f}", fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Loss", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Loss по временным окнам", fontsize=13, y=1.01)
    plt.tight_layout()
    save_fig(fig, exp_dir, "fig_loss_curves.pdf")


def plot_n_features_vs_time(results_by_window, exp_dir):
    """График 4: Число PCA компонент vs время."""
    T_list  = [r["Tmax"]       for r in results_by_window]
    n_feats = [r["n_features"] for r in results_by_window]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(T_list)), n_feats, color="#5aab7d", alpha=0.8)
    ax.set_xticks(range(len(T_list)))
    ax.set_xticklabels([f"{t:.2f}" for t in T_list], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Tmax")
    ax.set_ylabel("Число PCA компонент")
    ax.set_title("Информативные компоненты PCA по окнам")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    save_fig(fig, exp_dir, "fig_n_pca_features.pdf")


def plot_best_epoch_vs_time(results_by_window, config, exp_dir):
    """График 5: Эпоха лучшей точности vs время."""
    T_list   = [r["Tmax"]       for r in results_by_window]
    epochs   = [r["best_epoch"] for r in results_by_window]
    acc_list = [r["best_valid_acc"] for r in results_by_window]

    fig, ax1 = plt.subplots(figsize=(9, 4))
    color_e = "#e07b39"
    color_a = "#2271b3"

    ax1.bar(range(len(T_list)), epochs, color=color_e, alpha=0.7, label="Эпоха лучшей точности")
    ax1.set_xticks(range(len(T_list)))
    ax1.set_xticklabels([f"{t:.2f}" for t in T_list], rotation=45, ha="right", fontsize=8)
    ax1.set_xlabel("Tmax")
    ax1.set_ylabel("Эпоха лучшей точности", color=color_e)
    ax1.tick_params(axis="y", labelcolor=color_e)
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twinx()
    ax2.plot(range(len(T_list)), [a * 100 for a in acc_list],
             "o-", color=color_a, lw=2, ms=6, label="Val accuracy (%)")
    ax2.set_ylabel("Val accuracy (%)", color=color_a)
    ax2.tick_params(axis="y", labelcolor=color_a)
    ax2.set_ylim(0, 105)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")
    ax1.set_title("Скорость сходимости и точность по окнам")
    ax1.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    save_fig(fig, exp_dir, "fig_convergence.pdf")


def plot_pca_eigenvalues(results_by_window, exp_dir, n_show=5):
    """График 6: Первые N собственных значений PCA по окнам."""
    T_list = [r["Tmax"] for r in results_by_window]
    eig_matrix = []
    for r in results_by_window:
        lam = r["pca_params"].get("eigenvalues", None)
        if lam is not None:
            eig_matrix.append(np.array(lam[:n_show]))
        else:
            eig_matrix.append(np.zeros(n_show))

    eig_matrix = np.array(eig_matrix)
    if eig_matrix.max() == 0:
        print("  [SKIP] Собственные значения не найдены в pca_params, пропускаем график 6")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_show))
    for k in range(min(n_show, eig_matrix.shape[1])):
        ax.plot(T_list, eig_matrix[:, k], "o-", color=colors[k],
                lw=1.5, ms=5, label=f"λ{k+1}")

    ax.set_xlabel("Tmax")
    ax.set_ylabel("Собственное значение")
    ax.set_yscale("log")
    ax.set_title(f"Первые {n_show} собственных значений PCA по окнам")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    save_fig(fig, exp_dir, "fig_pca_eigenvalues.pdf")


def plot_summary_panel(T_list, acc_list, results_by_window, config, exp_dir):
    """График 7: Сводная панель — accuracy + n_features + best_epoch."""
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    n_feats  = [r["n_features"] for r in results_by_window]
    b_epochs = [r["best_epoch"] for r in results_by_window]

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(T_list, [a * 100 for a in acc_list],
             "o-", lw=2.5, ms=8, color="#2271b3", label="Нейросеть")

    if config.mode == ExperimentMode.BOUNDARY_SENSING:
        lt = config.physics.LatticeLength
        ax1.axvline(lt, color="red", ls="--", alpha=0.7, label=f"Световое время ≈ {lt:.0f}")
        ax1.axhline(50, color="gray", ls=":", alpha=0.5, label="Случайное (50%)")
        ax1.set_title("Fig. 2 — Boundary Sensing", fontsize=13, fontweight="bold")
    else:
        heis = 1.0 / config.physics.wD
        ax1.axvline(heis, color="red", ls="--", alpha=0.7,
                    label=f"Время Гейзенберга = {heis:.2f}")
        ax1.axhline(10, color="gray", ls=":", alpha=0.5, label="Случайное (10%)")
        ax1.set_title("Fig. 3 — Thermometry", fontsize=13, fontweight="bold")

    ax1.set_xlabel("Время взаимодействия Tmax", fontsize=11)
    ax1.set_ylabel("Точность валидации (%)", fontsize=11)
    ax1.set_ylim(0, 105)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.bar(range(len(T_list)), n_feats, color="#5aab7d", alpha=0.8)
    ax2.set_xticks(range(0, len(T_list), max(1, len(T_list)//6)))
    ax2.set_xticklabels([f"{T_list[i]:.1f}" for i in range(0, len(T_list), max(1, len(T_list)//6))],
                         fontsize=8)
    ax2.set_xlabel("Tmax", fontsize=10)
    ax2.set_ylabel("PCA компонент", fontsize=10)
    ax2.set_title("Информативные PCA компоненты", fontsize=11)
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.grid(True, axis="y", alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(T_list, b_epochs, "s-", lw=1.5, ms=6, color="#e07b39")
    ax3.set_xlabel("Tmax", fontsize=10)
    ax3.set_ylabel("Эпоха лучшей точности", fontsize=10)
    ax3.set_title("Скорость сходимости", fontsize=11)
    ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax3.grid(True, alpha=0.3)

    save_fig(fig, exp_dir, "fig_summary_panel.pdf")


def plot_true_vs_pred(results_by_window, config, exp_dir):
    """
    График 8: y_true vs y_pred для задачи регрессии (термометрия).
    Показывает качество предсказаний нейросети.
    Теоретически точки должны лежать на прямой y_true = y_pred.
    """
    if not config.is_regression:
        print("  [SKIP] y_true vs y_pred строится только для регрессии")
        return

    n = len(results_by_window)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()

    for i, r in enumerate(results_by_window):
        ax = axes[i]
        preds = r.get("preds_denorm")
        targets = r.get("targets_denorm")

        if preds is None or targets is None:
            ax.text(0.5, 0.5, "Нет данных", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")
            ax.set_title(f"Tmax={r['Tmax']:.2f}", fontsize=9)
            continue

        ax.scatter(targets, preds, alpha=0.5, s=20, color="#2271b3", edgecolors="none")

        # Прямая y_true = y_pred
        all_vals = np.concatenate([preds, targets])
        min_val, max_val = all_vals.min(), all_vals.max()
        margin = (max_val - min_val) * 0.05
        ax.plot([min_val - margin, max_val + margin],
                [min_val - margin, max_val + margin],
                "r--", lw=1.2, alpha=0.7, label="y_pred = y_true")

        # Текст с MAE и accuracy
        mae = np.abs(preds - targets).mean()
        ax.text(0.05, 0.92, f"MAE={mae:.4f}", transform=ax.transAxes,
                fontsize=7, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        ax.set_xlabel("y_true", fontsize=8)
        ax.set_ylabel("y_pred", fontsize=8)
        ax.set_title(f"Tmax={r['Tmax']:.2f}  acc={r['best_valid_acc']*100:.1f}%", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("y_true vs y_pred (регрессия, термометрия)", fontsize=13, y=1.01)
    plt.tight_layout()
    save_fig(fig, exp_dir, "fig_true_vs_pred.pdf")


###############################################################################
# Основной эксперимент
###############################################################################

def run_experiment(config: ExperimentConfig, exp_dir: str):
    print("=" * 70)
    print(f"Эксперимент: {config.mode.value}")
    print(f'Режим: {"Регрессия" if config.is_regression else "Классификация"}')
    print(f"Случаев: {len(config.cases)}")
    print(f"N_tom: {config.measurement.N_tom:.0e}")
    print(f"N_samples на случай: {config.ml.N_samples}")
    print(f"Архитектура: [n_features] → {config.ml.n_hidden} → {config.n_classes}")
    print("=" * 70)

    evolution = FieldEvolution(config)
    windows = config.measurement.measurement_windows
    n_windows = len(windows) - 1
    results_by_window = []

    print(f"\nОбработка {n_windows} временных окон...")
    print("-" * 70)

    for t_idx in range(1, len(windows)):
        Tmin = windows[t_idx - 1]
        Tmax = windows[t_idx]
        X, y = generate_window_data(evolution, config, Tmin, Tmax, t_idx, exp_dir)
        result = process_window(X, y, config, Tmax)
        results_by_window.append(result)

    T_list   = [r["Tmax"]          for r in results_by_window]
    acc_list = [r["best_valid_acc"] for r in results_by_window]

    # Сводная таблица
    print("\n" + "=" * 70)
    print("СВОДКА ПО ВРЕМЕННЫМ ОКНАМ")
    print("=" * 70)
    print(f'{"Tmax":>8s}  {"Accuracy":>10s}  {"Features":>8s}  {"BestEpoch":>10s}')
    print("-" * 42)
    for r in results_by_window:
        print(f'{r["Tmax"]:8.3f}  {r["best_valid_acc"]:10.4f}  '
              f'{r["n_features"]:8d}  {r["best_epoch"]:10d}')

    # CSV
    df = pd.DataFrame({
        "Tmax":          T_list,
        "best_valid_acc": acc_list,
        "n_features":    [r["n_features"] for r in results_by_window],
        "best_epoch":    [r["best_epoch"]  for r in results_by_window],
    })
    df.to_csv(os.path.join(exp_dir, "accuracy_vs_time.csv"), index=False)

    # Графики
    print("\nСохранение графиков...")
    plot_accuracy_vs_time(T_list, acc_list, config, exp_dir)
    plot_training_curves(results_by_window, config, exp_dir)
    plot_loss_curves(results_by_window, config, exp_dir)
    plot_n_features_vs_time(results_by_window, exp_dir)
    plot_best_epoch_vs_time(results_by_window, config, exp_dir)
    plot_pca_eigenvalues(results_by_window, exp_dir)
    plot_summary_panel(T_list, acc_list, results_by_window, config, exp_dir)
    plot_true_vs_pred(results_by_window, config, exp_dir)

    best_idx = int(np.argmax(acc_list))
    best_T   = T_list[best_idx]
    best_acc = acc_list[best_idx]
    print(f"\nЛучший результат: accuracy = {best_acc:.4f} при Tmax = {best_T:.3f}")
    print(f"Результаты сохранены в: {exp_dir}")

    return best_acc, T_list, acc_list


###############################################################################
# Точки входа
###############################################################################

def run_boundary_sensing():
    config = create_boundary_config()
    print("\n" + "=" * 70)
    print("REMOTE BOUNDARY SENSING (Секция IV, Fig. 2)")
    print(f"Классификация: Full Bond / No Bond / Signal")
    print(f"Времён: {config.measurement.Tmin}–{config.measurement.Tmax}, "
          f"окон: {len(config.measurement.measurement_windows)-1}")
    print(f"N_tom: {config.measurement.N_tom:.0e}")
    print("=" * 70)

    exp_dir = setup_directories("results", "boundary_sensing")
    with open(os.path.join(exp_dir, "config.txt"), "w") as f:
        f.write(f"Experiment: boundary_sensing\n")
        for k, v in vars(config.physics).items():
            f.write(f"{k}: {v}\n")
        for k, v in vars(config.measurement).items():
            f.write(f"{k}: {v}\n")

    best_acc, T_list, acc_list = run_experiment(config, exp_dir)
    print("\n" + "=" * 70)
    print(f"ФИНАЛЬНЫЙ РЕЗУЛЬТАТ: Лучшая точность = {best_acc:.4f}")
    print("=" * 70)
    return best_acc


def run_thermometry():
    config = create_thermometry_config()
    print("\n" + "=" * 70)
    print("THERMOMETRY (Секция V, Fig. 3)")
    print(f"Регрессия: T от 0.9·T_mean до 1.1·T_mean")
    print(f"T_mean = {config.physics.TMean:.6f}")
    print(f"Времён: {config.measurement.Tmin}–{config.measurement.Tmax}")
    print(f"Heisenberg time = {1.0/config.physics.wD:.3f}")
    print(f"N_tom: {config.measurement.N_tom:.0e}")
    print("=" * 70)

    exp_dir = setup_directories("results", "thermometry")
    with open(os.path.join(exp_dir, "config.txt"), "w") as f:
        f.write(f"Experiment: thermometry\n")
        for k, v in vars(config.physics).items():
            f.write(f"{k}: {v}\n")
        for k, v in vars(config.measurement).items():
            f.write(f"{k}: {v}\n")

    best_acc, T_list, acc_list = run_experiment(config, exp_dir)
    print("\n" + "=" * 70)
    print(f"ФИНАЛЬНЫЙ РЕЗУЛЬТАТ: Лучшая точность (±1%) = {best_acc:.4f}")
    print("=" * 70)
    return best_acc


def run_test():
    config = create_boundary_config()
    config.measurement.N_tom   = 1e4
    config.measurement.dt      = 2.0
    config.measurement.Tmin    = 2.0
    config.measurement.Tmax    = 14.0
    config.ml.N_samples        = 300
    config.ml.n_epochs         = 200

    print("\n" + "=" * 70)
    print("ТЕСТОВЫЙ ЗАПУСК")
    print(f"N_tom={config.measurement.N_tom:.0e}, N_samples={config.ml.N_samples}, "
          f"N_epochs={config.ml.n_epochs}")
    print(f"Окон: {len(config.measurement.measurement_windows)-1}")
    print("=" * 70)

    exp_dir = setup_directories("results", "test")
    best_acc, T_list, acc_list = run_experiment(config, exp_dir)
    print("\n" + "=" * 70)
    print(f"РЕЗУЛЬТАТ: Лучшая точность = {best_acc:.4f}")
    print("=" * 70)
    return best_acc


def main():
    parser = argparse.ArgumentParser(
        description="Decoding QFT with ML — воспроизведение экспериментов"
    )
    parser.add_argument(
        "mode",
        choices=["boundary", "thermo", "test"],
        help="Режим: boundary, thermo, test",
    )
    args = parser.parse_args()

    if args.mode == "boundary":
        run_boundary_sensing()
    elif args.mode == "thermo":
        run_thermometry()
    elif args.mode == "test":
        run_test()


if __name__ == "__main__":
    main()