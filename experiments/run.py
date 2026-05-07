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


def setup_directories(base_dir: str, experiment_name: str) -> str:
    """Создаёт структуру директорий для эксперимента."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)

    return exp_dir


def generate_window_data(
    evolution: FieldEvolution,
    config: ExperimentConfig,
    Tmin: float,
    Tmax: float,
    window_idx: int,
    exp_dir: str,
) -> tuple:
    """Генерирует данные для одного временного окна."""
    print(f"  Window {window_idx}: Tmin={Tmin:.3f}, Tmax={Tmax:.3f}")

    X = evolution.generate_data_for_window(Tmin, Tmax)

    N_features = X.shape[1] - 1
    X_data = X[:, :N_features]
    y_data = X[:, N_features]

    window_df = pd.DataFrame(X)
    window_df.to_csv(
        os.path.join(exp_dir, "data", f"window_{window_idx:03d}.csv"),
        header=False,
        index=False,
    )

    return X_data, y_data


def process_window(
    X: np.ndarray, y: np.ndarray, config: ExperimentConfig, window_Tmax: float
) -> dict:
    """Обрабатывает одно временное окно: PCA + обучение."""
    import torch  # Добавь в начало функции

    # Определяем устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Разделение данных
    X_train, y_train, X_valid, y_valid = split_and_shuffle_data(
        X,
        y,
        f_train=config.ml.f_train,
        f_valid=config.ml.f_valid,
        f_test=config.ml.f_test,
        random_seed=42,
    )

    print(f"    Train: {X_train.shape[0]}, Valid: {X_valid.shape[0]}")
    print(
        f"    Train y unique: {len(np.unique(y_train))}, Valid y unique: {len(np.unique(y_valid))}"
    )

    # Нормализация для регрессии
    y_min, y_max = None, None
    if config.is_regression:
        y_train_norm, y_min, y_max = normalize_labels_for_regression(y_train)
        y_valid_norm, _, _ = normalize_labels_for_regression(
            y_valid, y_train=y_train
        )
        print(
            f"    y_train_norm: [{y_train_norm.min():.3f}, {y_train_norm.max():.3f}]"
        )
        print(
            f"    y_valid_norm: [{y_valid_norm.min():.3f}, {y_valid_norm.max():.3f}]"
        )
    else:
        y_train_norm = y_train.copy()
        y_valid_norm = y_valid.copy()

    # PCA
    X_train_pca, y_train_final, X_valid_pca, y_valid_final, pca_params = (
        preprocess_pipeline(
            X_train,
            y_train_norm,
            X_valid,
            y_valid_norm,
            var_keep=config.ml.pca_var_keep,
            is_regression=config.is_regression,
        )
    )

    n_features = X_train_pca.shape[1]

    # Модель
    model = create_model(
        n_features=n_features,
        n_output=config.n_classes,
        is_regression=config.is_regression,
        n_hidden=config.ml.n_hidden,
    )

    # Обучение
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

    # === ОТЛАДОЧНЫЙ ВЫВОД ДЛЯ РЕГРЕССИИ ===
    if config.is_regression:
        model.eval()
        # Переносим данные на то же устройство, что и модель
        X_valid_t = torch.FloatTensor(X_valid_pca).to(device)
        with torch.no_grad():
            preds = model(X_valid_t).squeeze().cpu().numpy()

        print(
            f"    [DEBUG] Raw preds: min={preds.min():.4f}, max={preds.max():.4f}, mean={preds.mean():.4f}"
        )
        print(
            f"    [DEBUG] Raw targets: min={y_valid_final.min():.4f}, max={y_valid_final.max():.4f}, mean={y_valid_final.mean():.4f}"
        )

        # Денормализация
        if y_min is not None and y_max is not None:
            preds_denorm = (preds - 0.25) * 2.0 * (y_max - y_min) + y_min
            targets_denorm = (y_valid_final - 0.25) * 2.0 * (
                y_max - y_min
            ) + y_min
            print(
                f"    [DEBUG] Denorm preds: min={preds_denorm.min():.4f}, max={preds_denorm.max():.4f}, mean={preds_denorm.mean():.4f}"
            )
            print(
                f"    [DEBUG] Denorm targets: min={targets_denorm.min():.4f}, max={targets_denorm.max():.4f}, mean={targets_denorm.mean():.4f}"
            )
            print(f"    [DEBUG] First 5 preds:  {preds_denorm[:5]}")
            print(f"    [DEBUG] First 5 targets: {targets_denorm[:5]}")

            mae = np.abs(preds_denorm - targets_denorm).mean()
            print(f"    [DEBUG] MAE (денорм): {mae:.6f}")

    # Лучшая точность
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
    }


def run_experiment(config: ExperimentConfig, exp_dir: str):
    """Полный пайплайн с обработкой по временным окнам."""
    print("=" * 70)
    print(f"Эксперимент: {config.mode.value}")
    print(f'Режим: {"Регрессия" if config.is_regression else "Классификация"}')
    print(f"Случаев: {len(config.cases)}")
    print(f"N_tom: {config.measurement.N_tom:.0e}")
    print(f"N_samples на случай: {config.ml.N_samples}")
    print(
        f"Архитектура: [n_features] → {config.ml.n_hidden} → {config.n_classes}"
    )
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

        X, y = generate_window_data(
            evolution, config, Tmin, Tmax, t_idx, exp_dir
        )
        result = process_window(X, y, config, Tmax)
        results_by_window.append(result)

    # Сводная таблица
    print("\n" + "=" * 70)
    print("СВОДКА ПО ВРЕМЕННЫМ ОКНАМ")
    print("=" * 70)
    print(f'{"Tmax":>8s}  {"Accuracy":>10s}  {"Features":>8s}')
    print("-" * 30)

    T_list = []
    acc_list = []

    for r in results_by_window:
        print(
            f'{r["Tmax"]:8.1f}  {r["best_valid_acc"]:10.4f}  {r["n_features"]:8d}'
        )
        T_list.append(r["Tmax"])
        acc_list.append(r["best_valid_acc"])

    # График
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(T_list, acc_list, "o-", linewidth=2, markersize=8, color="blue")

    if config.mode == ExperimentMode.BOUNDARY_SENSING:
        light_time = config.physics.LatticeLength
        ax.axvline(
            x=light_time,
            color="red",
            linestyle="--",
            alpha=0.5,
            label=f"Light crossing time ≈ {light_time:.0f}",
        )
        ax.legend()
        ax.set_title(
            "Boundary Sensing — Accuracy vs Measurement Time (Fig. 2)"
        )
    elif config.mode == ExperimentMode.THERMOMETRY:
        heisenberg_time = 1.0 / config.physics.wD
        ax.axvline(
            x=heisenberg_time,
            color="red",
            linestyle="--",
            alpha=0.5,
            label=f"Heisenberg time = {heisenberg_time:.3f}",
        )
        ax.legend()
        ax.set_title("Thermometry — Accuracy vs Measurement Time (Fig. 3)")

    ax.set_xlabel("Measurement time Tmax")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(exp_dir, "plots", "accuracy_vs_time.pdf"), dpi=150
    )
    plt.close()

    df_summary = pd.DataFrame({"Tmax": T_list, "best_valid_acc": acc_list})
    df_summary.to_csv(
        os.path.join(exp_dir, "accuracy_vs_time.csv"), index=False
    )

    best_idx = np.argmax(acc_list)
    best_T = T_list[best_idx]
    best_acc = acc_list[best_idx]

    print(
        f"\nЛучший результат: accuracy = {best_acc:.4f} при Tmax = {best_T:.1f}"
    )
    print(f"Результаты сохранены в: {exp_dir}")

    return best_acc, T_list, acc_list


def run_boundary_sensing():
    """Boundary Sensing (Секция IV, Fig. 2)"""
    config = create_boundary_config()

    print("\n" + "=" * 70)
    print("REMOTE BOUNDARY SENSING (Секция IV, Fig. 2)")
    print(f"Классификация: Full Bond / No Bond / Signal")
    print(
        f"Времён: {config.measurement.Tmin}–{config.measurement.Tmax}, "
        f"окон: {len(config.measurement.measurement_windows)-1}"
    )
    print(f"N_tom: {config.measurement.N_tom:.0e}")
    print("=" * 70)

    exp_dir = setup_directories("results", "boundary_sensing")

    with open(os.path.join(exp_dir, "config.txt"), "w") as f:
        f.write(f"Experiment: boundary_sensing\n")
        f.write(f"LatticeLength: {config.physics.LatticeLength}\n")
        f.write(f"mcc: {config.physics.mcc}\n")
        f.write(f"wD: {config.physics.wD}\n")
        f.write(f"lam: {config.physics.lam}\n")
        f.write(f"Tmin: {config.measurement.Tmin}\n")
        f.write(f"Tmax: {config.measurement.Tmax}\n")
        f.write(f"dt: {config.measurement.dt}\n")
        f.write(f"N_tom: {config.measurement.N_tom:.0e}\n")
        f.write(f"N_samples: {config.ml.N_samples}\n")
        f.write(f"N_epochs: {config.ml.n_epochs}\n")

    best_acc, T_list, acc_list = run_experiment(config, exp_dir)

    print("\n" + "=" * 70)
    print(f"ФИНАЛЬНЫЙ РЕЗУЛЬТАТ: Лучшая точность = {best_acc:.4f}")
    print("=" * 70)

    return best_acc


def run_thermometry():
    """Thermometry (Секция V, Fig. 3)"""
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
        f.write(f"T_mean: {config.physics.TMean:.6f}\n")
        f.write(f"T_dev: {config.physics.TDev:.6e}\n")
        f.write(f"Tmin: {config.measurement.Tmin}\n")
        f.write(f"Tmax: {config.measurement.Tmax}\n")
        f.write(f"dt: {config.measurement.dt}\n")
        f.write(f"N_tom: {config.measurement.N_tom:.0e}\n")
        f.write(f"N_samples: {config.ml.N_samples}\n")
        f.write(f"N_epochs: {config.ml.n_epochs}\n")
        f.write(f"Heisenberg time: {1.0/config.physics.wD:.3f}\n")

    best_acc, T_list, acc_list = run_experiment(config, exp_dir)

    print("\n" + "=" * 70)
    print(f"ФИНАЛЬНЫЙ РЕЗУЛЬТАТ: Лучшая точность (±1%) = {best_acc:.4f}")
    print("=" * 70)

    return best_acc


def run_test():
    """Быстрый тест"""
    config = create_boundary_config()
    config.measurement.N_tom = 1e4
    config.measurement.dt = 2.0
    config.measurement.Tmin = 2.0
    config.measurement.Tmax = 14.0
    config.ml.N_samples = 300
    config.ml.n_epochs = 200

    print("\n" + "=" * 70)
    print("ТЕСТОВЫЙ ЗАПУСК")
    print(
        f"N_tom={config.measurement.N_tom:.0e}, N_samples={config.ml.N_samples}, "
        f"N_epochs={config.ml.n_epochs}"
    )
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
        help="Режим: boundary (boundary sensing), thermo (thermometry), test (быстрый тест)",
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
