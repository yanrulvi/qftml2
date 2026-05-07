# data/preprocess.py
"""
Предобработка данных: PCA, whitening, разделение на выборки.

Точно воспроизводит логику из оригинального DataGenerator.py:
1. Центрирование данных (вычитание среднего)
2. PCA через сингулярное разложение ковариационной матрицы
3. Whitening (деление на sqrt(λ))
4. Опциональное сжатие размерности

Формулы из Appendix A:
- Ковариационная матрица: (1/(n-1)) X_trainᵀ X_train = Vᵀ Λ V
- PCA: X → V X
- Whitening: X → Λ^(-1/2) X
"""

import numpy as np
from typing import Tuple, Optional


def compute_pca(
    X_train: np.ndarray,
    var_keep: float = 1.0,
    remove_low_variance: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Выполняет PCA на тренировочных данных.

    В точности воспроизводит функцию PCA() из Utilities.py.

    Args:
        X_train: тренировочные данные размером (n_samples, n_features)
        var_keep: доля сохраняемой дисперсии (0 < var_keep <= 1 или 'All')
        remove_low_variance: удалять ли компоненты с очень низкой дисперсией

    Returns:
        X_mean: среднее тренировочных данных (n_features,)
        eigenvalues: собственные значения (n_features,)
        eigenvectors: собственные векторы (n_features, n_features) — строки!
        n_components: число сохраняемых компонент
    """
    n, d = X_train.shape

    # 1. Центрирование
    X_mean = np.mean(X_train, axis=0).real
    X_centered = X_train - np.tile(X_mean, (n, 1))

    # 2. Ковариационная матрица
    Cov = (X_centered.T @ X_centered) / (n - 1)

    # 3. Сингулярное разложение
    eigenvalues, eigenvectors = np.linalg.eig(Cov)

    # Берём реальные части
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    # 4. Сортировка по убыванию собственных значений
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    eigenvectors = eigenvectors.T  # Теперь строки — собственные векторы

    # 5. Корректировка отрицательных собственных значений
    for k in range(1, len(eigenvalues)):
        if eigenvalues[k] < 0:
            eigenvalues[k] = eigenvalues[k - 1] / 10.0

    # 6. Определение числа компонент для сохранения
    if var_keep == "All" or var_keep >= 1.0:
        n_components = d
    elif var_keep > 0:
        sum_lam = eigenvalues.sum()
        cumsum = 0.0
        n_components = 0
        for dim_c in range(len(eigenvalues)):
            cumsum += eigenvalues[dim_c]
            n_components += 1
            if cumsum / sum_lam > var_keep:
                break
    else:
        n_components = d

    # 7. Удаление компонент с экстремально низкой дисперсией
    if remove_low_variance and var_keep >= 1.0:
        # Удаляем компоненты, где дисперсия меньше 10^(-10) от максимальной
        threshold = eigenvalues[0] * 1e-10
        n_components = np.sum(eigenvalues > threshold)

    # Минимальное число компонент
    n_components = max(n_components, 2)

    return X_mean, eigenvalues, eigenvectors, n_components


def apply_pca(
    X: np.ndarray,
    X_mean: np.ndarray,
    eigenvectors: np.ndarray,
    eigenvalues: np.ndarray,
    n_components: int,
) -> np.ndarray:
    """
    Применяет PCA + whitening к данным, используя параметры, вычисленные на train.

    Формулы (Appendix A):
    1. X_centered = X - X_mean
    2. X_pca = X_centered @ Mᵀ (где M — матрица собственных векторов-строк)
    3. X_whitened = X_pca / sqrt(λ)

    Args:
        X: данные для трансформации (n_samples, n_features)
        X_mean: среднее тренировочных данных
        eigenvectors: матрица собственных векторов-строк (n_features, n_features)
        eigenvalues: собственные значения
        n_components: число сохраняемых компонент

    Returns:
        Трансформированные данные (n_samples, n_components)
    """
    # Центрирование
    X_centered = X - np.tile(X_mean, (X.shape[0], 1))

    # PCA: проекция на собственные векторы
    # eigenvectors — строки, поэтому умножаем на транспонированную матрицу
    X_pca = np.real(X_centered @ eigenvectors[:n_components].T)

    # Whitening: деление на sqrt(λ)
    X_whitened = X_pca / np.sqrt(eigenvalues[:n_components])

    return X_whitened


def split_and_shuffle_data(
    X: np.ndarray,
    y: np.ndarray,
    f_train: float = 0.75,
    f_valid: float = 0.25,
    f_test: float = 0.0,
    random_seed: int = 42,
) -> Tuple:
    """
    Перемешивает данные и разделяет на train/valid/test.

    Из оригинального кода:
    permut = np.arange(N)
    np.random.shuffle(permut)
    X = X[permut, :]
    y = y[permut]

    Args:
        X: признаки (n_samples, n_features)
        y: метки (n_samples,)
        f_train, f_valid, f_test: доли выборок
        random_seed: зерно для воспроизводимости

    Returns:
        (X_train, y_train, X_valid, y_valid, X_test, y_test) — если test есть
        (X_train, y_train, X_valid, y_valid) — если test == 0
    """
    np.random.seed(random_seed)

    n = X.shape[0]

    # Случайная перестановка
    permut = np.arange(n)
    np.random.shuffle(permut)
    X = X[permut]
    y = y[permut]

    # Разделение
    n_train = int(f_train * n)
    n_valid = int(f_valid * n)

    X_train = X[:n_train]
    y_train = y[:n_train]

    X_valid = X[n_train : n_train + n_valid]
    y_valid = y[n_train : n_train + n_valid]

    if f_test > 0:
        X_test = X[n_train + n_valid :]
        y_test = y[n_train + n_valid :]
        return X_train, y_train, X_valid, y_valid, X_test, y_test
    else:
        return X_train, y_train, X_valid, y_valid


def preprocess_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    var_keep: float = 1.0,
    is_regression: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Полный пайплайн предобработки: PCA + whitening.

    Из оригинального DataGenerator.py:
    1. PCA на тренировочных данных
    2. Применение трансформации к train и valid
    3. Опциональное сжатие размерности

    Returns:
        X_train_pca, y_train, X_valid_pca, y_valid, pca_params
        где pca_params = {'mean': ..., 'eigenvectors': ..., 'eigenvalues': ..., 'n_components': ...}
    """
    # PCA на тренировочных данных
    X_mean, eigenvalues, eigenvectors, n_components = compute_pca(
        X_train, var_keep=var_keep
    )

    print(
        f"PCA: {X_train.shape[1]} → {n_components} компонент "
        f"(сохранено {100 * eigenvalues[:n_components].sum() / eigenvalues.sum():.1f}% дисперсии)"
    )

    # Применяем PCA к train и valid
    X_train_pca = apply_pca(
        X_train, X_mean, eigenvectors, eigenvalues, n_components
    )
    X_valid_pca = apply_pca(
        X_valid, X_mean, eigenvectors, eigenvalues, n_components
    )

    pca_params = {
        "mean": X_mean,
        "eigenvectors": eigenvectors,
        "eigenvalues": eigenvalues,
        "n_components": n_components,
    }

    return X_train_pca, y_train, X_valid_pca, y_valid, pca_params


def normalize_labels_for_regression(
    y: np.ndarray, y_train: np.ndarray | None = None
) -> Tuple[np.ndarray, float, float]:
    """
    Нормализует метки для регрессии: y → (y - min)/(max - min) * 0.5 + 0.25

    Из оригинального кода (DataGenerator.py):
    reglist = (reglist - minr) / (maxr - minr)
    reglist = 0.5 * reglist + 0.25

    Это отображает значения в диапазон [0.25, 0.75], что подходит для сигмоиды на выходе.

    Args:
        y: метки для нормализации
        y_train: тренировочные метки (для вычисления min/max). Если None, используется y

    Returns:
        y_norm, y_min, y_max
    """
    if y_train is not None:
        y_min = y_train.min()
        y_max = y_train.max()
    else:
        y_min = y.min()
        y_max = y.max()

    y_norm = (y - y_min) / (y_max - y_min)
    y_norm = 0.5 * y_norm + 0.25

    return y_norm, y_min, y_max


def denormalize_predictions(
    y_norm: np.ndarray, y_min: float, y_max: float
) -> np.ndarray:
    """
    Обратное преобразование для нормализованных предсказаний.

    Из оригинального кода:
    temp = (y_norm - 0.25) * 2 * (maxr - minr) + minr
    """
    return (y_norm - 0.25) * 2.0 * (y_max - y_min) + y_min
