# physics/evolution.py
"""
Симплектическая эволюция и построение проекторов для измерений.

Реализует:
- Симплектическую форму Ω (multOmega)
- Унитарную эволюцию U(t) = exp(ΩF t) (DefProjList)
- Проекторы на квадратуры детектора q₀, p₀, r₀
- Вычисление точных средних значений (траекторий)

Основано на оригинальных функциях:
- multOmega()
- DefProjList()
- Sexp() (для справки, но мы используем expm)
"""

import numpy as np
from scipy.linalg import expm
from typing import List, Tuple, Optional


def symplectic_form(dim: int) -> np.ndarray:
    """
    Строит симплектическую форму Ω.

    Из оригинального кода (multOmega):
    Для матрицы sigma размером (2p, 2p):
    Ω·sigma = [sigma[p:2p, 0:p],           sigma[p:2p, p:2p]
               [-sigma[0:p, 0:p],          -sigma[0:p, p:2p]]

    Это соответствует Ω = [[0, I], [-I, 0]].

    Args:
        dim: полная размерность фазового пространства (чётная)

    Returns:
        Матрица Ω размером (dim, dim)
    """
    p = dim // 2
    Omega = np.zeros((dim, dim))
    Omega[:p, p:] = np.eye(p)
    Omega[p:, :p] = -np.eye(p)
    return Omega


def mult_omega(matrix: np.ndarray) -> np.ndarray:
    """
    Умножает симплектическую форму Ω на матрицу.

    Точно воспроизводит оригинальную функцию multOmega().

    Для матрицы sigma размером (2p, 2p):
    res[:p, :p] = sigma[p:2p, :p]
    res[:p, p:2p] = sigma[p:2p, p:2p]
    res[p:2p, :p] = -sigma[:p, :p]
    res[p:2p, p:2p] = -sigma[:p, p:2p]
    """
    dim = matrix.shape[0]
    p = dim // 2

    res = np.zeros((dim, dim))
    res[:p, :p] = matrix[p:, :p]
    res[:p, p:] = matrix[p:, p:]
    res[p:, :p] = -matrix[:p, :p]
    res[p:, p:] = -matrix[:p, p:]

    return res


def compute_unitaries(F: np.ndarray, t_list: np.ndarray) -> List[np.ndarray]:
    """
    Вычисляет унитарные операторы эволюции U(t) = exp(ΩF t).

    Из оригинального кода: DefProjList, шаг вычисления Ulist.

    Args:
        F: матрица квадратичного гамильтониана (H = ½XᵀFX)
        t_list: список времён для вычисления унитарных операторов

    Returns:
        Список матриц U(t) для каждого времени
    """
    Omega_F = mult_omega(F)
    U_list = []

    for t in t_list:
        U_t = expm(Omega_F * t)
        U_list.append(U_t)

    return U_list


def compute_unitaries_stepped(
    F: np.ndarray, N_times: int, Tmin: float, Tmax: float
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Вычисляет унитарные операторы по шагам, как в оригинальном коде.

    Из DefProjList:
    - Создаёт равномерную сетку времён от Tmin до Tmax
    - Вычисляет U_step = exp(ΩF · dT)
    - Последовательно умножает: U[k] = U_step @ U[k-1]
    - U[0] = exp(ΩF · Tmin)

    Args:
        F: матрица гамильтониана
        N_times: число шагов по времени
        Tmin: начальное время
        Tmax: конечное время

    Returns:
        (U_list, U_adj_list) — унитарные операторы и их транспонированные версии
    """
    dim = F.shape[0]

    # Сетка времён
    t_array = np.linspace(Tmin, Tmax, N_times)
    dT = t_array[1] - t_array[0] if N_times > 1 else 0

    Omega_F = mult_omega(F)

    U_list = []
    U_adj_list = []

    # Первый шаг
    U_0 = expm(Omega_F * Tmin)
    U_list.append(U_0)
    U_adj_list.append(U_0.T)

    if N_times > 1:
        U_step = expm(Omega_F * dT)
        for k in range(1, N_times):
            U_k = U_step @ U_list[k - 1]
            U_list.append(U_k)
            U_adj_list.append(U_k.T)

    return U_list, U_adj_list


def build_projectors(N_total: int) -> List[np.ndarray]:
    """
    Строит проекторы на квадратуры детектора.

    Из оригинального кода (DefProjList):

    m = N_total (поле + детектор)
    Фазовый порядок: (q₁, ..., q_N, q_det, p₁, ..., p_N, p_det)

    Индексы детектора:
    - q_det: m-1 (последний среди q)
    - p_det: 2*m-1 (последний среди p)

    Проекторы:
    - P0: проектор на q_det
    - P2: проектор на p_det
    - P1: проектор на r_det = (q_det + p_det)/√2

    P1 строится как:
    P1[m-1, 2*m-1] = -1/2
    P1[2*m-1, m-1] = -1/2
    P1 += (P0 + P2)/2

    Это даёт правильное среднее для r̂ = (q̂ + p̂)/√2:
    ⟨r̂⟩ = Tr[P1 · ρ]
    """
    m = N_total  # m = количество осцилляторов (поле + детектор)
    dim = 2 * m

    P0 = np.zeros((dim, dim))  # Проектор на q_det
    P2 = np.zeros((dim, dim))  # Проектор на p_det
    P1 = np.zeros((dim, dim))  # Проектор на r_det

    # Индексы детектора
    q_idx = m - 1  # Последний q
    p_idx = 2 * m - 1  # Последний p

    P0[q_idx, q_idx] = 1.0
    P2[p_idx, p_idx] = 1.0

    # r = (q + p)/√2
    # Для проектора: r̂² требует перекрёстных членов
    # Формула из кода авторов:
    P1[q_idx, p_idx] = -0.5
    P1[p_idx, q_idx] = -0.5
    P1 += (P0 + P2) / 2.0

    return [P0, P1, P2]


def evolve_projectors(
    U_list: List[np.ndarray], U_adj_list: List[np.ndarray]
) -> np.ndarray:
    """
    Вычисляет эволюцию проекторов: P(t) = U†(t) P U(t).

    Из оригинального кода (DefProjList):
    ProjList[n, r] = U_adj[n] @ Proj0[r] @ U[n]

    Args:
        U_list: список унитарных операторов U(t)
        U_adj_list: список U†(t) = U(t)ᵀ (вещественный случай)

    Returns:
        Массив размером (N_times, 3, dim, dim)
        ProjList[t, 0] — проектор на q_det во время t
        ProjList[t, 1] — проектор на r_det во время t
        ProjList[t, 2] — проектор на p_det во время t
    """
    dim = U_list[0].shape[0]
    N_times = len(U_list)

    # Строим базовые проекторы
    N_total = dim // 2
    Proj0 = build_projectors(N_total)

    ProjList = np.zeros((N_times, 3, dim, dim))

    for n in range(N_times):
        for r in range(3):
            ProjList[n, r] = U_adj_list[n] @ Proj0[r] @ U_list[n]

    return ProjList


def compute_exact_trajectory(
    ProjList: np.ndarray, sigma_0: np.ndarray
) -> np.ndarray:
    """
    Вычисляет точные средние значения квадратур детектора.

    Формула: ⟨P(t)⟩ = Tr[P(t) · σ₀]
    где P(t) — эволюционировавший проектор, σ₀ — начальная ковариационная матрица.

    Возвращает массив размером (N_times, 3):
    - [t, 0] = ⟨q_det(t)⟩
    - [t, 1] = ⟨r_det(t)⟩
    - [t, 2] = ⟨p_det(t)⟩
    """
    N_times = ProjList.shape[0]
    trajectory = np.zeros((N_times, 3))

    for n in range(N_times):
        for r in range(3):
            trajectory[n, r] = np.trace(ProjList[n, r] @ sigma_0).real

    return trajectory


def compute_median_trajectory_optimized(
    ProjList_list: List[np.ndarray],
    sigma_0_list: List[np.ndarray],
    labels: List[int],
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Вычисляет траектории с оптимизацией через медианные значения.

    Из оригинального кода:
    1. Вычисляет MedProj — медианный проектор по всем случаям
    2. Вычисляет MedRSE0 — медианное начальное состояние
    3. Для каждого случая:
       dP = Proj - MedProj
       dS = RSE0 - MedRSE0
       aS = Tr[dP @ MedRSE0] + Tr[MedProj @ dS] + Tr[dP @ dS]

    Это позволяет избежать повторного вычисления Tr[MedProj @ MedRSE0].

    Args:
        ProjList_list: список проекторов для каждого случая
        sigma_0_list: список начальных ковариационных матриц
        labels: метки случаев (для группировки)

    Returns:
        (med_trajectory, case_trajectories)
    """
    N_cases = len(ProjList_list)
    N_times = ProjList_list[0].shape[0]

    # Медианные значения
    MedProj = np.median(np.array(ProjList_list), axis=0)
    MedRSE0 = np.median(np.array(sigma_0_list), axis=0)

    # Медианная траектория
    med_traj = np.zeros((N_times, 3))
    for n in range(N_times):
        for r in range(3):
            med_traj[n, r] = np.trace(MedProj[n, r] @ MedRSE0).real
    med_traj = med_traj.flatten()

    # Траектории для каждого случая
    case_trajs = []
    for k in range(N_cases):
        dP = ProjList_list[k] - MedProj
        dS = sigma_0_list[k] - MedRSE0

        aS = np.zeros((N_times, 3))
        for n in range(N_times):
            for r in range(3):
                # Три слагаемых
                aS[n, r] += np.trace(dP[n, r] @ MedRSE0).real
                aS[n, r] += np.trace(MedProj[n, r] @ dS).real
                aS[n, r] += np.trace(dP[n, r] @ dS).real

        case_trajs.append(aS.flatten())

    return med_traj, case_trajs


def apply_tomography_noise(
    trajectory: np.ndarray, med_trajectory: np.ndarray, N_tom: float
) -> np.ndarray:
    """
    Добавляет томографический шум к точной траектории.

    ТОЧНО воспроизводит оригинальную функцию Tomography():

    Для N_tom == 'Infinity': возвращает точные значения
    Для N_tom <= 10:
        a_tom = (Med + a) * chi2(N_tom-1) / (N_tom-1) - Med
    Для N_tom > 10:
        a_tom = a + (a + Med) * random.normal(0, sqrt(2/(N_tom-1)))

    где a — точное значение траектории, Med — медианное значение
    """
    if N_tom == float("inf") or N_tom >= 1e15:
        return trajectory.copy()

    N_tom = float(N_tom)
    result = np.zeros_like(trajectory)

    if N_tom <= 10:
        # Точное хи-квадрат распределение
        chi2_sample = np.random.chisquare(N_tom - 1)
        for i in range(len(trajectory)):
            result[i] = (med_trajectory[i] + trajectory[i]) * chi2_sample / (
                N_tom - 1
            ) - med_trajectory[i]
    else:
        # Гауссово приближение по ЦПТ
        for i in range(len(trajectory)):
            noise = np.random.randn() * np.sqrt(2.0 / (N_tom - 1))
            result[i] = (
                trajectory[i] + (trajectory[i] + med_trajectory[i]) * noise
            )

    return result


class FieldEvolution:
    """
    Класс для управления эволюцией системы поле + детектор.

    Объединяет:
    - Построение гамильтонианов для всех случаев
    - Вычисление унитарных операторов
    - Вычисление проекторов
    - Расчёт точных траекторий
    - Добавление томографического шума
    """

    def __init__(self, config):
        """
        Args:
            config: ExperimentConfig
        """
        from physics.lattice import LatticeField

        self.config = config
        self.physics = config.physics
        self.measurement = config.measurement

        # Создаём решёточное поле
        self.lattice = LatticeField(
            n_modes=self.physics.N_modes,
            mass=self.physics.mcc,
            lattice_spacing=self.physics.a,
        )

        # Получаем безразмерные параметры
        norm = self.physics.get_normalized_params()
        self.sigma_norm = norm["sigma_norm"]
        self.mcc_norm = norm["mcc_norm"]
        self.wD_norm = norm["wD_norm"]
        self.lam_norm = norm["lam_norm"]

        # Строим гамильтонианы для каждого случая
        self.H_dynamic = []
        self.H_thermal = []

        for case in config.cases:
            F_dyn, F_th = self.lattice.build_full_hamiltonian_matrix(
                boundary_type=case.boundary_type,
                detector_distance=case.distance,
                omega_D=self.wD_norm,
                lam=self.lam_norm,
                sigma=self.sigma_norm,
            )
            self.H_dynamic.append(F_dyn)
            self.H_thermal.append(F_th)

    def compute_projectors_for_window(
        self, Tmin: float, Tmax: float
    ) -> List[np.ndarray]:
        """
        Вычисляет проекторы для одного временного окна [Tmin, Tmax].

        Соответствует вызову DefProjList для каждого случая.
        """
        N_times = self.measurement.N_times
        ProjList_cases = []

        for k, F in enumerate(self.H_dynamic):
            U_list, U_adj_list = compute_unitaries_stepped(
                F, N_times, Tmin, Tmax
            )
            proj = evolve_projectors(U_list, U_adj_list)
            ProjList_cases.append(proj)

        return ProjList_cases

    def generate_data_for_window(
        self, Tmin: float, Tmax: float, return_exact: bool = False
    ) -> np.ndarray:
        """
        Генерирует данные для одного временного окна.

        ИСПРАВЛЕНИЕ ДЛЯ ТЕРМОМЕТРИИ:
        Для регрессии каждая температура генерируется N_samples/pop раз,
        но метка — СРЕДНЯЯ температура бина (не случайная).
        Томографический шум создаёт разброс вокруг точного значения.
        """
        N_times = self.measurement.N_times
        N_tom = self.measurement.N_tom
        N_samples_total = self.config.ml.N_samples
        cases = self.config.cases
        N_cases = len(cases)

        # 1. Проекторы
        ProjList_cases = self.compute_projectors_for_window(Tmin, Tmax)

        # 2. Начальные состояния
        sigma_0_list = []
        for k, case in enumerate(cases):
            is_signal = case.boundary_type == 3
            sigma_0 = self.lattice.get_initial_covariance(
                F_thermal=self.H_thermal[k],
                temperature=case.temperature,
                is_signal=is_signal,
                Gsignal=self.physics.Gsignal,
            )
            sigma_0_list.append(sigma_0)

        # 3. Медианные значения и точные траектории
        MedProj = np.median(np.array(ProjList_cases), axis=0)
        MedRSE0 = np.median(np.array(sigma_0_list), axis=0)

        # Вычисляем точные траектории
        med_traj, case_trajs = compute_median_trajectory_optimized(
            ProjList_cases, sigma_0_list, [c.y for c in cases]
        )

        N_features = 3 * N_times

        if self.config.is_regression:
            # ДЛЯ РЕГРЕССИИ: генерируем N_samples_total на каждый бин,
            # но метка = средняя температура бина (НЕ случайная!)
            # Разброс создаётся только за счёт томографического шума

            total_samples = N_cases * N_samples_total
            ExpData = np.zeros((total_samples, N_features + 1))

            # Нормализуем температуры для меток
            temps = np.array([c.temperature for c in cases])
            min_T = temps.min()
            max_T = temps.max()

            for c in range(N_cases):
                # Точная траектория для этого случая
                aS_base = case_trajs[c]

                # Нормализованная метка
                T_norm = (temps[c] - min_T) / (max_T - min_T)
                T_norm = 0.5 * T_norm + 0.25

                for s in range(N_samples_total):
                    idx = c * N_samples_total + s

                    # Добавляем томографический шум
                    aS_tom = apply_tomography_noise(aS_base, med_traj, N_tom)

                    ExpData[idx, :N_features] = aS_tom
                    ExpData[idx, N_features] = T_norm

            return ExpData

        else:
            # ДЛЯ КЛАССИФИКАЦИИ: как раньше
            total_samples = N_cases * N_samples_total
            ExpData = np.zeros((total_samples, N_features + 1))

            for c in range(N_cases):
                aS_base = case_trajs[c]
                for s in range(N_samples_total):
                    idx = c * N_samples_total + s
                    aS_tom = apply_tomography_noise(aS_base, med_traj, N_tom)
                    ExpData[idx, :N_features] = aS_tom
                    ExpData[idx, N_features] = cases[c].y

            return ExpData

    def generate_all_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Генерирует данные для всех измерительных окон.

        Returns:
            (X, y) — признаки и метки
        """
        windows = self.measurement.measurement_windows
        all_data = []

        for t_idx in range(1, len(windows)):
            Tmin = windows[t_idx - 1]
            Tmax = windows[t_idx]

            print(
                f"  Window {t_idx}/{len(windows)-1}: Tmin={Tmin:.3f}, Tmax={Tmax:.3f}"
            )
            window_data = self.generate_data_for_window(Tmin, Tmax)
            all_data.append(window_data)

        # Объединяем данные из всех окон
        combined = np.vstack(all_data)

        N_features = combined.shape[1] - 1
        X = combined[:, :N_features]
        y = combined[:, N_features]

        return X, y
