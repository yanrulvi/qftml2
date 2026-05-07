# physics/lattice.py
"""
Решёточная модель квантового поля.

Реализует:
- Построение цепочки гармонических осцилляторов (Appendix B)
- Свободный гамильтониан поля (формула 7)
- Взаимодействие ближайших соседей
- Модификации границы (Таблица I)
- Связь детектор-поле (формулы 1, 8)
- Начальное состояние (вакуум / тепловое / сжатое)

Фазовое пространство (из оригинального кода):
Координаты: (q₁, q₂, ..., q_N, q_det, p₁, p₂, ..., p_N, p_det)
Индексация: q-поле [0:N-1], q-детектор [N], p-поле [N+1:2N], p-детектор [2N+1]

ВАЖНОЕ ИСПРАВЛЕНИЕ:
- Ковариационная матрица вакуума = I (тогда ⟨q²⟩ = ⟨p²⟩ = 1/2)
- Тепловое состояние через coth(βω/2) от V-матрицы
- Сжатие (squeezing) сигнального осциллятора
"""

import numpy as np
from typing import List, Tuple


class LatticeField:
    """
    Одномерная цепочка гармонических осцилляторов с ближайшими соседями.

    Гамильтониан (формула 7):
    H_UV = Σₙ (mc²/2)(p̂²ₙ + q̂²ₙ) + (ħ²/2ma²)(q̂ₙ₊₁ - q̂ₙ)²

    с граничными условиями из Таблицы I.
    """

    def __init__(
        self, n_modes: int, mass: float, lattice_spacing: float = 1.0
    ):
        """
        Args:
            n_modes: число пространственных мод (N в статье)
            mass: масса поля m·c² (в безразмерных единицах)
            lattice_spacing: шаг решётки a (по умолчанию 1)
        """
        self.N = n_modes
        self.mass = mass
        self.a = lattice_spacing

        # Полное число степеней свободы: поле + детектор
        self.N_total = n_modes + 1
        self.dim = 2 * self.N_total

    def build_adjacency_list(self) -> List[List[int]]:
        """
        Строит список соседей для одномерной цепочки.

        Из оригинального кода:
        SquareLatticeAdjList(L, d=1, IncludeBulk=True, IncludePeriodic=False)

        Returns:
            adj[i] — список соседей узла i
        """
        adj = []
        for i in range(self.N):
            neighbors = []
            if i > 0:
                neighbors.append(i - 1)  # Левый сосед
            if i < self.N - 1:
                neighbors.append(i + 1)  # Правый сосед
            adj.append(neighbors)
        return adj

    def build_env_interaction_matrix(
        self, adj: List[List[int]] | None = None
    ) -> np.ndarray:
        """
        Строит матрицу взаимодействия ближайших соседей.

        Из оригинального кода: EnvIntHam(adjList)
        Для каждой пары соседей (i,j): F[i,j] = -1 (один раз для каждой связи)

        Возвращает:
            Матрица размером (N, N) с взаимодействиями
        """
        if adj is None:
            adj = self.build_adjacency_list()

        F = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in adj[i]:
                if i < j:  # Каждую связь учитываем один раз
                    F[i, j] = -1.0
                    F[j, i] = -1.0

        return F

    def build_free_hamiltonian_q(self) -> np.ndarray:
        """
        Строит q-часть свободного гамильтониана поля (HE0_q).

        Из оригинального кода:
        HE0_q = (m + 2/m) * eye(N) + (1/m) * EnvIntHam

        В безразмерных единицах (ħ=c=a=1) формула (7) даёт:
        m/2 · p̂²ₙ + (m/2 + 1/m) · q̂²ₙ - 1/m · q̂ₙq̂ₙ₊₁

        Матрица F для H = ½XᵀFX:
        F_qq = m + 2/m (диагональ) + (1/m) * EnvIntHam
        """
        F_q = np.zeros((self.N, self.N))

        # Диагональные члены (свободное поле)
        np.fill_diagonal(F_q, self.mass + 2.0 / self.mass)

        # Взаимодействие ближайших соседей
        adj = self.build_adjacency_list()
        F_int = self.build_env_interaction_matrix(adj)
        F_q += (1.0 / self.mass) * F_int

        return F_q

    def build_free_hamiltonian_p(self) -> np.ndarray:
        """
        Строит p-часть свободного гамильтониана поля (HE0_p).

        Из оригинального кода:
        HE0_p = m * eye(N)

        В безразмерных единицах p-часть: m·p̂²ₙ → матрица F_pp = m·I.
        """
        return self.mass * np.eye(self.N)

    def build_full_hamiltonian_matrix(
        self,
        boundary_type: int = 1,
        detector_distance: int | None = None,
        omega_D: float = 10.0,
        lam: float = 10.0,
        sigma: float = 4.2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Строит полную матрицу гамильтониана F (H = ½XᵀFX)
        для системы поле + детектор.

        Реализует ComputeHams из оригинального кода.

        Фазовая структура:
        X = (q₁, ..., q_N, q_det, p₁, ..., p_N, p_det)

        Returns:
            (F_dynamic, F_thermal) — матрицы для динамики и для теплового состояния

        Таблица I:
        - B=1 (Full Bond): полная связь
        - B=2 (No Bond): разорвана связь между узлами 0 и 1
        - B=3 (Signal): dynamic=Full Bond, thermal=No Bond + сжатие
        """
        N = self.N
        dim_total = 2 * (N + 1)

        # Матрицы для q и p частей поля
        F_q = self.build_free_hamiltonian_q()
        F_p = self.build_free_hamiltonian_p()

        # Модификация границы (Таблица I)
        F_q_thermal = F_q.copy()

        if boundary_type == 1:
            # Full Bond — ничего не меняем
            pass
        elif boundary_type == 2:
            # No Bond — убираем связь между узлами 0 и 1 (дальний конец)
            F_q[0, 1] = 0.0
            F_q[1, 0] = 0.0
            F_q_thermal = F_q.copy()
        elif boundary_type == 3:
            # Signal — dynamic с полной связью, thermal без связи
            F_q_thermal[0, 1] = 0.0
            F_q_thermal[1, 0] = 0.0

        # Строим полную матрицу
        F = np.zeros((dim_total, dim_total))

        # q-часть поля: индексы 0..N-1
        F[:N, :N] = F_q

        # q-часть детектора: индекс N
        F[N, N] = omega_D

        # p-часть поля: индексы N+1..2N
        F[N + 1 : 2 * N + 1, N + 1 : 2 * N + 1] = F_p

        # p-часть детектора: индекс 2N+1
        F[2 * N + 1, 2 * N + 1] = omega_D  # Единичный коэффициент при p²_det

        # Взаимодействие детектор-поле (SetupSAHam)
        if detector_distance is None:
            detector_distance = N

        n_A = (
            detector_distance - 1
        )  # Индекс узла (0-based), к которому подключён детектор

        for x in range(N):
            # Гауссово смазывание (формула B1)
            weight = np.exp(-((x - n_A) ** 2) / (2 * sigma**2)) / (
                sigma * np.sqrt(2 * np.pi)
            )
            # Связь q_det (индекс N) с q_x (индекс x)
            F[N, x] += lam * weight
            F[x, N] += lam * weight

        # Тепловая матрица (может отличаться для B=3)
        F_thermal = F.copy()
        if boundary_type == 3:
            F_thermal[:N, :N] = (
                F_q_thermal  # Используем модифицированную q-часть
            )

        self.F_dynamic = F
        self.F_thermal = F_thermal

        return F, F_thermal

    def get_initial_covariance(
        self,
        F_thermal: np.ndarray,
        temperature: float = 0.0,
        is_signal: bool = False,
        Gsignal: float = 3.1548,
        force_thermal_F: np.ndarray = None,  # <-- НОВЫЙ ПАРАМЕТР
    ) -> np.ndarray:
        """
        Начальная ковариационная матрица.

        Если force_thermal_F задана, используется ОНА для построения начального
        состояния (вместо F_thermal). Это позволяет сделать начальное состояние
        одинаковым для всех случаев.
        """
        from scipy.linalg import sqrtm, inv, tanhm

        N = self.N
        dim_total = 2 * (N + 1)

        # Если задана принудительная тепловая матрица, используем её
        actual_F = (
            force_thermal_F if force_thermal_F is not None else F_thermal
        )

        V = actual_F[:N, :N]
        T_mat = actual_F[N + 1 : 2 * N + 1, N + 1 : 2 * N + 1]

        w = T_mat[0, 0]
        V_norm = V / w
        T_norm = T_mat / w

        SqrtM = sqrtm(V_norm)
        SqrtMinv = inv(SqrtM)

        # Базовое состояние
        sigma = np.eye(dim_total)
        sigma[:N, :N] = SqrtMinv
        sigma[N + 1 : 2 * N + 1, N + 1 : 2 * N + 1] = SqrtM

        if temperature > 0:
            from scipy.linalg import expm as scipy_expm

            beta = 1.0 / max(temperature, 1e-15)
            arg = beta * w * SqrtM / 2.0
            max_arg = np.max(np.abs(arg))
            if max_arg > 7:
                coth = np.eye(N)
                coth += 2 * scipy_expm(-2 * arg)
                coth += 2 * scipy_expm(-4 * arg)
            else:
                coth = inv(tanhm(arg))
            sigma[:N, :N] = sigma[:N, :N] @ coth
            sigma[N + 1 : 2 * N + 1, N + 1 : 2 * N + 1] = (
                sigma[N + 1 : 2 * N + 1, N + 1 : 2 * N + 1] @ coth
            )

        if is_signal:
            g = Gsignal
            # По образу авторов: ЗАМЕНЯЕМ значения, а не умножаем
            sigma[0, 0] = g  # q_0 = Gsignal
            sigma[0, N + 1] = 0.0  # обнуляем корреляции
            sigma[N + 1, 0] = 0.0
            sigma[N + 1, N + 1] = 1.0 / g  # p_0 = 1/Gsignal

        return sigma

    def get_detector_indices(self) -> Tuple[int, int]:
        """
        Возвращает индексы детектора в фазовом пространстве.

        Returns:
            (q_idx, p_idx) — индексы q_det и p_det
        """
        return self.N, 2 * self.N + 1
