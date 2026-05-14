# config.py
"""
Конфигурация экспериментов из статьи:
'Decoding Quantum Field Theory with Machine Learning'

Воспроизводит:
- Секция IV: Remote Boundary Sensing (Fig. 2)
- Секция V: Thermometry (Fig. 3)

Масштабы величин
=================
Безразмерные единицы: ħ = c = a = 1

Единица энергии: E₀ = ħcK/π = (ħc/σ)·(Kσ)/π
  = 7.14 · 7 / 3.14 ≈ 15.9 (в единицах исходного конфига)
  ≈ 15.9 GHz (для термометрии)

Единица времени: t₀ = ħ/E₀ ≈ 0.063 (безразмерных)
  Физическое время: t_phys = t_dimensionless · 10 ps (для термометрии)
  Физическое время: t_phys = t_dimensionless · 10 as (для boundary sensing)

Единица длины: a₀ = a = π/K = πσ/Kσ ≈ 1.88 (в единицах σ)

Детектор
--------
ω₀ = 10.0 (безразмерных) = 10 GHz / 15.9 GHz ≈ 0.63

Boundary sensing (Секция IV, Fig. 2)
-------------------------------------
Физические параметры статьи:
  σ = 53 pm (боровский радиус)
  L = 90σ = 4.7 nm
  ω₀ = 130 eV
  m = 1 eV
  Световое время: L/c ≈ 15 as

В безразмерных единицах:
  L = 10
  Световое время: t_light = L/c = 10
  Время Гейзенберга: 1/ω₀ ≈ 0.1

Thermometry (Секция V, Fig. 3)
--------------------------------
Физические параметры статьи:
  σ = 18 mm
  L = 100a = 353 mm
  ω₀ = 10 GHz
  m = 0.1 GHz
  T ≈ 0.5 K (ħω₀/k_B)
  Время Гейзенберга: 1/ω₀ = 1/(2π·10 GHz) ≈ 16 ps

В безразмерных единицах:
  ω₀ = 10 GHz / 15.9 GHz ≈ 0.63
  Время Гейзенберга: 1/ω₀ ≈ 1.59 единиц времени ≈ 16 ps
  Тепловая энергия: kT ≈ ħω₀ → T ≈ 1.0 (безразмерных)

Пересчёт времени:
  t_phys [ps] = t_dimensionless · 10 ps
  t_phys [as] = t_dimensionless · 10 as
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict
from enum import Enum


class ExperimentMode(Enum):
    BOUNDARY_SENSING = "boundary_sensing"
    THERMOMETRY = "thermometry"


@dataclass
class PhysicsConfig:
    """
    Физические параметры системы: поле + детектор + взаимодействие.

    Все величины в безразмерных единицах (ħ = c = a = 1).
    """

    sigma: float = 4.2  # Ширина смазывания детектора
    HbarCbySig: float = 7.14  # ħc/σ
    Ksig: float = 7.0  # K·σ — UV-обрезание
    LatticeLength: int = 10  # Длина решётки (число мод)
    mcc: float = 0.1  # Масса поля m·c²
    wD: float = 10.0  # Частота детектора ω₀
    lam: float = 10.0  # Сила связи λ
    Gsignal: float = 3.1548  # Сжатие сигнального осциллятора (8 dB)
    TMean: float = 1.0 / 127.0  # Средняя температура
    TDev: float = 0.01 * (
        1.0 / 127.0
    )  # Разброс температуры (±1% вокруг TMean)

    @property
    def a(self) -> float:
        """Шаг решётки a = π/K = πσ/Kσ"""
        return np.pi * self.sigma / self.Ksig

    @property
    def E0(self) -> float:
        """Единица энергии E₀ = ħcK/π = (ħc/σ)·(Kσ)/π"""
        return self.HbarCbySig * self.Ksig / np.pi

    @property
    def a0(self) -> float:
        """Единица расстояния a₀ = a"""
        return self.a

    @property
    def t0(self) -> float:
        """Единица времени t₀ = ħ/E₀"""
        return 1.0 / self.E0

    @property
    def N_modes(self) -> int:
        """Число пространственных мод поля"""
        return self.LatticeLength

    @property
    def N_total(self) -> int:
        """Полное число степеней свободы: поле + детектор"""
        return self.LatticeLength + 1

    @property
    def phase_space_dim(self) -> int:
        """Размерность фазового пространства"""
        return 2 * self.N_total

    def get_normalized_params(self) -> Dict[str, float]:
        """Параметры в безразмерных единицах"""
        return {
            "sigma_norm": self.sigma / self.a0,
            "mcc_norm": self.mcc / self.E0,
            "wD_norm": self.wD / self.E0,
            "lam_norm": self.lam / self.E0,
            "Gsignal": self.Gsignal,
            "TMean": self.TMean,
            "TDev": self.TDev,
        }


@dataclass
class MeasurementConfig:
    """
    Параметры измерительного протокола M₀ (Секция II.A).

    Steps 1-6 из статьи:
    1. Инициализировать поле с меткой y
    2. Инициализировать детектор в основном состоянии
    3. Включить взаимодействие в t=0
    4. Измерить q̂₀, p̂₀, r̂₀ в моменты t_M
    5. Повторить N_tom раз
    6. Повторить всё N_samples раз
    """

    dt: float = 0.4  # Шаг между окнами измерений
    Tmin: float = 2.4  # Начало первого окна
    Tmax: float = 3.2  # Конец последнего окна
    N_times: int = 10  # Число времён в каждом окне
    N_tom: float = 1e20  # Число томографических повторений

    @property
    def measurement_windows(self) -> List[float]:
        """
        Границы измерительных окон.
        Из оригинального кода: linspace(Tmin, Tmax, (Tmax-Tmin)/dt + 1)
        """
        n_windows = int((self.Tmax - self.Tmin) / self.dt) + 1
        if n_windows < 2:
            n_windows = 2
        return list(np.linspace(self.Tmin, self.Tmax, n_windows))

    @property
    def heisenberg_time(self) -> float:
        """Время Гейзенберга детектора"""
        return 1.0

    @property
    def light_crossing_time(self) -> float:
        """Световое время через полость (для boundary sensing)"""
        return 10.0  # LatticeLength


@dataclass
class MLConfig:
    """Параметры машинного обучения (Секция II.C, Appendix A)"""

    N_samples: int = 200  # Число примеров на случай
    f_train: float = 0.75  # Доля данных для обучения
    f_valid: float = 0.25  # Доля данных для валидации
    f_test: float = 0.0  # Доля данных для теста

    run_pca: bool = True  # Применять ли PCA
    pca_var_keep: float = 1.0  # Доля сохраняемой дисперсии
    n_pca_plot: int = 1000  # Число точек для PCA-графика

    n_hidden: int = 30  # Число нейронов в скрытом слое
    learning_rate: float = 0.01  # Скорость обучения (SGD)
    n_epochs: int = 1000  # Число эпох
    minibatch_size: int = 100  # Размер мини-батча
    L2_reg: float = 0.001  # L2-регуляризация

    def __post_init__(self):
        fsum = self.f_train + self.f_valid + self.f_test
        if fsum > 0:
            self.f_train /= fsum
            self.f_valid /= fsum
            self.f_test /= fsum


@dataclass
class CaseLabel:
    """Описание одного случая (строка в LPYD)"""

    name: str  # Имя случая
    abv: str  # Аббревиатура (y-метка)
    prob: float  # Вероятность (0=класс, 1=регрессия)
    y: int  # y-значение
    boundary_type: int  # Тип границы (1=Full, 2=No, 3=Signal)
    distance: int  # Расстояние до границы
    temperature: float  # Температура


@dataclass
class ExperimentConfig:
    """Полная конфигурация эксперимента"""

    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    mode: ExperimentMode = ExperimentMode.BOUNDARY_SENSING
    cases: List[CaseLabel] = field(default_factory=list)

    @property
    def is_regression(self) -> bool:
        return self.mode == ExperimentMode.THERMOMETRY

    @property
    def n_classes(self) -> int:
        if self.is_regression:
            return 1
        return max(len(set(c.y for c in self.cases)), 2)

    @property
    def Y_labels(self) -> List[str]:
        return list(dict.fromkeys(c.abv for c in self.cases))


# ============================================================
# Фабрики конфигураций
# ============================================================


def create_boundary_config() -> ExperimentConfig:
    """
    Boundary Sensing (Секция IV, Fig. 2).

    Физическая картина:
    - Детектор у одной стенки резонатора (x≈0)
    - Дальняя граница на расстоянии L = 10 (безразмерных единиц)
    - Световое время: t_light = L/c = 10
    - Три случая:
      B=1 (Full Bond): граница на месте
      B=2 (No Bond): граница исчезла (связь разорвана)
      B=3 (Signal): от границы идёт возмущение (сжатие + включение связи)

    Временные масштабы:
      t < 6: сигнал от границы ещё не дошёл → точность ~33%
      t = 6-10: сигнал приходит → точность растёт
      t > 10: полная информация → точность ~100%

    Параметры как в статье:
      N_tom = 10^22 (здесь 10^6 для реалистичного графика)
    """
    config = ExperimentConfig()
    config.mode = ExperimentMode.BOUNDARY_SENSING

    config.physics = PhysicsConfig(
        sigma=4.2,
        HbarCbySig=7.14,
        Ksig=7.0,
        LatticeLength=10,
        mcc=0.1,
        wD=10.0,
        lam=10.0,
        Gsignal=3.1548,
    )

    # Временные окна: от 2 до 22 (20 окон)
    # Световое время = 10, измерения захватывают и до, и после
    config.measurement = MeasurementConfig(
        dt=0.25,
        Tmin=1.0,
        Tmax=22.0,
        N_times=10,
        N_tom=1e10,  # 10^6 для реалистичного графика (в статье 10^22)
    )

    config.ml = MLConfig(
        N_samples=500,
        n_epochs=500,
        minibatch_size=100,
        learning_rate=0.01,
        L2_reg=0.001,
    )

    L = config.physics.LatticeLength
    config.cases = [
        CaseLabel("Full Bond", "Uncut", 1 / 3, 0, 1, L, 0.0),
        CaseLabel("No Bond", "Cut", 1 / 3, 1, 2, L, 0.0),
        CaseLabel("Signal", "Sign", 1 / 3, 2, 3, L, 0.0),
    ]

    return config


def create_thermometry_config() -> ExperimentConfig:
    """
    Thermometry (Секция V, Fig. 3).

    Физическая картина:
    - Детектор в центре линии передачи (x = L/2)
    - Поле в тепловом состоянии с температурой T
    - Детектор измеряет поле, не успев термализоваться
    - Информация о температуре извлекается из локальных корреляций

    Временные масштабы:
      t < 1 ps: детектор только включился → точность ~10%
      t ≈ 16 ps: время Гейзенберга 1/ω₀ → точность ~50-70%
      t ≈ 50 ps: полная информация → точность ~90-100%

    Конвертация времени:
      t_phys [ps] = t_dimensionless · 10 ps

    Параметры как в статье:
      N_tom = 10^20 (здесь 10^6 для реалистичного графика)
    """
    config = ExperimentConfig()
    config.mode = ExperimentMode.THERMOMETRY

    # TMean = 1.0 соответствует kT ≈ ħω₀ (режим статьи)
    TMean = 1.0

    config.physics = PhysicsConfig(
        sigma=4.2,
        HbarCbySig=7.14,
        Ksig=7.0,
        LatticeLength=10,
        mcc=0.1,
        wD=10.0,  # ω₀ = 10 (безразмерных) ≈ 10 GHz / 15.9 GHz
        lam=10.0,
        TMean=TMean,
        TDev=0.1 * TMean,  # 10% разброс температуры
    )

    # Временные окна: от 0.5 до 5.0 (9 окон)
    # 0.5 ≈ 5 ps (начало), 1.6 ≈ 16 ps (время Гейзенберга), 5.0 ≈ 50 ps (конец)
    config.measurement = MeasurementConfig(
        dt=0.2,
        Tmin=0.1,
        Tmax=13.1,
        N_times=10,
        N_tom=1e14,  # 10^6 для реалистичного графика (в статье 10^20)
    )

    config.ml = MLConfig(
        N_samples=500,
        n_epochs=500,
        minibatch_size=100,
        learning_rate=0.01,
        L2_reg=0.001,
    )

    # 11 температурных бинов от 0.9·T до 1.1·T с шагом 2%
    L = config.physics.LatticeLength
    config.cases = []
    for i, frac in enumerate(np.arange(0.90, 1.11, 0.02)):
        T = frac * TMean
        config.cases.append(
            CaseLabel(
                name=f"{int(frac*100)}-{int((frac+0.02)*100)}%",
                abv=f"{frac:.2f}",
                prob=1.0,
                y=i,
                boundary_type=1,
                distance=L // 2,  # Детектор в центре
                temperature=T,
            )
        )

    return config


def create_test_config() -> ExperimentConfig:
    """Быстрый тест — облегчённая версия boundary sensing"""
    config = create_boundary_config()
    config.measurement.N_tom = 1e4
    config.measurement.dt = 2.0
    config.measurement.Tmin = 2.0
    config.measurement.Tmax = 14.0
    config.ml.N_samples = 300
    config.ml.n_epochs = 200
    return config
