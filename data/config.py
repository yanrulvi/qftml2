# config.py
"""
Конфигурация экспериментов из статьи:
'Decoding Quantum Field Theory with Machine Learning'
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
    sigma: float = 4.2
    HbarCbySig: float = 7.14
    Ksig: float = 7.0
    LatticeLength: int = 10
    mcc: float = 0.1
    wD: float = 10.0
    lam: float = 10.0
    Gsignal: float = 3.1548
    TMean: float = 1.0 / 127.0
    TDev: float = 0.01 * (1.0 / 127.0)

    @property
    def a(self) -> float:
        return np.pi * self.sigma / self.Ksig

    @property
    def E0(self) -> float:
        return self.HbarCbySig * self.Ksig / np.pi

    @property
    def a0(self) -> float:
        return self.a

    @property
    def N_modes(self) -> int:
        return self.LatticeLength

    @property
    def N_total(self) -> int:
        return self.LatticeLength + 1

    @property
    def phase_space_dim(self) -> int:
        return 2 * self.N_total

    def get_normalized_params(self) -> Dict[str, float]:
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
    dt: float = 0.4
    Tmin: float = 2.4
    Tmax: float = 3.2
    N_times: int = 10
    N_tom: float = 1e20

    @property
    def measurement_windows(self) -> List[float]:
        n_windows = int((self.Tmax - self.Tmin) / self.dt) + 1
        if n_windows < 2:
            n_windows = 2
        return list(np.linspace(self.Tmin, self.Tmax, n_windows))


@dataclass
class MLConfig:
    N_samples: int = 200
    f_train: float = 0.75
    f_valid: float = 0.25
    f_test: float = 0.0

    run_pca: bool = True
    pca_var_keep: float = 1.0
    n_pca_plot: int = 1000

    n_hidden: int = 30
    learning_rate: float = 0.01
    n_epochs: int = 1000
    minibatch_size: int = 100
    L2_reg: float = 0.001

    def __post_init__(self):
        fsum = self.f_train + self.f_valid + self.f_test
        if fsum > 0:
            self.f_train /= fsum
            self.f_valid /= fsum
            self.f_test /= fsum


@dataclass
class CaseLabel:
    name: str
    abv: str
    prob: float
    y: int
    boundary_type: int
    distance: int
    temperature: float


@dataclass
class ExperimentConfig:
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
    """Boundary Sensing (Секция IV, Fig. 2)"""
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

    config.measurement = MeasurementConfig(
        dt=1.0,
        Tmin=2.0,
        Tmax=22.0,
        N_times=10,
        N_tom=1e10,
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
    """Thermometry (Секция V, Fig. 3)"""
    config = ExperimentConfig()
    config.mode = ExperimentMode.THERMOMETRY

    TMean = 1.0

    config.physics = PhysicsConfig(
        sigma=4.2,
        HbarCbySig=7.14,
        Ksig=7.0,
        LatticeLength=10,
        mcc=0.1,
        wD=10.0,
        lam=10.0,
        TMean=TMean,
        TDev=0.1 * TMean,
    )

    config.measurement = MeasurementConfig(
        dt=0.01,
        Tmin=0.02,
        Tmax=0.20,
        N_times=10,
        N_tom=1e15,  # КАК В СТАТЬЕ
    )

    config.ml = MLConfig(
        N_samples=500,
        n_epochs=500,
        minibatch_size=100,
        learning_rate=0.01,
        L2_reg=0.001,
    )

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
                distance=L // 2,
                temperature=T,
            )
        )

    return config


def create_test_config() -> ExperimentConfig:
    """Быстрый тест — как boundary, но с меньшими параметрами"""
    return create_boundary_config()
