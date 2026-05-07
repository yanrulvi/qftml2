import sys

sys.path.insert(0, ".")
import numpy as np
from data.config import create_boundary_config
from physics.evolution import FieldEvolution

config = create_boundary_config()
config.measurement.N_tom = 1e6
evo = FieldEvolution(config)

# Сравним начальные состояния
for k, case in enumerate(config.cases):
    sigma_0 = evo.lattice.get_initial_covariance(
        evo.H_thermal[k],
        temperature=case.temperature,
        is_signal=(case.boundary_type == 3),
        Gsignal=config.physics.Gsignal,
    )
    # Посмотрим на корреляции детектора с полем
    N = evo.lattice.N
    det_q = N  # индекс q детектора
    print(f"Case {case.name} (B={case.boundary_type}):")
    print(
        f"  sigma_0[det_q, 0] = {sigma_0[det_q, 0]:.10f}  (связь детектора с дальним узлом 0)"
    )
    print(
        f"  sigma_0[det_q, :3] = {sigma_0[det_q, :3]}  (первые 3 корреляции)"
    )
    print(f"  sigma_0[0, 0] = {sigma_0[0, 0]:.6f}  (сжатие узла 0)")
    print()
