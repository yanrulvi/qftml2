# debug/check_thermo.py
"""Диагностика термометрии"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data.config import create_thermometry_config
from physics.evolution import FieldEvolution, compute_exact_trajectory, evolve_projectors, compute_unitaries_stepped

config = create_thermometry_config()
# Уменьшаем для теста
config.measurement.N_tom = 1e8
config.ml.N_samples = 20

evolution = FieldEvolution(config)

# Проверим, что температуры реально разные
print("=" * 60)
print("ПРОВЕРКА НАЧАЛЬНЫХ СОСТОЯНИЙ")
print("=" * 60)
for k, case in enumerate(config.cases):
    sigma_0 = evolution.lattice.get_initial_covariance(
        evolution.H_thermal[k], 
        temperature=case.temperature
    )
    # Проверим, что тепловое состояние меняет ковариацию
    trace_sigma = np.trace(sigma_0)
    # Энтропия ~ log(det(sigma))
    sign, logdet = np.linalg.slogdet(sigma_0)
    print(f"  T={case.temperature:.6f}: tr(σ)={trace_sigma:.2f}, log|σ|={logdet:.4f}")

# Сравним с вакуумом
sigma_vac = evolution.lattice.get_initial_covariance(
    evolution.H_thermal[0], temperature=0.0
)
sign, logdet_vac = np.linalg.slogdet(sigma_vac)
print(f"\n  Вакуум (T=0): tr(σ)={np.trace(sigma_vac):.2f}, log|σ|={logdet_vac:.4f}")

# Проверим показания детектора для разных температур
print("\n" + "=" * 60)
print("ПОКАЗАНИЯ ДЕТЕКТОРА (t=0.1)")
print("=" * 60)

t = 0.1
q_det_values = []

for k, case in enumerate(config.cases):
    F = evolution.H_dynamic[k]
    F_th = evolution.H_thermal[k]
    
    sigma_0 = evolution.lattice.get_initial_covariance(
        F_th, temperature=case.temperature
    )
    
    U_list, U_adj = compute_unitaries_stepped(F, N_times=1, Tmin=t, Tmax=t)
    ProjList = evolve_projectors(U_list, U_adj)
    
    traj = compute_exact_trajectory(ProjList, sigma_0)
    q_val = traj[0, 0]
    q_det_values.append(q_val)
    print(f"  T={case.temperature:.6f}: ⟨q_det⟩ = {q_val:.10f}")

q_det_values = np.array(q_det_values)
print(f"\n  Размах ⟨q_det⟩: {q_det_values.max() - q_det_values.min():.10f}")
print(f"  Средний шаг между бинами: {np.mean(np.diff(q_det_values)):.10f}")
print(f"  СКО: {np.std(q_det_values):.10f}")

# Проверим амплитуду шума
N_tom = config.measurement.N_tom
noise_amplitude = q_det_values.mean() * np.sqrt(2.0 / N_tom)
print(f"\n  Амплитуда шума (N_tom={N_tom:.0e}): ~{noise_amplitude:.10f}")
print(f"  Отношение сигнал/шум: {(q_det_values.max()-q_det_values.min())/noise_amplitude:.2f}")