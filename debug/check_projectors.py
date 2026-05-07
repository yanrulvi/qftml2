# debug/check_projectors.py
"""
Автономный диагностический скрипт.
Запускать из корня проекта: python debug/check_projectors.py

Проверяет 5 ключевых аспектов симуляции:
1. Инициализация детектора (вакуум = I)
2. Структура проекторов
3. Измерения без эволюции
4. Различимость граничных условий
5. Влияние взаимодействия на детектор
"""
import sys
import os

# Добавляем корень проекта в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"Python path: {sys.path[0]}")
print(f"Current dir: {os.getcwd()}")

import numpy as np
from scipy.linalg import expm

try:
    from data.config import create_test_config

    print("✅ config imported")
except Exception as e:
    print(f"❌ config import failed: {e}")
    sys.exit(1)

try:
    from physics.lattice import LatticeField

    print("✅ LatticeField imported")
except Exception as e:
    print(f"❌ LatticeField import failed: {e}")
    sys.exit(1)

try:
    from physics.evolution import (
        mult_omega,
        build_projectors,
        compute_unitaries_stepped,
        evolve_projectors,
    )

    print("✅ evolution imported")
except Exception as e:
    print(f"❌ evolution import failed: {e}")
    sys.exit(1)


def test_probe_initialization():
    """
    ТЕСТ 1: Проверка инициализации детектора.

    Ожидание: в вакууме ковариационная матрица = I,
    поэтому матричный элемент σ_qq = 1.0,
    а физическое ⟨q²⟩ = σ_qq/2 = 0.5.
    """
    print("\n" + "=" * 60)
    print("ТЕСТ 1: Инициализация детектора")
    print("=" * 60)

    config = create_test_config()
    print(f"  LatticeLength: {config.physics.LatticeLength}")
    print(f"  N_total: {config.physics.N_total}")

    lattice = LatticeField(
        n_modes=config.physics.N_modes,
        mass=config.physics.mcc,
        lattice_spacing=config.physics.a,
    )

    norm = config.physics.get_normalized_params()
    print(f"  mcc_norm = {norm['mcc_norm']:.6f}")
    print(f"  wD_norm = {norm['wD_norm']:.6f}")

    # Строим гамильтониан БЕЗ взаимодействия
    F, F_th = lattice.build_full_hamiltonian_matrix(
        boundary_type=1,
        detector_distance=config.physics.LatticeLength,
        omega_D=norm["wD_norm"],
        lam=0.0,  # Без взаимодействия
        sigma=norm["sigma_norm"],
    )

    # Начальное состояние: детектор в вакууме
    sigma_0 = lattice.get_initial_covariance(F_th, temperature=0.0)

    q_idx, p_idx = lattice.get_detector_indices()

    sigma_qq = sigma_0[q_idx, q_idx]
    sigma_pp = sigma_0[p_idx, p_idx]

    print(f"  Детектор q индекс: {q_idx}")
    print(f"  Детектор p индекс: {p_idx}")
    print(f"  sigma_0 shape: {sigma_0.shape}")
    print(f"  σ_qq (матричный элемент) = {sigma_qq:.6f}")
    print(f"  σ_pp (матричный элемент) = {sigma_pp:.6f}")
    print(f"  Физическое ⟨q²⟩ = σ_qq/2 = {sigma_qq/2:.6f} (ожидание 0.5)")
    print(f"  Физическое ⟨p²⟩ = σ_pp/2 = {sigma_pp/2:.6f} (ожидание 0.5)")

    # Проверяем: вакуум должен быть единичной матрицей
    all_ok = True
    if abs(sigma_qq - 1.0) > 1e-6:
        print(f"  ❌ σ_qq != 1.0, sigma_0 не единичная!")
        all_ok = False
    if abs(sigma_pp - 1.0) > 1e-6:
        print(f"  ❌ σ_pp != 1.0, sigma_0 не единичная!")
        all_ok = False

    # Проверяем, что вся матрица единичная
    is_identity = np.allclose(sigma_0, np.eye(sigma_0.shape[0]), atol=1e-10)
    if is_identity:
        print("  ✅ Полная sigma_0 = I (единичная матрица)")
    else:
        off_diag = np.max(np.abs(sigma_0 - np.eye(sigma_0.shape[0])))
        print(
            f"  ⚠️ sigma_0 отличается от I, макс. недиаг. элемент: {off_diag:.2e}"
        )

    if all_ok:
        print("  ✅ Детектор правильно инициализирован")

    return sigma_0


def test_projectors_structure():
    """
    ТЕСТ 2: Проверка структуры проекторов.

    P0 должен иметь 1 на позиции [q_idx, q_idx],
    P2 — 1 на позиции [p_idx, p_idx],
    P1 — правильную комбинацию.
    """
    print("\n" + "=" * 60)
    print("ТЕСТ 2: Структура проекторов")
    print("=" * 60)

    config = create_test_config()
    N_total = config.physics.N_total
    dim = 2 * N_total

    P0, P1, P2 = build_projectors(N_total)

    q_idx = N_total - 1
    p_idx = 2 * N_total - 1

    print(f"  N_total = {N_total}")
    print(f"  dim = {dim}")
    print(f"  q_det индекс = {q_idx}")
    print(f"  p_det индекс = {p_idx}")

    # Проверка P0
    p0_val = P0[q_idx, q_idx]
    print(f"  P0[{q_idx},{q_idx}] = {p0_val:.6f} (ожидание 1)")

    # Проверка P2
    p2_val = P2[p_idx, p_idx]
    print(f"  P2[{p_idx},{p_idx}] = {p2_val:.6f} (ожидание 1)")

    # Проверка следов с единичной матрицей
    trace_q = np.trace(P0)
    trace_p = np.trace(P2)
    print(f"  Tr[P0] = {trace_q:.6f} (ожидание 1)")
    print(f"  Tr[P2] = {trace_p:.6f} (ожидание 1)")

    all_ok = True
    if abs(p0_val - 1.0) > 1e-10:
        print(f"  ❌ P0[{q_idx},{q_idx}] != 1")
        all_ok = False
    if abs(p2_val - 1.0) > 1e-10:
        print(f"  ❌ P2[{p_idx},{p_idx}] != 1")
        all_ok = False

    # Проверка P1 = (P0 + P2)/2 + перекрёстные члены
    expected_P1 = (P0 + P2) / 2.0
    expected_P1[q_idx, p_idx] = -0.5
    expected_P1[p_idx, q_idx] = -0.5
    if np.allclose(P1, expected_P1, atol=1e-10):
        print("  ✅ P1 имеет правильную структуру")
    else:
        print("  ❌ P1 неправильный!")
        print(
            f"  P1[{q_idx},{p_idx}] = {P1[q_idx, p_idx]:.6f} (ожидание -0.5)"
        )
        print(
            f"  P1[{p_idx},{q_idx}] = {P1[p_idx, q_idx]:.6f} (ожидание -0.5)"
        )
        all_ok = False

    if all_ok:
        print("  ✅ Проекторы имеют правильную структуру")


def test_no_evolution():
    """
    ТЕСТ 3: Измерения без эволюции (t=0).

    При t=0 и без взаимодействия детектор в вакууме:
    ⟨q⟩ = Tr[P0 @ I] = 1? НЕТ!

    ВАЖНО: проекторы измеряют НЕ СРЕДНИЕ, а КВАДРАТЫ!
    P0 — проектор на q_det², поэтому:
    Tr[P0 @ I] = Tr[P0] = 1 (для чистого проектора ранга 1)

    Но физически ⟨q²⟩ = Tr[P0 @ sigma]/2 = 1/2 для вакуума.

    В оригинальном коде проекторы используются ТОЛЬКО для
    вычисления Tr[P @ sigma], и это даёт правильные значения.
    """
    print("\n" + "=" * 60)
    print("ТЕСТ 3: Измерения без эволюции (t=0)")
    print("=" * 60)

    config = create_test_config()
    lattice = LatticeField(
        n_modes=config.physics.N_modes,
        mass=config.physics.mcc,
        lattice_spacing=config.physics.a,
    )

    norm = config.physics.get_normalized_params()

    # Гамильтониан без взаимодействия
    F, F_th = lattice.build_full_hamiltonian_matrix(
        boundary_type=1,
        detector_distance=config.physics.LatticeLength,
        omega_D=norm["wD_norm"],
        lam=0.0,
        sigma=norm["sigma_norm"],
    )

    sigma_0 = lattice.get_initial_covariance(F_th, temperature=0.0)

    # Эволюция при t=0
    U_list, U_adj_list = compute_unitaries_stepped(
        F, N_times=1, Tmin=0.0, Tmax=0.0
    )
    ProjList = evolve_projectors(U_list, U_adj_list)

    # "Измерения" — след проектора с sigma_0
    traj = np.zeros((1, 3))
    for r in range(3):
        traj[0, r] = np.trace(ProjList[0, r] @ sigma_0).real

    print(f"  Tr[P_q(t=0) @ sigma_0] = {traj[0, 0]:.10f}")
    print(f"  Tr[P_r(t=0) @ sigma_0] = {traj[0, 1]:.10f}")
    print(f"  Tr[P_p(t=0) @ sigma_0] = {traj[0, 2]:.10f}")
    print(f"  Ожидание: Tr[P @ I] = 1 (след проектора)")
    print(f"  Физическое ⟨q²⟩ = Tr[P_q @ sigma_0]/2 = {traj[0, 0]/2:.6f}")
    print(f"  Физическое ⟨p²⟩ = Tr[P_p @ sigma_0]/2 = {traj[0, 2]/2:.6f}")

    # Проверяем, что след проектора с единичной матрицей = 1
    if abs(traj[0, 0] - 1.0) < 1e-6 and abs(traj[0, 2] - 1.0) < 1e-6:
        print("  ✅ Следы проекторов с I равны 1 (правильно)")
    else:
        print("  ❌ ОШИБКА: следы не равны 1")


def test_boundary_difference():
    """
    ТЕСТ 4: Различимость граничных условий.

    Проверяем, что B=1, B=2, B=3 дают разные показания детектора.
    Также проверяем, что обрезание связи действительно работает.
    """
    print("\n" + "=" * 60)
    print("ТЕСТ 4: Различимость граничных условий")
    print("=" * 60)

    config = create_test_config()
    lattice = LatticeField(
        n_modes=config.physics.N_modes,
        mass=config.physics.mcc,
        lattice_spacing=config.physics.a,
    )

    norm = config.physics.get_normalized_params()
    print(f"  N_modes = {lattice.N}")
    print(f"  mass = {norm['mcc_norm']:.6f}")
    print(f"  lam_norm = {norm['lam_norm']:.3f}")
    print(f"  wD_norm = {norm['wD_norm']:.3f}")
    print(f"  sigma_norm = {norm['sigma_norm']:.3f}")

    # Сначала проверим, что F действительно разный для B=1 и B=2
    print("\n  --- Проверка F-матриц ---")
    F1, _ = lattice.build_full_hamiltonian_matrix(
        boundary_type=1,
        detector_distance=config.physics.LatticeLength,
        omega_D=norm["wD_norm"],
        lam=norm["lam_norm"],
        sigma=norm["sigma_norm"],
    )
    F2, _ = lattice.build_full_hamiltonian_matrix(
        boundary_type=2,
        detector_distance=config.physics.LatticeLength,
        omega_D=norm["wD_norm"],
        lam=norm["lam_norm"],
        sigma=norm["sigma_norm"],
    )

    print(f"  F1[0,1] = {F1[0, 1]:.10f}")
    print(f"  F1[1,0] = {F1[1, 0]:.10f}")
    print(f"  F2[0,1] = {F2[0, 1]:.10f} (должно быть 0 для No Bond)")
    print(f"  F2[1,0] = {F2[1, 0]:.10f} (должно быть 0 для No Bond)")
    print(f"  ||F1 - F2||_F = {np.linalg.norm(F1 - F2):.6f}")

    if (
        abs(F1[0, 1]) > 1e-10
        and abs(F1[1, 0]) > 1e-10
        and abs(F2[0, 1]) < 1e-10
        and abs(F2[1, 0]) < 1e-10
    ):
        print("  ✅ Связь [0,1] правильно обнуляется для B=2")
    else:
        print("  ❌ Проблема с обнулением связи!")

    # Теперь сравним показания детектора
    print("\n  --- Показания детектора при t=3.0 ---")

    trajectories = {}
    t_measure = 3.0

    for B in [1, 2, 3]:
        F, F_th = lattice.build_full_hamiltonian_matrix(
            boundary_type=B,
            detector_distance=config.physics.LatticeLength,
            omega_D=norm["wD_norm"],
            lam=norm["lam_norm"],
            sigma=norm["sigma_norm"],
        )

        is_signal = B == 3
        sigma_0 = lattice.get_initial_covariance(
            F_th, temperature=0.0, is_signal=is_signal, Gsignal=norm["Gsignal"]
        )

        Omega_F = mult_omega(F)
        U = expm(Omega_F * t_measure)
        U_adj = U.T

        N_total = config.physics.N_total
        Proj0 = build_projectors(N_total)

        traj = []
        for r in range(3):
            P_evolved = U_adj @ Proj0[r] @ U
            val = np.trace(P_evolved @ sigma_0).real
            traj.append(val)

        trajectories[B] = traj
        print(
            f"  B={B}: ⟨q⟩={traj[0]:.8f}, ⟨r⟩={traj[1]:.8f}, ⟨p⟩={traj[2]:.8f}"
        )

    # Проверяем различимость
    diff_12 = np.linalg.norm(
        np.array(trajectories[1]) - np.array(trajectories[2])
    )
    diff_13 = np.linalg.norm(
        np.array(trajectories[1]) - np.array(trajectories[3])
    )
    diff_23 = np.linalg.norm(
        np.array(trajectories[2]) - np.array(trajectories[3])
    )

    print(f"\n  ||B1 - B2|| = {diff_12:.8f}")
    print(f"  ||B1 - B3|| = {diff_13:.8f}")
    print(f"  ||B2 - B3|| = {diff_23:.8f}")

    # B2 и B3 должны различаться ТОЛЬКО за счёт сжатия
    print(f"\n  Сжатие Gsignal = {norm['Gsignal']:.4f}")
    print(
        f"  sigma_0[0,0] для B=3 = {sigma_0[0, 0]:.6f} (должно быть 1/g = {1/norm['Gsignal']:.6f})"
    )

    if diff_12 > 1e-8:
        print("  ✅ B1 и B2 различимы")
    else:
        print(
            "  ⚠️ B1 и B2 почти неразличимы — это ожидаемо для данного времени"
        )

    if diff_23 > 1e-8:
        print("  ✅ B2 и B3 различимы (сжатие работает)")
    else:
        print("  ⚠️ B2 и B3 почти неразличимы — возможно, сжатие не влияет")


def test_interaction_strength():
    """
    ТЕСТ 5: Влияние взаимодействия на детектор.

    Сравнивает показания с λ=0 и с λ≠0.
    """
    print("\n" + "=" * 60)
    print("ТЕСТ 5: Влияние взаимодействия на детектор")
    print("=" * 60)

    config = create_test_config()
    lattice = LatticeField(
        n_modes=config.physics.N_modes,
        mass=config.physics.mcc,
        lattice_spacing=config.physics.a,
    )

    norm = config.physics.get_normalized_params()

    results = {}
    t_measure = 3.0

    for lam_val in [0.0, norm["lam_norm"]]:
        F, F_th = lattice.build_full_hamiltonian_matrix(
            boundary_type=1,
            detector_distance=config.physics.LatticeLength,
            omega_D=norm["wD_norm"],
            lam=lam_val,
            sigma=norm["sigma_norm"],
        )

        sigma_0 = lattice.get_initial_covariance(F_th, temperature=0.0)

        Omega_F = mult_omega(F)
        U = expm(Omega_F * t_measure)
        U_adj = U.T

        N_total = config.physics.N_total
        Proj0 = build_projectors(N_total)

        P_evolved = U_adj @ Proj0[0] @ U
        val = np.trace(P_evolved @ sigma_0).real

        results[lam_val] = val
        print(f"  λ={lam_val:.3f}: ⟨q_det⟩ = {val:.8f}")

    diff = abs(results[0.0] - results[norm["lam_norm"]])
    print(f"  Разница: {diff:.8f}")

    if diff > 1e-8:
        print("  ✅ Взаимодействие влияет на измерения")
    else:
        print("  ❌ Взаимодействие не влияет — проблема в H_int!")


def test_F_matrices_detail():
    """
    ДОПОЛНИТЕЛЬНЫЙ ТЕСТ: Детальная проверка F-матриц.

    Сравнивает q-части F для B=1 и B=2.
    """
    print("\n" + "=" * 60)
    print("ДОП. ТЕСТ: Детали F-матриц")
    print("=" * 60)

    config = create_test_config()
    lattice = LatticeField(
        n_modes=config.physics.N_modes,
        mass=config.physics.mcc,
        lattice_spacing=config.physics.a,
    )

    norm = config.physics.get_normalized_params()

    F1, _ = lattice.build_full_hamiltonian_matrix(
        boundary_type=1,
        detector_distance=config.physics.LatticeLength,
        omega_D=norm["wD_norm"],
        lam=0.0,
        sigma=norm["sigma_norm"],
    )
    F2, _ = lattice.build_full_hamiltonian_matrix(
        boundary_type=2,
        detector_distance=config.physics.LatticeLength,
        omega_D=norm["wD_norm"],
        lam=0.0,
        sigma=norm["sigma_norm"],
    )

    N = lattice.N

    print(f"  q-часть F1 (первые 3x3):")
    print(F1[:3, :3])
    print(f"\n  q-часть F2 (первые 3x3):")
    print(F2[:3, :3])
    print(f"\n  Разница F1 - F2 (первые 3x3):")
    diff = F1[:3, :3] - F2[:3, :3]
    print(diff)

    # Проверяем значение недиагонального элемента
    original_coupling = F1[0, 1]
    print(f"\n  Исходная связь F1[0,1] = {original_coupling:.10f}")
    print(f"  Масса = {lattice.mass:.10f}")
    print(f"  Ожидаемая связь = -1/mass = {-1/lattice.mass:.10f}")

    if abs(original_coupling + 1.0 / lattice.mass) < 1e-10:
        print("  ✅ Связь правильная: -1/mass")
    else:
        print(f"  ⚠️ Связь отличается от ожидаемой!")


if __name__ == "__main__":
    print("=" * 60)
    print("ЗАПУСК ДИАГНОСТИКИ")
    print("=" * 60)

    tests = [
        ("ТЕСТ 1", test_probe_initialization),
        ("ТЕСТ 2", test_projectors_structure),
        ("ТЕСТ 3", test_no_evolution),
        ("ТЕСТ 4", test_boundary_difference),
        ("ТЕСТ 5", test_interaction_strength),
        ("ДОП. ТЕСТ", test_F_matrices_detail),
    ]

    for name, test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"\n❌ {name} УПАЛ: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print("ДИАГНОСТИКА ЗАВЕРШЕНА")
    print("=" * 60)
