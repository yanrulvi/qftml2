# physics/__init__.py
from physics.lattice import LatticeField
from physics.evolution import (
    symplectic_form,
    mult_omega,
    compute_unitaries,
    compute_unitaries_stepped,
    build_projectors,
    evolve_projectors,
    compute_exact_trajectory,
    compute_median_trajectory_optimized,
    apply_tomography_noise,
    FieldEvolution,
)