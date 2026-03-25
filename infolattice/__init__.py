from .lattice import InformationLattice
from .physics import State, TightBindingGS, entropy_stable, random_singlets

__all__ = [
    "InformationLattice",
    "State",
    "TightBindingGS",
    "entropy_stable",
    "random_singlets",
    "plot_3d_array",
    "save_rotating_3d_array",
]


def plot_3d_array(*args, **kwargs):
    from .plotting import plot_3d_array as _plot_3d_array

    return _plot_3d_array(*args, **kwargs)


def save_rotating_3d_array(*args, **kwargs):
    from .plotting import save_rotating_3d_array as _save_rotating_3d_array

    return _save_rotating_3d_array(*args, **kwargs)
