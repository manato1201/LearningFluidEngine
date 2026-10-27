"""FluidKit solver package."""

from .stable_fluids    import StableFluids2D
from .sph_solver       import SPHSolver, Particle, SpatialHash
from .flip_solver      import FLIPSolver, MACGrid
from .reproducibility  import SimulationRecord
from .runner           import run_simulation_sandboxed, validate_config, timeout
from .curl_noise       import CurlNoise2D, CurlNoise3D, divergence_residual

__all__ = [
    "StableFluids2D",
    "SPHSolver", "Particle", "SpatialHash",
    "FLIPSolver", "MACGrid",
    "SimulationRecord",
    "run_simulation_sandboxed", "validate_config", "timeout",
    "CurlNoise2D", "CurlNoise3D", "divergence_residual",
]
