"""
macro_mc 汇集两类核心算法：
1. 单向吸附 MC（球面凸多边形）
2. 可逆压缩 MC（球面凸体）

命令行入口现整合为顶层模块：
- 单向吸附：`python -m unidirectional_adsorption_cli`
- 可逆压缩：`python -m reversible_compression_cli`
"""

from .geometry_rtt import RTTParams, truncated_tetrahedron_vertices, sweep_radius
from .reversible_compression_mc import (
    ConvexSpheropolyhedron,
    ReversibleCompressionConfig,
    ReversibleCompressionMC,
    build_rtt_particle,
)

from .unidirectional_adsorption_mc import (
    AdsorptionPatch,
    adsorption_polygons_on_sphere,
    square_vertices,
    regular_polygon_vertices,
    octagon_from_cube_face,
    log_map,
    rot2,
    metrics_from_patches,
    estimate_adsorption_coverage,
)
from .spherical_polyhedra_mc import (
    SimulationConfig as SphericalMCConfig,
    PolyhedronModel,
    build_polyhedron,
    HardPolyhedraConfinementMC,
)

__all__ = [
    # 几何 / MC
    "RTTParams",
    "truncated_tetrahedron_vertices",
    "sweep_radius",
    "ConvexSpheropolyhedron",
    "ReversibleCompressionConfig",
    "ReversibleCompressionMC",
    "build_rtt_particle",
    # 单向吸附
    "AdsorptionPatch",
    "adsorption_polygons_on_sphere",
    "square_vertices",
    "regular_polygon_vertices",
    "octagon_from_cube_face",
    "log_map",
    "rot2",
    "metrics_from_patches",
    "estimate_adsorption_coverage",
    # 球内硬多面体
    "SphericalMCConfig",
    "PolyhedronModel",
    "build_polyhedron",
    "HardPolyhedraConfinementMC",
]
