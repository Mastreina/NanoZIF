"""单向吸附 MC（球面凸多边形）公共接口。"""

from .unidirectional_adsorption_polygons import (
    AdsorptionPatch,
    adsorption_polygons_on_sphere,
    square_vertices,
    regular_polygon_vertices,
    octagon_from_cube_face,
    log_map,
    rot2,
    metrics_from_patches,
)
from .unidirectional_phiA_fast import estimate_adsorption_coverage

__all__ = [
    "AdsorptionPatch",
    "adsorption_polygons_on_sphere",
    "square_vertices",
    "regular_polygon_vertices",
    "octagon_from_cube_face",
    "log_map",
    "rot2",
    "metrics_from_patches",
    "estimate_adsorption_coverage",
]
