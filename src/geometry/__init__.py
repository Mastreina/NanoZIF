"""
Geometry module containing core math, shapes, and polyhedron definitions.
"""

from .core import (
    normalize,
    random_unit_vector,
    orthonormal_basis,
    rotate_basis,
    gnomonic_project_point,
    lift_to_sphere,
    quat_from_axis_angle,
    quat_mul,
    quat_conj,
    rotate_vec_by_quat,
    quat_from_two_unit_vectors,
    fibonacci_sphere,
    geodesic_sphere_points,
    SpherePoint,
)
from .shapes import (
    cube_vertices,
    octa_vertices,
    rhombic_dodecahedron_vertices,
    truncated_cube_vertices,
    truncated_tetrahedron_vertices,
    truncated_rd_vertices,
)
from .polyhedron import ConvexSpheropolyhedron, sat_overlap_spheropolyhedra
from .rtt import RTTParams

__all__ = [
    "normalize",
    "random_unit_vector",
    "orthonormal_basis",
    "rotate_basis",
    "gnomonic_project_point",
    "lift_to_sphere",
    "quat_from_axis_angle",
    "quat_mul",
    "quat_conj",
    "rotate_vec_by_quat",
    "quat_from_two_unit_vectors",
    "fibonacci_sphere",
    "geodesic_sphere_points",
    "SpherePoint",
    "cube_vertices",
    "octa_vertices",
    "rhombic_dodecahedron_vertices",
    "truncated_cube_vertices",
    "truncated_tetrahedron_vertices",
    "truncated_rd_vertices",
    "ConvexSpheropolyhedron",
    "sat_overlap_spheropolyhedra",
    "RTTParams",
]
