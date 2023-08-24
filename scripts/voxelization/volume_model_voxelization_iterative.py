"""
Voxelization of a volume model using "iterative" method.
"""
import volmdlr
from volmdlr.core import VolumeModel
from volmdlr.primitives3d import Cylinder, Sphere
from volmdlr.voxelization import PointBasedVoxelization

VOXEL_SIZE = 0.01

# Create a volume model
sphere = Sphere(volmdlr.O3D, 0.1, name="Sphere")
cylinder = Cylinder(volmdlr.OXYZ.translation(0.1 * volmdlr.Z3D), 0.1, 0.2, name="Cylinder")
volume_model = VolumeModel([sphere, cylinder])

# Voxelize the volume model (it uses the triangulated model to create the voxelization)
voxelization = PointBasedVoxelization.from_volume_model(volume_model, VOXEL_SIZE, name="Voxelization")

# Display the result
voxelization_primitive = voxelization.to_closed_triangle_shell()
voxelization_primitive.alpha = 0.5
voxelization_primitive.color = (1, 0, 0)

volume_model.primitives.append(voxelization_primitive)
volume_model.babylonjs()
