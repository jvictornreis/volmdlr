"""
Example of voxelization from a STL file.
"""
from volmdlr.voxelization import Voxelization
from volmdlr.stl import Stl

VOXEL_SIZE = 0.0015
STL_MODEL_FILE_PATH = "../stl/simple.stl"

# Load and convert the STL
volume_model = Stl.load_from_file(STL_MODEL_FILE_PATH).to_volume_model()

# Voxelize the model
voxelization = Voxelization.from_volume_model(volume_model, VOXEL_SIZE, method="iterative", name="Voxelization")

# Display the result
voxelization_primitive = voxelization.to_closed_triangle_shell()
voxelization_primitive.alpha = 0.5
voxelization_primitive.color = (1, 0, 0)

volume_model.primitives.append(voxelization_primitive)
volume_model.babylonjs()
