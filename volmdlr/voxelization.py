"""
Class for voxel representation of volmdlr models
"""
import warnings
from typing import List, Set, Tuple

import numpy as np
from dessia_common.core import PhysicalObject
from tqdm import tqdm
from volmdlr import Point3D, Vector3D, Point2D
from volmdlr.core import VolumeModel
from volmdlr.faces import Triangle3D, PlaneFace3D
from volmdlr.surfaces import Surface2D, PLANE3D_OYZ, PLANE3D_OXY, PLANE3D_OXZ
from volmdlr.wires import ClosedPolygon2D
from volmdlr.shells import ClosedShell3D, ClosedTriangleShell3D

# Typings
Point = Tuple[float, ...]
Triangle = Tuple[Point, ...]
Mesh = List[Triangle]


class Voxelization(PhysicalObject):
    """
    Class for manipulation voxelization of geometry
    """

    def __init__(self, voxels: Set[Point], voxel_size: float, name: str = ""):
        self.voxels = voxels
        self.voxel_size = voxel_size

        PhysicalObject.__init__(self, name=name)

    @classmethod
    def from_closed_triangle_shell(
        cls, closed_triangle_shell: ClosedTriangleShell3D, voxel_size: float, name: str = ""
    ):
        triangles = cls._closed_triangle_shell_to_triangles(closed_triangle_shell)
        voxels = cls._triangles_to_voxels(triangles, voxel_size)
        return cls(voxels, voxel_size, name=name)

    @classmethod
    def from_closed_shell(cls, closed_shell: ClosedShell3D, voxel_size: float, name: str = ""):
        triangles = cls._closed_shell_to_triangles(closed_shell)
        voxels = cls._triangles_to_voxels(triangles, voxel_size)
        return cls(voxels, voxel_size, name=name)

    @classmethod
    def from_volume_model(cls, volume_model: VolumeModel, voxel_size: float, name: str = ""):
        triangles = cls._volume_model_to_triangles(volume_model)
        voxels = cls._triangles_to_voxels(triangles, voxel_size)
        return cls(voxels, voxel_size, name=name)

    @staticmethod
    def _closed_triangle_shell_to_triangles(cs: ClosedTriangleShell3D):
        return list(
            tuple(tuple([point.x, point.y, point.z]) for point in [tr.point1, tr.point2, tr.point3])
            for tr in cs.primitives
        )

    @staticmethod
    def _closed_shell_to_triangles(cs: ClosedShell3D):
        triangulation = cs.triangulation()
        return list(
            tuple(
                tuple([point.x, point.y, point.z])
                for point in [
                    triangulation.points[triangle[0]],
                    triangulation.points[triangle[1]],
                    triangulation.points[triangle[2]],
                ]
            )
            for triangle in triangulation.triangles
        )

    @staticmethod
    def _volume_model_to_triangles(vm: VolumeModel):
        triangles = []
        for primitive in vm.primitives:
            triangulation = primitive.triangulation()
            triangles += list(
                tuple(
                    tuple([point.x, point.y, point.z])
                    for point in [
                        triangulation.points[triangle[0]],
                        triangulation.points[triangle[1]],
                        triangulation.points[triangle[2]],
                    ]
                )
                for triangle in triangulation.triangles
            )
        return triangles

    @staticmethod
    def _triangle_aabb_intersection(
        triangle: np.ndarray, box_center: np.ndarray, box_extents: np.ndarray
    ) -> bool:
        X, Y, Z = 0, 1, 2

        # Translate triangle as conceptually moving AABB to origin
        v = triangle - box_center

        # Compute edge vectors for triangle
        f = np.empty_like(triangle)
        f[0] = triangle[1] - triangle[0]
        f[1] = triangle[2] - triangle[1]
        f[2] = triangle[0] - triangle[2]

        for i in range(3):
            a = np.zeros(3)
            # axis a0i
            a[np.mod(i + 1, 3)] = -f[i][np.mod(i + 2, 3)]
            a[np.mod(i + 2, 3)] = f[i][np.mod(i + 1, 3)]
            p = np.dot(v, a)
            r = box_extents[np.mod(i + 1, 3)] * abs(f[i][np.mod(i + 2, 3)]) + box_extents[
                np.mod(i + 2, 3)
            ] * abs(f[i][np.mod(i + 1, 3)])
            if (max(-max(p), min(p))) > r:
                return False

            # axis a1i
            a[np.mod(i + 2, 3)] = -f[i][np.mod(i, 3)]
            a[np.mod(i, 3)] = f[i][np.mod(i + 2, 3)]
            p = np.dot(v, a)
            r = box_extents[np.mod(i, 3)] * abs(f[i][np.mod(i + 2, 3)]) + box_extents[
                np.mod(i + 2, 3)
            ] * abs(f[i][np.mod(i, 3)])
            if (max(-max(p), min(p))) > r:
                return False

            # axis a2i
            a[np.mod(i, 3)] = -f[i][np.mod(i + 1, 3)]
            a[np.mod(i + 1, 3)] = f[i][np.mod(i, 3)]
            p = np.dot(v, a)
            r = box_extents[np.mod(i, 3)] * abs(f[i][np.mod(i + 1, 3)]) + box_extents[
                np.mod(i + 1, 3)
            ] * abs(f[i][np.mod(i, 3)])
            if (max(-max(p), min(p))) > r:
                return False

        # Test the three axes corresponding to the face normals of AABB b (category 1)
        if np.any(np.max(v, axis=0) < -box_extents) or np.any(np.min(v, axis=0) > box_extents):
            return False

        # Test separating axis corresponding to triangle face normal (category 2)

        plane_normal = np.cross(f[0], f[1])
        plane_distance = np.dot(plane_normal, v[0])

        # Compute the projection interval radius of b onto L(t) = b.c + t * p.n
        r = (
            box_extents[X] * abs(plane_normal[X])
            + box_extents[Y] * abs(plane_normal[Y])
            + box_extents[Z] * abs(plane_normal[Z])
        )

        # Intersection occurs when plane distance falls within [-r,+r] interval
        if abs(plane_distance) > r:
            return False

        return True

    @staticmethod
    def _intersecting_boxes(
        min_point: Point,
        max_point: Point,
        voxel_size: float,
    ) -> List[Point]:
        """
        Calculate the indices of the cubes that intersect with a bounding box.

        :param min_point: the minimum point of the bounding box
        :type min_point: Tuple[float, ...]
        :param max_point: the maximum point of the bounding box
        :type max_point: Tuple[float, ...]
        :param voxel_size: the size of the grid cubes
        :type voxel_size: float
        :return: a list of the centers of the intersecting cubes
        :rtype: List[Tuple[float, ...]]
        """
        # Calculate the indices of the cubes that intersect with the bounding box
        x_indices = range(int(min_point[0] / voxel_size) - 1, int(max_point[0] / voxel_size) + 1)
        y_indices = range(int(min_point[1] / voxel_size) - 1, int(max_point[1] / voxel_size) + 1)
        z_indices = range(int(min_point[2] / voxel_size) - 1, int(max_point[2] / voxel_size) + 1)

        # Create a list of all the intersecting bounding boxes
        centers = []
        for x in x_indices:
            for y in y_indices:
                for z in z_indices:
                    center = tuple(round((_ + 1 / 2) * voxel_size, 6) for _ in [x, y, z])
                    centers.append(center)

        return centers

    @staticmethod
    def _intersecting_voxels(
        voxel_array: np.ndarray,
        voxel_size: float,
    ) -> Set[Point]:
        """
        Calculate the indices of the cubes that intersect with a bounding box.

        :param voxel_array: array of voxel centers
        :type voxel_array: np.ndarray
        :param voxel_size: the size of the grid cubes
        :type voxel_size: float
        :return: a set of the centers of the intersecting cubes
        :rtype: Set[Point]
        """
        # Compute the voxel indices
        indices = np.floor(voxel_array / voxel_size).astype(int)

        # Compute unique indices to avoid duplicates
        unique_indices = np.unique(indices, axis=0)

        # Convert back to voxel centers
        centers = np.around((unique_indices + 0.5) * voxel_size, 6)

        return set(tuple(center) for center in centers)

    @staticmethod
    def _surrounding_voxels(
        voxel_array: np.ndarray,
        voxel_size: float,
    ) -> Set[Point]:
        surrounding_voxels = Voxelization._round_point_cloud_voxel_size(voxel_array, voxel_size)

        split_surruding_voxels = set()
        for voxel in surrounding_voxels:
            split_surruding_voxels.update(Voxelization._subdivide(voxel, voxel_size * 2))

        return split_surruding_voxels

    @staticmethod
    def _subdivide(voxel, voxel_size):
        children = []
        half_size = voxel_size / 2
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # calculate the center of the child node
                    child = (
                        voxel[0] + (i - 0.5) * half_size,
                        voxel[1] + (j - 0.5) * half_size,
                        voxel[2] + (k - 0.5) * half_size,
                    )

                    children.append(child)

        return children

    @staticmethod
    def _triangles_to_voxels(
        triangles: List[Triangle],
        voxel_size: float,
    ) -> Set[Point]:
        """
        Convert a list of triangles into a voxel representation.

        :param triangles: a list of tuples representing the triangles
        :type triangles: List[Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]]
        :param voxel_size: the size of the grid cubes
        :type voxel_size: float
        :return: a set of the centers of the bounding boxes that intersect with the triangles
        :rtype: set
        """
        bbox_centers = set()

        for triangle in tqdm(triangles):
            min_point = tuple(min(p[i] for p in triangle) for i in range(3))
            max_point = tuple(max(p[i] for p in triangle) for i in range(3))

            for bbox_center in Voxelization._intersecting_boxes(min_point, max_point, voxel_size):
                if bbox_center not in bbox_centers:
                    if Voxelization._triangle_aabb_intersection(
                        np.array(triangle),
                        np.array(bbox_center),
                        np.array([voxel_size for _ in range(3)]),
                    ):
                        bbox_centers.add(bbox_center)

        return bbox_centers

    @staticmethod
    def _voxel_faces(center: Point, voxel_size: float) -> List[Triangle]:
        x, y, z = center
        sx, sy, sz = voxel_size, voxel_size, voxel_size
        hx, hy, hz = sx / 2, sy / 2, sz / 2
        faces = [
            # Front face
            tuple(
                [
                    (x - hx, y - hy, z + hz),
                    (x - hx, y + hy, z + hz),
                    (x + hx, y + hy, z + hz),
                ]
            ),
            tuple(
                [
                    (x - hx, y - hy, z + hz),
                    (x + hx, y + hy, z + hz),
                    (x + hx, y - hy, z + hz),
                ]
            ),
            # Back face
            tuple(
                [
                    (x - hx, y - hy, z - hz),
                    (x - hx, y + hy, z - hz),
                    (x + hx, y + hy, z - hz),
                ]
            ),
            tuple(
                [
                    (x - hx, y - hy, z - hz),
                    (x + hx, y + hy, z - hz),
                    (x + hx, y - hy, z - hz),
                ]
            ),
            # Left face
            tuple(
                [
                    (x - hx, y - hy, z - hz),
                    (x - hx, y + hy, z - hz),
                    (x - hx, y + hy, z + hz),
                ]
            ),
            tuple(
                [
                    (x - hx, y - hy, z - hz),
                    (x - hx, y + hy, z + hz),
                    (x - hx, y - hy, z + hz),
                ]
            ),
            # Right face
            tuple(
                [
                    (x + hx, y - hy, z - hz),
                    (x + hx, y + hy, z - hz),
                    (x + hx, y + hy, z + hz),
                ]
            ),
            tuple(
                [
                    (x + hx, y - hy, z - hz),
                    (x + hx, y + hy, z + hz),
                    (x + hx, y - hy, z + hz),
                ]
            ),
            # Top face
            tuple(
                [
                    (x - hx, y + hy, z - hz),
                    (x + hx, y + hy, z - hz),
                    (x + hx, y + hy, z + hz),
                ]
            ),
            tuple(
                [
                    (x - hx, y + hy, z - hz),
                    (x + hx, y + hy, z + hz),
                    (x - hx, y + hy, z + hz),
                ]
            ),
            # Bottom face
            tuple(
                [
                    (x - hx, y - hy, z - hz),
                    (x + hx, y - hy, z - hz),
                    (x + hx, y - hy, z + hz),
                ]
            ),
            tuple(
                [
                    (x - hx, y - hy, z - hz),
                    (x + hx, y - hy, z + hz),
                    (x - hx, y - hy, z + hz),
                ]
            ),
        ]

        faces = [tuple(tuple(round(_, 6) for _ in point) for point in face) for face in faces]
        return faces

    @staticmethod
    def _triangles_to_segments(triangles: List[Triangle]):
        segments_x = {}
        segments_y = {}
        segments_z = {}

        for triangle in triangles:
            if triangle[0][0] == triangle[1][0] == triangle[2][0]:
                segments = segments_x.get(triangle[0][0], set())
                triangle_segments = [
                    (tuple(triangle[0][1:]), tuple(triangle[1][1:])),
                    (tuple(triangle[1][1:]), tuple(triangle[2][1:])),
                    (tuple(triangle[2][1:]), tuple(triangle[0][1:])),
                ]
                for segment in triangle_segments:
                    if segment not in segments and segment[::-1] not in segments:
                        segments.add(segment)
                    else:
                        if segment in segments:
                            segments.remove(segment)
                        else:
                            segments.remove(segment[::-1])

                segments_x[triangle[0][0]] = segments

            elif triangle[0][1] == triangle[1][1] == triangle[2][1]:
                segments = segments_y.get(triangle[0][1], set())
                triangle_segments = [
                    (
                        tuple([triangle[0][0], triangle[0][2]]),
                        tuple([triangle[1][0], triangle[1][2]]),
                    ),
                    (
                        tuple([triangle[1][0], triangle[1][2]]),
                        tuple([triangle[2][0], triangle[2][2]]),
                    ),
                    (
                        tuple([triangle[2][0], triangle[2][2]]),
                        tuple([triangle[0][0], triangle[0][2]]),
                    ),
                ]
                for segment in triangle_segments:
                    if segment not in segments and segment[::-1] not in segments:
                        segments.add(segment)
                    else:
                        if segment in segments:
                            segments.remove(segment)
                        else:
                            segments.remove(segment[::-1])

                segments_y[triangle[0][1]] = segments

            else:
                segments = segments_z.get(triangle[0][2], set())
                triangle_segments = [
                    (tuple(triangle[0][:2]), tuple(triangle[1][:2])),
                    (tuple(triangle[1][:2]), tuple(triangle[2][:2])),
                    (tuple(triangle[2][:2]), tuple(triangle[0][:2])),
                ]
                for segment in triangle_segments:
                    if segment not in segments and segment[::-1] not in segments:
                        segments.add(segment)
                    else:
                        if segment in segments:
                            segments.remove(segment)
                        else:
                            segments.remove(segment[::-1])

                segments_z[triangle[0][2]] = segments

        return segments_x, segments_y, segments_z

    @staticmethod
    def _order_segments(segments):
        segments = list(segments)
        ordered_segments_list = []

        while segments:
            # Start a new contour with the first segment
            ordered_segments = [segments.pop()]
            ordered_segments_list.append(ordered_segments)

            while True:
                last_segment = ordered_segments[-1]

                # Find the next segment
                for i, s in enumerate(segments):
                    if s[0] == last_segment[1]:
                        ordered_segments.append(segments.pop(i))
                        break

                    elif s[1] == last_segment[1]:
                        # If the segment is in the opposite direction, reverse it
                        ordered_segments.append((s[1], s[0]))
                        segments.pop(i)
                        break
                else:
                    # No more segments connect to this contour, start a new one
                    break

        return ordered_segments_list

    @staticmethod
    def _triangles_to_closed_polygons(triangles):
        segments_x, segments_y, segments_z = Voxelization._triangles_to_segments(triangles)
        polygons = [{}, {}, {}]

        for i, segments_i in enumerate([segments_x, segments_y, segments_z]):
            for absicssa, segments in segments_i.items():
                # Order the segments lists
                ordered_segments_list = Voxelization._order_segments(segments)

                # Split the self-intersecting polygons
                splitted_ordered_segments_list = []
                for ordered_segments in ordered_segments_list:
                    splitted_ordered_segments_list.extend(
                        Voxelization._split_ordered_segments(ordered_segments)
                    )

                polygons[i][absicssa] = [
                    Voxelization._ordered_segments_to_closed_polygon_2d(splitted_ordered_segments)
                    for splitted_ordered_segments in splitted_ordered_segments_list
                ]

        return polygons

    @staticmethod
    def _split_ordered_segments(ordered_segments):
        points = [segment[0] for segment in ordered_segments]

        if len(points) == len(set(points)):
            # If no duplicate points, the polygon is not self-intersecting
            return [ordered_segments]

        graph = Graph(ordered_segments)
        ordered_segments_list = []

        for cycle in graph.find_cycles():
            ordered_segments_list.extend(Voxelization._order_segments(cycle))
            # print(Voxelization._order_segments(cycle))

        polygons = {}
        for ordered_segments in ordered_segments_list:
            polygons[tuple(ordered_segments)] = Voxelization._ordered_segments_to_closed_polygon_2d(ordered_segments)

        # Check for polygon inclusion
        children = {}
        parents = {}
        for polygon in polygons.values():
            children[polygon] = []
            parents[polygon] = []

        for polygon_1 in polygons.values():
            for polygon_2 in polygons.values():
                if polygon_1 != polygon_2:
                    if polygon_1.is_inside(polygon_2):
                        children[polygon_1].append(polygon_2)
                        parents[polygon_2].append(polygon_1)
                    elif polygon_2.is_inside(polygon_1):
                        children[polygon_2].append(polygon_1)
                        parents[polygon_1].append(polygon_2)

        # Actual splited polygons have no childen in the cycle-defined polygons from the self-intersecting polygon
        candidate_polygons = {}
        for ordered_segments, polygon in polygons.items():
            if len(children[polygon]) == 0:
                candidate_polygons[ordered_segments] = polygon

        # If the polygon has all its segments part of the other polygons segments, it is an inner polygon
        ordered_segments_list = []
        for ordered_segments in candidate_polygons.keys():
            ordered_segments_list.append(list(ordered_segments))

        ordered_segments_list_valid = []
        for i in range(len(ordered_segments_list)):
            all_current_polygon_segments = []
            all_other_polygons_segments = []

            for segment in ordered_segments_list[i]:
                all_current_polygon_segments.append(segment)
                all_current_polygon_segments.append(segment[::-1])

            for segments in ordered_segments_list[:i] + ordered_segments_list[i+1:]:
                for segment in segments:
                    all_other_polygons_segments.append(segment)
                    all_other_polygons_segments.append(segment[::-1])

            for segment in all_current_polygon_segments:
                if segment not in all_other_polygons_segments:
                    ordered_segments_list_valid.append(ordered_segments_list[i])
                    break

        return ordered_segments_list_valid

    @staticmethod
    def _simplify_polygon_points(points):
        simplifyed_point = []

        if len(points) != len(set(points)):
            ClosedPolygon2D([Point2D(*point) for point in points]).plot()

        for i in range(len(points) - 1):
            if not (
                points[i - 1][0] == points[i][0] == points[i + 1][0]
                or points[i - 1][1] == points[i][1] == points[i + 1][1]
            ):
                simplifyed_point.append(points[i])

        if not (
            points[-2][0] == points[-1][0] == points[0][0]
            or points[-2][1] == points[-1][1] == points[0][1]
        ):
            simplifyed_point.append(points[-1])

        return simplifyed_point

    @staticmethod
    def _ordered_segments_to_closed_polygon_2d(ordered_segments):
        return ClosedPolygon2D(
            points=[
                Point2D(*point)
                for point in Voxelization._simplify_polygon_points(
                    [segment[0] for segment in ordered_segments]
                )
            ]
        )

    def to_triangles(self) -> Set[Triangle]:
        triangles = set()

        for voxel in tqdm(self.voxels):
            for triangle in self._voxel_faces(voxel, self.voxel_size):
                if triangle not in triangles:
                    triangles.add(triangle)
                else:
                    triangles.remove(triangle)

        return triangles

    def to_closed_shell(self) -> ClosedShell3D:
        polygons = self._triangles_to_closed_polygons(self.to_triangles())
        planes = [PLANE3D_OYZ, PLANE3D_OXZ, PLANE3D_OXY]
        faces = []

        for i, polygons_i in enumerate(polygons):
            for abscissa, polygons in polygons_i.items():
                offset = [0, 0, 0]
                offset[i] = abscissa
                offset = Vector3D(*offset)

                plane = planes[i].translation(offset)

                if len(polygons) == 1:
                    faces.append(PlaneFace3D(plane, Surface2D(polygons[0], [])))
                else:
                    # Check for polygon inclusion
                    children = {}
                    parents = {}
                    for polygon in polygons:
                        children[polygon] = []
                        parents[polygon] = []

                    for polygon_1 in polygons:
                        for polygon_2 in polygons:
                            if polygon_1 != polygon_2:
                                if polygon_1.is_inside(polygon_2):
                                    children[polygon_1].append(polygon_2)
                                    parents[polygon_2].append(polygon_1)
                                elif polygon_2.is_inside(polygon_1):
                                    children[polygon_2].append(polygon_1)
                                    parents[polygon_1].append(polygon_2)

                    for polygon in polygons:
                        if len(parents[polygon]) // 2 == 0:
                            inner_polygons = [
                                child
                                for child in children[polygon]
                                if set(children[polygon]).intersection(children[child]) == set()
                            ]
                            faces.append(PlaneFace3D(plane, Surface2D(polygon, inner_polygons)))

        return ClosedShell3D(faces, name=self.name)

    def to_closed_triangle_shell(self) -> ClosedTriangleShell3D:
        triangles3d = [
            Triangle3D(Point3D(*triangle[0]), Point3D(*triangle[1]), Point3D(*triangle[2]))
            for triangle in tqdm(self.to_triangles())
        ]
        shell = ClosedTriangleShell3D(triangles3d, name=self.name)

        return shell

    def volmdlr_primitives(self, **kwargs):
        if len(self.voxels) == 0:
            warnings.warn("Empty voxelization.")
            return []

        return [self.to_closed_shell()]

    def intersection(self, other_voxelization: "Voxelization") -> "Voxelization":
        if self.voxel_size != other_voxelization.voxel_size:
            raise ValueError(
                "Both voxelizations must have same voxel_size to perform intersection."
            )

        return Voxelization(self.voxels.intersection(other_voxelization.voxels), self.voxel_size)

    def union(self, other_voxelization: "Voxelization") -> "Voxelization":
        if self.voxel_size != other_voxelization.voxel_size:
            raise ValueError("Both voxelizations must have same voxel_size to perform union.")

        return Voxelization(self.voxels.union(other_voxelization.voxels), self.voxel_size)

    def difference(self, other_voxelization: "Voxelization") -> "Voxelization":
        if self.voxel_size != other_voxelization.voxel_size:
            raise ValueError("Both voxelizations must have same voxel_size to perform difference.")

        return Voxelization(self.voxels.difference(other_voxelization.voxels), self.voxel_size)

    @staticmethod
    def _round_point_cloud_voxel_size(point_cloud: np.ndarray, voxel_size: float) -> np.ndarray:
        """
        Round the coordinates of a point cloud dictionary to be the center point of the voxel it's in.
        """
        transformed_point_cloud = np.round((point_cloud / voxel_size) + 0.5, 6) * voxel_size
        transformed_point_cloud = np.round(transformed_point_cloud, 6)

        return transformed_point_cloud

    @staticmethod
    def _rotation_matrix(axis: Vector3D, angle: float):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        axis = np.array([axis.x, axis.y, axis.z])

        axis = axis / np.linalg.norm(axis)
        a = np.cos(angle / 2.0)
        b, c, d = -axis * np.sin(angle / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array(
            [
                [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
            ]
        )

    def rotation(self, center: Point3D, axis: Vector3D, angle: float):
        R = self._rotation_matrix(axis, angle)
        voxel_array = np.array(list(self.voxels)) - np.array([center.x, center.y, center.z])
        rotated_voxels = np.dot(voxel_array, R.T)
        rotated_voxels += np.array([center.x, center.y, center.z])

        intersecting_voxels = self._intersecting_voxels(rotated_voxels, self.voxel_size)
        # intersecting_voxels = set(tuple(voxel) for voxel in self.round_point_cloud_voxel_size(rotated_voxels, self.voxel_size))
        # intersecting_voxels = self.surrounding_voxels(rotated_voxels, self.voxel_size)

        return Voxelization(intersecting_voxels, self.voxel_size)

    def translation(self, offset: Vector3D):
        voxel_array = np.array(list(self.voxels))
        translated_voxels = voxel_array + np.array([offset.x, offset.y, offset.z])

        intersecting_voxels = self._intersecting_voxels(translated_voxels, self.voxel_size)
        # intersecting_voxels = set(tuple(voxel) for voxel in self.round_point_cloud_voxel_size(translated_voxels, self.voxel_size))
        # intersecting_voxels = self.surrounding_voxels(translated_voxels, self.voxel_size)

        return Voxelization(intersecting_voxels, self.voxel_size)


class Graph:
    """Helper Class for subdividing self-crossing polygons."""
    def __init__(self, edges):
        self.graph = self.build_graph(edges)
        self.visited = {node: False for node in self.graph}

    @staticmethod
    def build_graph(edges):
        """Builds adjacency list representation of graph from edge list"""
        graph = {}
        for edge in edges:
            if edge[0] not in graph:
                graph[edge[0]] = []
            if edge[1] not in graph:
                graph[edge[1]] = []
            graph[edge[0]].append(edge[1])
            graph[edge[1]].append(edge[0])  # assuming undirected graph
        return graph

    def find_cycles(self):
        """Finds all unique cycles in the graph"""
        cycles = []
        for node in self.graph:
            self._dfs(node, node, [], cycles)
        return self._convert_to_edge_cycles(cycles)

    def _dfs(self, node, start, path, cycles):
        """Performs depth-first search to find cycles"""
        self.visited[node] = True
        path.append(node)
        for neighbor in self.graph[node]:
            if not self.visited[neighbor]:
                self._dfs(neighbor, start, path, cycles)
            elif neighbor == start and len(path) >= 3:
                cycles.append(path[:])
        path.pop()
        self.visited[node] = False

    @staticmethod
    def _convert_to_edge_cycles(cycles):
        """Converts node-based cycles to edge-based cycles, removing duplicates"""
        edge_cycles_set = set()
        for cycle in cycles:
            edge_cycle = []
            for i in range(len(cycle)):
                edge_cycle.append(tuple(sorted([cycle[i], cycle[(i + 1) % len(cycle)]])))
            edge_cycle.sort()  # sort the edges in the cycle
            edge_cycles_set.add(tuple(edge_cycle))  # add it to the set
        return [list(cycle) for cycle in edge_cycles_set]


class OctreeNode:
    def __init__(self, center: Point, size: float, depth: int, max_depth: int, count: int = 1):
        self.children = []
        self.center = center
        self.size = size
        self.depth = depth
        self.max_depth = max_depth
        if depth == 1:
            print(f"{count} of 8")

    def subdivide(self, mesh: Mesh):
        children = []
        if self.depth < self.max_depth:
            half_size = self.size / 2
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        # calculate the center of the child node
                        child_center = (
                            self.center[0] + (i - 0.5) * half_size,
                            self.center[1] + (j - 0.5) * half_size,
                            self.center[2] + (k - 0.5) * half_size,
                        )

                        # create a new OctreeNode for the child
                        count = k + 2 * j + 4 * i + 1
                        child_node = OctreeNode(
                            child_center,
                            half_size,
                            self.depth + 1,
                            self.max_depth,
                            count,
                        )

                        # check if the child node intersects with the mesh
                        if self.depth != 0:
                            # add the child node to the list of children
                            self.children.extend(self.process_node(child_node, mesh))
                        else:
                            children.append(child_node)

            if self.depth == 0:
                for node in tqdm(
                        self.process_node(node, mesh) for node in children
                ):
                    self.children.extend(node)

        else:
            return  # reached max depth, do not subdivide further.

    def intersecting_triangles(self, mesh: Mesh) -> List[Triangle]:
        intersecting_triangles = [
            triangle
            for triangle in mesh
            if Voxelization._triangle_aabb_intersection(np.array(triangle), np.array(self.center), np.array([self.size for _ in range(3)]))
        ]

        return intersecting_triangles

    def get_leaf_centers(self):
        if self.depth == self.max_depth:  # if the node has no children, it is a leaf node
            return [self.center]
        else:
            centers = []
            for child in self.children:
                centers += child.get_leaf_centers()
            return centers

    @staticmethod
    def process_node(node, mesh):
        triangles = node.intersecting_triangles(mesh)
        if len(triangles) > 0:
            # recursively subdivide the child node
            node.subdivide(triangles)
            return [node]
        return []