"""volmdlr module for 3D Surfaces."""
import math
import warnings
from itertools import chain
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as npy
import triangle as triangle_lib
from geomdl import NURBS, BSpline, utilities
from geomdl.construct import extract_curves
from geomdl.fitting import approximate_surface, interpolate_surface
from geomdl.operations import split_surface_u, split_surface_v
from scipy.optimize import least_squares, minimize


from dessia_common.core import DessiaObject
from volmdlr.bspline_evaluators import evaluate_single
import volmdlr.bspline_compiled
import volmdlr.core_compiled
import volmdlr.core
from volmdlr import display, edges, grid, wires
import volmdlr.geometry
import volmdlr.utils.parametric as vm_parametric
from volmdlr.core import EdgeStyle
from volmdlr.core import point_in_list
from volmdlr.utils.parametric import array_range_search, repair_start_end_angle_periodicity, angle_discontinuity


def knots_vector_inv(knots_vector):
    """
    Compute knot-elements and multiplicities based on the global knot vector.

    """

    knots = sorted(set(knots_vector))
    multiplicities = [knots_vector.count(knot) for knot in knots]

    return knots, multiplicities


class Surface2D(volmdlr.core.Primitive2D):
    """
    A surface bounded by an outer contour.

    """

    def __init__(self, outer_contour: wires.Contour2D,
                 inner_contours: List[wires.Contour2D],
                 name: str = 'name'):
        self.outer_contour = outer_contour
        self.inner_contours = inner_contours
        self._area = None

        volmdlr.core.Primitive2D.__init__(self, name=name)

    def __hash__(self):
        return hash((self.outer_contour, tuple(self.inner_contours)))

    def _data_hash(self):
        return hash(self)

    def copy(self):
        """
        Copies the surface2d.

        """
        return self.__class__(outer_contour=self.outer_contour.copy(),
                              inner_contours=[c.copy() for c in self.inner_contours],
                              name=self.name)

    def area(self):
        """
        Computes the area of the surface.

        """
        if not self._area:
            self._area = self.outer_contour.area() - sum(contour.area() for contour in self.inner_contours)
        return self._area

    def second_moment_area(self, point: volmdlr.Point2D):
        """
        Computes the second moment area of the surface.

        """
        i_x, i_y, i_xy = self.outer_contour.second_moment_area(point)
        for contour in self.inner_contours:
            i_xc, i_yc, i_xyc = contour.second_moment_area(point)
            i_x -= i_xc
            i_y -= i_yc
            i_xy -= i_xyc
        return i_x, i_y, i_xy

    def center_of_mass(self):
        """
        Compute the center of mass of the 2D surface.

        :return: The center of mass of the surface.
        :rtype: :class:`volmdlr.Point2D`
        """
        center = self.outer_contour.area() * self.outer_contour.center_of_mass()
        for contour in self.inner_contours:
            center -= contour.area() * contour.center_of_mass()
        return center / self.area()

    def point_belongs(self, point2d: volmdlr.Point2D):
        """
        Check whether a point belongs to the 2D surface.

        :param point2d: The point to check.
        :type point2d: :class:`volmdlr.Point2D`
        :return: True if the point belongs to the surface, False otherwise.
        :rtype: bool
        """
        if not self.outer_contour.point_belongs(point2d):
            if self.outer_contour.point_over_contour(point2d):
                return True
            return False

        for inner_contour in self.inner_contours:
            if inner_contour.point_belongs(point2d) and not inner_contour.point_over_contour(point2d):
                return False
        return True

    def random_point_inside(self):
        """
        Generate a random point inside the 2D surface.

        Taking into account any inner contours (holes) it may have.

        :return: A random point inside the surface.
        :rtype: :class:`volmdlr.Point2D`
        """
        valid_point = False
        point_inside_outer_contour = None
        while not valid_point:
            point_inside_outer_contour = self.outer_contour.random_point_inside()
            inside_inner_contour = False
            for inner_contour in self.inner_contours:
                if inner_contour.point_belongs(point_inside_outer_contour):
                    inside_inner_contour = True
            if not inside_inner_contour and \
                    point_inside_outer_contour is not None:
                valid_point = True

        return point_inside_outer_contour

    @staticmethod
    def triangulation_without_holes(vertices, segments, points_grid, tri_opt):
        """
        Triangulates a surface without holes.

        :param vertices: vertices of the surface.
        :param segments: segments defined as tuples of vertices.
        :param points_grid: to do.
        :param tri_opt: triangulation option: "p"
        :return:
        """
        vertices_grid = [(p.x, p.y) for p in points_grid]
        vertices.extend(vertices_grid)
        tri = {'vertices': npy.array(vertices).reshape((-1, 2)),
               'segments': npy.array(segments).reshape((-1, 2)),
               }
        triagulation = triangle_lib.triangulate(tri, tri_opt)
        triangles = triagulation['triangles'].tolist()
        number_points = triagulation['vertices'].shape[0]
        points = [display.Node2D(*triagulation['vertices'][i, :]) for i in range(number_points)]
        return display.DisplayMesh2D(points, triangles=triangles)

    def triangulation(self, number_points_x: int = 15, number_points_y: int = 15):
        """
        Triangulates the Surface2D using the Triangle library.

        :param number_points_x: Number of discretization points in x direction.
        :type number_points_x: int
        :param number_points_y: Number of discretization points in y direction.
        :type number_points_y: int
        :return: The triangulated surface as a display mesh.
        :rtype: :class:`volmdlr.display.DisplayMesh2D`
        """
        area = self.bounding_rectangle().area()
        tri_opt = "p"
        if math.isclose(area, 0., abs_tol=1e-6):
            return display.DisplayMesh2D([], triangles=[])

        triangulates_with_grid = number_points_x > 0 or number_points_y > 0

        outer_polygon = self.outer_contour.to_polygon(angle_resolution=15, discretize_line=triangulates_with_grid)

        if not self.inner_contours and not triangulates_with_grid:
            return outer_polygon.triangulation()

        points_grid, x, y, grid_point_index = outer_polygon.grid_triangulation_points(number_points_x=number_points_x,
                                                                                      number_points_y=number_points_y)
        points = [display.Node2D(*point) for point in outer_polygon.points]
        vertices = [(point.x, point.y) for point in points]
        n = len(points)
        segments = [(i, i + 1) for i in range(n - 1)]
        segments.append((n - 1, 0))

        if not self.inner_contours:  # No holes
            return self.triangulation_without_holes(vertices, segments, points_grid, tri_opt)

        point_index = {p: i for i, p in enumerate(points)}
        holes = []
        for inner_contour in self.inner_contours:
            inner_polygon = inner_contour.to_polygon(angle_resolution=10, discretize_line=triangulates_with_grid)
            inner_polygon_nodes = [display.Node2D.from_point(p) for p in inner_polygon.points]
            for point in inner_polygon_nodes:
                if point not in point_index:
                    points.append(point)
                    vertices.append((point.x, point.y))
                    point_index[point] = n
                    n += 1

            for point1, point2 in zip(inner_polygon_nodes[:-1],
                                      inner_polygon_nodes[1:]):
                segments.append((point_index[point1], point_index[point2]))
            segments.append((point_index[inner_polygon_nodes[-1]], point_index[inner_polygon_nodes[0]]))
            rpi = inner_polygon.barycenter()
            if not inner_polygon.point_belongs(rpi, include_edge_points=False):
                rpi = inner_polygon.random_point_inside(include_edge_points=False)
            holes.append([rpi.x, rpi.y])

            if triangulates_with_grid:
                # removes with a region search the grid points that are in the inner contour
                xmin, xmax, ymin, ymax = inner_polygon.bounding_rectangle.bounds()
                x_grid_range = array_range_search(x, xmin, xmax)
                y_grid_range = array_range_search(y, ymin, ymax)
                for i in x_grid_range:
                    for j in y_grid_range:
                        point = grid_point_index.get((i, j))
                        if not point:
                            continue
                        if inner_polygon.point_belongs(point):
                            points_grid.remove(point)
                            grid_point_index.pop((i, j))

        if triangulates_with_grid:
            vertices_grid = [(p.x, p.y) for p in points_grid]
            vertices.extend(vertices_grid)

        tri = {'vertices': npy.array(vertices).reshape((-1, 2)),
               'segments': npy.array(segments).reshape((-1, 2)),
               'holes': npy.array(holes).reshape((-1, 2))
               }
        triangulation = triangle_lib.triangulate(tri, tri_opt)
        triangles = triangulation['triangles'].tolist()
        number_points = triangulation['vertices'].shape[0]
        points = [display.Node2D(*triangulation['vertices'][i, :]) for i in range(number_points)]
        return display.DisplayMesh2D(points, triangles=triangles)

    def split_by_lines(self, lines):
        """
        Returns a list of cut surfaces given by the lines provided as argument.
        """
        cutted_surfaces = []
        iteration_surfaces = self.cut_by_line(lines[0])

        for line in lines[1:]:
            iteration_surfaces2 = []
            for surface in iteration_surfaces:
                line_cutted_surfaces = surface.cut_by_line(line)

                llcs = len(line_cutted_surfaces)

                if llcs == 1:
                    cutted_surfaces.append(line_cutted_surfaces[0])
                else:
                    iteration_surfaces2.extend(line_cutted_surfaces)

            iteration_surfaces = iteration_surfaces2[:]

        cutted_surfaces.extend(iteration_surfaces)
        return cutted_surfaces

    def split_regularly(self, n):
        """
        Split in n slices.
        """
        bounding_rectangle = self.outer_contour.bounding_rectangle
        lines = []
        for i in range(n - 1):
            xi = bounding_rectangle[0] + (i + 1) * (bounding_rectangle[1] - bounding_rectangle[0]) / n
            lines.append(edges.Line2D(volmdlr.Point2D(xi, 0),
                                    volmdlr.Point2D(xi, 1)))
        return self.split_by_lines(lines)

    def cut_by_line(self, line: edges.Line2D):
        """
        Returns a list of cut Surface2D by the given line.

        :param line: The line to cut the Surface2D with.
        :type line: :class:`edges.Line2D`
        :return: A list of 2D surfaces resulting from the cut.
        :rtype: List[:class:`volmdlr.faces.Surface2D`]
        """
        surfaces = []
        splitted_outer_contours = self.outer_contour.cut_by_line(line)
        splitted_inner_contours_table = []
        for inner_contour in self.inner_contours:
            splitted_inner_contours = inner_contour.cut_by_line(line)
            splitted_inner_contours_table.append(splitted_inner_contours)

        # First part of the external contour
        for outer_split in splitted_outer_contours:
            inner_contours = []
            for splitted_inner_contours in splitted_inner_contours_table:
                for inner_split in splitted_inner_contours:
                    inner_split.order_contour()
                    point = inner_split.random_point_inside()
                    if outer_split.point_belongs(point):
                        inner_contours.append(inner_split)

            if inner_contours:
                surface2d = self.from_contours(outer_split, inner_contours)
                surfaces.append(surface2d)
            else:
                surfaces.append(Surface2D(outer_split, []))
        return surfaces

    def line_crossings(self, line: edges.Line2D):
        """
        Find intersection points between a line and the 2D surface.

        :param line: The line to intersect with the shape.
        :type line: :class:`edges.Line2D`
        :return: A list of intersection points sorted by increasing abscissa
            along the line. Each intersection point is a tuple
            (point, primitive) where point is the intersection point and
            primitive is the intersected primitive.
        :rtype: List[Tuple[:class:`volmdlr.Point2D`,
            :class:`volmdlr.core.Primitive2D`]]

        """
        intersection_points = []
        for primitive in self.outer_contour.primitives:
            for point in primitive.line_crossings(line):
                if (point, primitive) not in intersection_points:
                    intersection_points.append((point, primitive))
        for inner_contour in self.inner_contours:
            for primitive in inner_contour.primitives:
                for point in primitive.line_crossings(line):
                    if (point, primitive) not in intersection_points:
                        intersection_points.append((point, primitive))
        return sorted(intersection_points, key=lambda ip: line.abscissa(ip[0]))

    def split_at_centers(self):
        """
        Split in n slices.

        # TODO: is this used ?
        """

        cutted_contours = []
        center_of_mass1 = self.inner_contours[0].center_of_mass()
        center_of_mass2 = self.inner_contours[1].center_of_mass()
        cut_line = edges.Line2D(center_of_mass1, center_of_mass2)

        iteration_contours2 = []

        surface_cut = self.cut_by_line(cut_line)

        iteration_contours2.extend(surface_cut)

        iteration_contours = iteration_contours2[:]
        cutted_contours.extend(iteration_contours)

        return cutted_contours

    def cut_by_line2(self, line):
        """
        Cuts a Surface2D with line (2).

        :param line: DESCRIPTION
        :type line: TYPE
        :raises NotImplementedError: DESCRIPTION
        :return: DESCRIPTION
        :rtype: TYPE

        """

        all_contours = []
        inner_1 = self.inner_contours[0]
        inner_2 = self.inner_contours[1]

        inner_intersections_1 = inner_1.line_intersections(line)
        inner_intersections_2 = inner_2.line_intersections(line)

        arc1, arc2 = inner_1.split(inner_intersections_1[1],
                                   inner_intersections_1[0])
        arc3, arc4 = inner_2.split(inner_intersections_2[1],
                                   inner_intersections_2[0])
        new_inner_1 = wires.Contour2D([arc1, arc2])
        new_inner_2 = wires.Contour2D([arc3, arc4])

        intersections = [(inner_intersections_1[0], arc1), (inner_intersections_1[1], arc2)]
        intersections += self.outer_contour.line_intersections(line)
        intersections.append((inner_intersections_2[0], arc3))
        intersections.append((inner_intersections_2[1], arc4))
        intersections += self.outer_contour.line_intersections(line)

        if not intersections:
            all_contours.extend([self])
        if len(intersections) < 4:
            return [self]
        if len(intersections) >= 4:
            if isinstance(intersections[0][0], volmdlr.Point2D) and \
                    isinstance(intersections[1][0], volmdlr.Point2D):
                ip1, ip2 = sorted(
                    [new_inner_1.primitives.index(intersections[0][1]),
                     new_inner_1.primitives.index(intersections[1][1])])
                ip5, ip6 = sorted(
                    [new_inner_2.primitives.index(intersections[4][1]),
                     new_inner_2.primitives.index(intersections[5][1])])
                ip3, ip4 = sorted(
                    [self.outer_contour.primitives.index(intersections[2][1]),
                     self.outer_contour.primitives.index(intersections[3][1])])

                # sp11, sp12 = intersections[2][1].split(intersections[2][0])
                # sp21, sp22 = intersections[3][1].split(intersections[3][0])
                sp33, sp34 = intersections[6][1].split(intersections[6][0])
                sp44, sp43 = intersections[7][1].split(intersections[7][0])

                primitives1 = [edges.LineSegment2D(intersections[6][0], intersections[1][0]),
                               new_inner_1.primitives[ip1],
                               edges.LineSegment2D(intersections[0][0], intersections[5][0]),
                               new_inner_2.primitives[ip5],
                               edges.LineSegment2D(intersections[4][0], intersections[7][0]),
                               sp44
                               ]
                primitives1.extend(self.outer_contour.primitives[ip3 + 1:ip4])
                primitives1.append(sp34)

                primitives2 = [edges.LineSegment2D(intersections[7][0], intersections[4][0]),
                               new_inner_2.primitives[ip6],
                               edges.LineSegment2D(intersections[5][0], intersections[0][0]),
                               new_inner_1.primitives[ip2],
                               edges.LineSegment2D(intersections[1][0], intersections[6][0]),
                               sp33
                               ]

                primitives2.extend(self.outer_contour.primitives[:ip3].reverse())
                primitives2.append(sp43)

                all_contours.extend([wires.Contour2D(primitives1),
                                     wires.Contour2D(primitives2)])

            else:
                raise NotImplementedError(
                    'Non convex contour not supported yet')

        return all_contours

    def bounding_rectangle(self):
        """
        Returns bounding rectangle.

        :return: Returns a python object with useful methods
        :rtype: :class:`volmdlr.core.BoundingRectangle
        """

        return self.outer_contour.bounding_rectangle

    @classmethod
    def from_contours(cls, outer_contour, inner_contours):
        """
        Create a Surface2D object from an outer contour and a list of inner contours.

        :param outer_contour: The outer contour that bounds the surface.
        :type outer_contour: wires.Contour2D
        :param inner_contours: The list of inner contours that define the holes of the surface.
        :type inner_contours : List[wires.Contour2D]
        :return: Surface2D defined by the given contours.
        """
        surface2d_inner_contours = []
        surface2d_outer_contour = outer_contour
        for inner_contour in inner_contours:
            if surface2d_outer_contour.shared_primitives_extremities(
                    inner_contour):
                # inner_contour will be merged with outer_contour
                merged_contours = surface2d_outer_contour.merge_with(
                    inner_contour)
                if len(merged_contours) >= 2:
                    raise NotImplementedError
                surface2d_outer_contour = merged_contours[0]
            else:
                # inner_contour will be added to the inner contours of the
                # Surface2D
                surface2d_inner_contours.append(inner_contour)
        return cls(surface2d_outer_contour, surface2d_inner_contours)

    def plot(self, ax=None, color='k', alpha=1, equal_aspect=False):

        if ax is None:
            _, ax = plt.subplots()
        self.outer_contour.plot(ax=ax, edge_style=EdgeStyle(color=color, alpha=alpha,
                                                            equal_aspect=equal_aspect))
        for inner_contour in self.inner_contours:
            inner_contour.plot(ax=ax, edge_style=EdgeStyle(color=color, alpha=alpha,
                                                           equal_aspect=equal_aspect))

        if equal_aspect:
            ax.set_aspect('equal')

        ax.margins(0.1)
        return ax

    def axial_symmetry(self, line):
        """
        Finds out the symmetric 2D surface according to a line.

        """

        outer_contour = self.outer_contour.axial_symmetry(line)
        inner_contours = []
        if self.inner_contours:
            inner_contours = [contour.axial_symmetry(line) for contour in self.inner_contours]

        return self.__class__(outer_contour=outer_contour,
                              inner_contours=inner_contours)

    def rotation(self, center, angle):
        """
        Surface2D rotation.

        :param center: rotation center.
        :param angle: angle rotation.
        :return: a new rotated Surface2D.
        """

        outer_contour = self.outer_contour.rotation(center, angle)
        if self.inner_contours:
            inner_contours = [contour.rotation(center, angle) for contour in self.inner_contours]
        else:
            inner_contours = []

        return self.__class__(outer_contour, inner_contours)

    def rotation_inplace(self, center, angle):
        """
        Rotate the surface in-place.
        """
        warnings.warn("'in-place' methods are deprecated. Use a not in-place method instead.", DeprecationWarning)

        new_surface2d = self.rotation(center, angle)
        self.outer_contour = new_surface2d.outer_contour
        self.inner_contours = new_surface2d.inner_contours

    def translation(self, offset: volmdlr.Vector2D):
        """
        Surface2D translation.

        :param offset: translation vector.
        :return: A new translated Surface2D.
        """
        outer_contour = self.outer_contour.translation(offset)
        inner_contours = [contour.translation(offset) for contour in self.inner_contours]
        return self.__class__(outer_contour, inner_contours)

    def translation_inplace(self, offset: volmdlr.Vector2D):
        warnings.warn("'in-place' methods are deprecated. Use a not in-place method instead.", DeprecationWarning)

        new_contour = self.translation(offset)
        self.outer_contour = new_contour.outer_contour
        self.inner_contours = new_contour.inner_contours

    def frame_mapping(self, frame: volmdlr.Frame2D, side: str):
        outer_contour = self.outer_contour.frame_mapping(frame, side)
        inner_contours = [contour.frame_mapping(frame, side) for contour in self.inner_contours]
        return self.__class__(outer_contour, inner_contours)

    def frame_mapping_inplace(self, frame: volmdlr.Frame2D, side: str):
        warnings.warn("'in-place' methods are deprecated. Use a not in-place method instead.", DeprecationWarning)

        new_contour = self.frame_mapping(frame, side)
        self.outer_contour = new_contour.outer_contour
        self.inner_contours = new_contour.inner_contours

    def geo_lines(self):  # , mesh_size_list=None):
        """
        Gets the lines that define a Surface2D in a .geo file.
        """

        i, i_p = None, None
        lines, line_surface, lines_tags = [], [], []
        point_account, line_account, line_loop_account = 0, 0, 1
        for outer_contour, contour in enumerate(list(chain(*[[self.outer_contour], self.inner_contours]))):

            if isinstance(contour, wires.Circle2D):
                points = [volmdlr.Point2D(contour.center.x - contour.radius, contour.center.y),
                          contour.center,
                          volmdlr.Point2D(contour.center.x + contour.radius, contour.center.y)]
                index = []
                for i, point in enumerate(points):
                    lines.append(point.get_geo_lines(tag=point_account + i + 1,
                                                     point_mesh_size=None))
                    index.append(point_account + i + 1)

                lines.append('Circle(' + str(line_account + 1) +
                             ') = {' + str(index[0]) + ', ' + str(index[1]) + ', ' + str(index[2]) + '};')
                lines.append('Circle(' + str(line_account + 2) +
                             ') = {' + str(index[2]) + ', ' + str(index[1]) + ', ' + str(index[0]) + '};')

                lines_tags.append(line_account + 1)
                lines_tags.append(line_account + 2)

                lines.append('Line Loop(' + str(outer_contour + 1) + ') = {' + str(lines_tags)[1:-1] + '};')
                line_surface.append(line_loop_account)

                point_account = point_account + 2 + 1
                line_account, line_loop_account = line_account + 1 + 1, line_loop_account + 1
                lines_tags = []

            elif isinstance(contour, (wires.Contour2D, wires.ClosedPolygon2D)):
                if not isinstance(contour, wires.ClosedPolygon2D):
                    contour = contour.to_polygon(1)
                for i, point in enumerate(contour.points):
                    lines.append(point.get_geo_lines(tag=point_account + i + 1,
                                                     point_mesh_size=None))

                for i_p, primitive in enumerate(contour.primitives):
                    if i_p != len(contour.primitives) - 1:
                        lines.append(primitive.get_geo_lines(tag=line_account + i_p + 1,
                                                             start_point_tag=point_account + i_p + 1,
                                                             end_point_tag=point_account + i_p + 2))
                    else:
                        lines.append(primitive.get_geo_lines(tag=line_account + i_p + 1,
                                                             start_point_tag=point_account + i_p + 1,
                                                             end_point_tag=point_account + 1))
                    lines_tags.append(line_account + i_p + 1)

                lines.append('Line Loop(' + str(outer_contour + 1) + ') = {' + str(lines_tags)[1:-1] + '};')
                line_surface.append(line_loop_account)
                point_account = point_account + i + 1
                line_account, line_loop_account = line_account + i_p + 1, line_loop_account + 1
                lines_tags = []

        lines.append('Plane Surface(' + str(1) + ') = {' + str(line_surface)[1:-1] + '};')

        return lines

    def mesh_lines(self,
                   factor: float,
                   curvature_mesh_size: int = None,
                   min_points: int = None,
                   initial_mesh_size: float = 5):
        """
        Gets the lines that define mesh parameters for a Surface2D, to be added to a .geo file.

        :param factor: A float, between 0 and 1, that describes the mesh quality
        (1 for coarse mesh - 0 for fine mesh)
        :type factor: float
        :param curvature_mesh_size: Activate the calculation of mesh element sizes based on curvature
        (with curvature_mesh_size elements per 2*Pi radians), defaults to 0
        :type curvature_mesh_size: int, optional
        :param min_points: Check if there are enough points on small edges (if it is not, we force to have min_points
        on that edge), defaults to None
        :type min_points: int, optional
        :param initial_mesh_size: If factor=1, it will be initial_mesh_size elements per dimension, defaults to 5
        :type initial_mesh_size: float, optional

        :return: A list of lines that describe mesh parameters
        :rtype: List[str]
        """

        lines = []
        if factor == 0:
            factor = 1e-3

        size = (math.sqrt(self.area()) / initial_mesh_size) * factor

        if min_points:
            primitives, primitives_length = [], []
            for _, contour in enumerate(list(chain(*[[self.outer_contour], self.inner_contours]))):
                if isinstance(contour, wires.Circle2D):
                    primitives.append(contour)
                    primitives.append(contour)
                    primitives_length.append(contour.length() / 2)
                    primitives_length.append(contour.length() / 2)
                else:
                    for primitive in contour.primitives:
                        if ((primitive not in primitives)
                                and (primitive.reverse() not in primitives)):
                            primitives.append(primitive)
                            primitives_length.append(primitive.length())

            for i, length in enumerate(primitives_length):
                if length < min_points * size:
                    lines.append('Transfinite Curve {' + str(i) + '} = ' + str(min_points) + ' Using Progression 1;')

        lines.append('Field[1] = MathEval;')
        lines.append('Field[1].F = "' + str(size) + '";')
        lines.append('Background Field = 1;')
        if curvature_mesh_size:
            lines.append('Mesh.MeshSizeFromCurvature = ' + str(curvature_mesh_size) + ';')

        # lines.append('Coherence;')

        return lines

    def to_geo(self, file_name: str,
               factor: float, **kwargs):
        # curvature_mesh_size: int = None,
        # min_points: int = None,
        # initial_mesh_size: float = 5):
        """
        Gets the .geo file for the Surface2D.
        """

        for element in [('curvature_mesh_size', 0), ('min_points', None), ('initial_mesh_size', 5)]:
            if element[0] not in kwargs:
                kwargs[element[0]] = element[1]

        lines = self.geo_lines()
        lines.extend(self.mesh_lines(factor, kwargs['curvature_mesh_size'],
                                     kwargs['min_points'], kwargs['initial_mesh_size']))

        with open(file_name + '.geo', 'w', encoding="utf-8") as file:
            for line in lines:
                file.write(line)
                file.write('\n')
        file.close()

    def to_msh(self, file_name: str, mesh_dimension: int,
               factor: float, **kwargs):
        # curvature_mesh_size: int = 0,
        # min_points: int = None,
        # initial_mesh_size: float = 5):
        """
        Gets .msh file for the Surface2D generated by gmsh.

        :param file_name: The msh. file name
        :type file_name: str
        :param mesh_dimension: The mesh dimension (1: 1D-Edge, 2: 2D-Triangle, 3D-Tetrahedra)
        :type mesh_dimension: int
        :param factor: A float, between 0 and 1, that describes the mesh quality
        (1 for coarse mesh - 0 for fine mesh)
        :type factor: float
        :param curvature_mesh_size: Activate the calculation of mesh element sizes based on curvature
        (with curvature_mesh_size elements per 2*Pi radians), defaults to 0
        :type curvature_mesh_size: int, optional
        :param min_points: Check if there are enough points on small edges (if it is not, we force to have min_points
        on that edge), defaults to None
        :type min_points: int, optional
        :param initial_mesh_size: If factor=1, it will be initial_mesh_size elements per dimension, defaults to 5
        :type initial_mesh_size: float, optional

        :return: A txt file
        :rtype: .txt
        """

        for element in [('curvature_mesh_size', 0), ('min_points', None), ('initial_mesh_size', 5)]:
            if element[0] not in kwargs:
                kwargs[element[0]] = element[1]

        self.to_geo(file_name=file_name, mesh_dimension=mesh_dimension,
                    factor=factor, curvature_mesh_size=kwargs['curvature_mesh_size'],
                    min_points=kwargs['min_points'], initial_mesh_size=kwargs['initial_mesh_size'])

        volmdlr.core.VolumeModel.generate_msh_file(file_name, mesh_dimension)

        # gmsh.initialize()
        # gmsh.open(file_name + ".geo")

        # gmsh.model.geo.synchronize()
        # gmsh.model.mesh.generate(mesh_dimension)

        # gmsh.write(file_name + ".msh")

        # gmsh.finalize()


class Surface3D(DessiaObject):
    """
    Abstract class.

    """
    x_periodicity = None
    y_periodicity = None
    face_class = None

    def point2d_to_3d(self, point2d):
        raise NotImplementedError(f'point2d_to_3d is abstract and should be implemented in {self.__class__.__name__}')

    def point3d_to_2d(self, point3d):
        """
        Abstract method. Convert a 3D point to a 2D parametric point.

        :param point3d: The 3D point to convert, represented by 3 coordinates (x, y, z).
        :type point3d: `volmdlr.Point3D`
        :return: NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError(f'point3d_to_2d is abstract and should be implemented in {self.__class__.__name__}')

    def face_from_contours3d(self, contours3d: List[wires.Contour3D], name: str = ''):
        """
        Returns the face generated by a list of contours. Finds out which are outer or inner contours.

        :param name: the name to inject in the new face
        """

        lc3d = len(contours3d)

        if lc3d == 1:
            outer_contour2d = self.contour3d_to_2d(contours3d[0])
            inner_contours2d = []
        elif lc3d > 1:
            area = -1
            inner_contours2d = []

            contours2d = [self.contour3d_to_2d(contour3d) for contour3d in contours3d]

            check_contours = [not contour2d.is_ordered(tol=1e-3) for contour2d in contours2d]
            if any(check_contours):
                outer_contour2d, inner_contours2d = self.repair_contours2d(contours2d[0], contours2d[1:])
            else:
                for contour2d in contours2d:
                    if not contour2d.is_ordered(1e-4):
                        contour2d = vm_parametric.contour2d_healing(contour2d)
                    inner_contours2d.append(contour2d)
                    contour_area = contour2d.area()
                    if contour_area > area:
                        area = contour_area
                        outer_contour2d = contour2d
                inner_contours2d.remove(outer_contour2d)
        else:
            raise ValueError('Must have at least one contour')

        if isinstance(self.face_class, str):
            # class_ = globals()[self.face_class]
            class_ = getattr(volmdlr.faces, self.face_class)
        else:
            class_ = self.face_class
        if not outer_contour2d.is_ordered(1e-4):
            outer_contour2d = vm_parametric.contour2d_healing(outer_contour2d)
        surface2d = Surface2D(outer_contour=outer_contour2d,
                              inner_contours=inner_contours2d)
        return class_(self, surface2d=surface2d, name=name)

    def repair_primitives_periodicity(self, primitives2d):
        """
        Repairs the continuity of the 2D contour while using contour3d_to_2d on periodic surfaces.

        :param primitives2d: The primitives in parametric surface domain.
        :type primitives2d: list
        :return: A list of primitives.
        :rtype: list
        """
        x_periodicity = self.x_periodicity
        y_periodicity = self.y_periodicity
        # Search for a primitive that can be used as reference for repairing periodicity
        if x_periodicity or y_periodicity:
            pos = vm_parametric.find_index_defined_brep_primitive_on_periodical_surface(primitives2d,
                                                                                        [x_periodicity, y_periodicity])
            if pos != 0:
                primitives2d = primitives2d[pos:] + primitives2d[:pos]

        i = 1
        if x_periodicity is None:
            x_periodicity = -1
        if y_periodicity is None:
            y_periodicity = -1
        while i < len(primitives2d):
            previous_primitive = primitives2d[i - 1]
            delta = previous_primitive.end - primitives2d[i].start
            is_connected = math.isclose(delta.norm(), 0, abs_tol=1e-3)
            if not is_connected and \
                    primitives2d[i].end.is_close(primitives2d[i - 1].end, tol=1e-3) and \
                    math.isclose(primitives2d[i].length(), x_periodicity, abs_tol=1e-5):
                primitives2d[i] = primitives2d[i].reverse()
            elif not is_connected and \
                    primitives2d[i].end.is_close(primitives2d[i - 1].end, tol=1e-3):
                primitives2d[i] = primitives2d[i].reverse()
            elif not is_connected:
                primitives2d[i] = primitives2d[i].translation(delta)
            i += 1

        return primitives2d

    def repair_contours2d(self, outer_contour, inner_contours):
        """
        Abstract method. Repair 2D contours of a face on the parametric domain.

        :param outer_contour: Outer contour 2D.
        :type inner_contours: wires.Contour2D
        :param inner_contours: List of 2D contours.
        :type inner_contours: list
        :return: NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError(f'repair_contours2d is abstract and should be implemented in '
                                  f'{self.__class__.__name__}')

    def primitives3d_to_2d(self, primitives3d):
        """
        Helper function to perform conversion of 3D primitives into B-Rep primitives.

        :param primitives3d: List of 3D primitives from a 3D contour.
        :type primitives3d: List[edges.Edge]
        :return: A list of 2D primitives on parametric domain.
        :rtype: List[edges.Edge]
        """
        primitives2d = []
        for primitive3d in primitives3d:
            method_name = f'{primitive3d.__class__.__name__.lower()}_to_2d'
            if hasattr(self, method_name):
                primitives = getattr(self, method_name)(primitive3d)

                if primitives is None:
                    continue
                primitives2d.extend(primitives)
            else:
                raise AttributeError(f'Class {self.__class__.__name__} does not implement {method_name}')
        return primitives2d

    def contour3d_to_2d(self, contour3d):
        """
        Transforms a Contour3D into a Contour2D in the parametric domain of the surface.

        :param contour3d: The contour to be transformed.
        :type contour3d: :class:`wires.Contour3D`
        :return: A 2D contour object.
        :rtype: :class:`wires.Contour2D`
        """
        primitives2d = self.primitives3d_to_2d(contour3d.primitives)

        wire2d = wires.Wire2D(primitives2d)
        delta_x = abs(wire2d.primitives[0].start.x - wire2d.primitives[-1].end.x)
        if math.isclose(delta_x, volmdlr.TWO_PI, abs_tol=1e-3) and wire2d.is_ordered():
            return wires.Contour2D(primitives2d)
        # Fix contour
        if self.x_periodicity or self.y_periodicity:
            primitives2d = self.repair_primitives_periodicity(primitives2d)
        return wires.Contour2D(primitives2d)

    def contour2d_to_3d(self, contour2d):
        """
        Transforms a Contour2D in the parametric domain of the surface into a Contour3D in Cartesian coordinate.

        :param contour2d: The contour to be transformed.
        :type contour2d: :class:`wires.Contour2D`
        :return: A 3D contour object.
        :rtype: :class:`wires.Contour3D`
        """
        primitives3d = []
        for primitive2d in contour2d.primitives:
            method_name = f'{primitive2d.__class__.__name__.lower()}_to_3d'
            if hasattr(self, method_name):
                try:
                    primitives = getattr(self, method_name)(primitive2d)
                    if primitives is None:
                        continue
                    primitives3d.extend(primitives)
                except AttributeError:
                    print(f'Class {self.__class__.__name__} does not implement {method_name}'
                          f'with {primitive2d.__class__.__name__}')
            else:
                raise AttributeError(
                    f'Class {self.__class__.__name__} does not implement {method_name}')

        return wires.Contour3D(primitives3d)

    def linesegment3d_to_2d(self, linesegment3d):
        """
        A line segment on a surface will be in any case a line in 2D?.

        """
        return [edges.LineSegment2D(self.point3d_to_2d(linesegment3d.start),
                                  self.point3d_to_2d(linesegment3d.end))]

    def bsplinecurve3d_to_2d(self, bspline_curve3d):
        """
        Is this right?.
        """
        n = len(bspline_curve3d.control_points)
        points = [self.point3d_to_2d(p)
                  for p in bspline_curve3d.discretization_points(number_points=n)]
        return [edges.BSplineCurve2D.from_points_interpolation(
            points, bspline_curve3d.degree, bspline_curve3d.periodic)]

    def bsplinecurve2d_to_3d(self, bspline_curve2d):
        """
        Is this right?.

        """
        n = len(bspline_curve2d.control_points)
        points = [self.point2d_to_3d(p)
                  for p in bspline_curve2d.discretization_points(number_points=n)]
        return [edges.BSplineCurve3D.from_points_interpolation(
            points, bspline_curve2d.degree, bspline_curve2d.periodic)]

    def normal_from_point2d(self, point2d):

        raise NotImplementedError('NotImplemented')

    def normal_from_point3d(self, point3d):
        """
        Evaluates the normal vector of the bspline surface at this 3D point.
        """

        return (self.normal_from_point2d(self.point3d_to_2d(point3d)))[1]

    def geodesic_distance_from_points2d(self, point1_2d: volmdlr.Point2D,
                                        point2_2d: volmdlr.Point2D, number_points: int = 50):
        """
        Approximation of geodesic distance via line segments length sum in 3D.
        """
        # points = [point1_2d]
        current_point3d = self.point2d_to_3d(point1_2d)
        distance = 0.
        for i in range(number_points):
            next_point3d = self.point2d_to_3d(point1_2d + (i + 1) / number_points * (point2_2d - point1_2d))
            distance += next_point3d.point_distance(current_point3d)
            current_point3d = next_point3d
        return distance

    def geodesic_distance(self, point1_3d: volmdlr.Point3D, point2_3d: volmdlr.Point3D):
        """
        Approximation of geodesic distance between 2 3D points supposed to be on the surface.
        """
        point1_2d = self.point3d_to_2d(point1_3d)
        point2_2d = self.point3d_to_2d(point2_3d)
        return self.geodesic_distance_from_points2d(point1_2d, point2_2d)

    def point_projection(self, point3d):
        """
        Returns the projection of the point on the surface.

        :param point3d: Point to project.
        :type point3d: volmdlr.Point3D
        :return: A point on the surface
        :rtype: volmdlr.Point3D
        """
        return self.point2d_to_3d(self.point3d_to_2d(point3d))

    def point_distance(self, point3d: volmdlr.Point3D):
        """
        Calculates the minimal distance from a given point and the surface.

        :param point3d: point to verify distance.
        :type point3d: volmdlr.Point3D
        :return: point distance to the surface.
        :rtype: float
        """
        proj_point = self.point_projection(point3d)
        return proj_point.point_distance(point3d)


class Plane3D(Surface3D):
    """
    Defines a plane 3d.

    :param frame: u and v of frame describe the plane, w is the normal
    """
    face_class = 'PlaneFace3D'

    def __init__(self, frame: volmdlr.Frame3D, name: str = ''):

        self.frame = frame
        self.name = name
        Surface3D.__init__(self, name=name)

    def __hash__(self):
        return hash(('plane 3d', self.frame))

    def __eq__(self, other_plane):
        if other_plane.__class__.__name__ != self.__class__.__name__:
            return False
        return self.frame == other_plane.frame

    @classmethod
    def from_step(cls, arguments, object_dict, **kwargs):
        """
        Converts a step primitive to a Plane3D.

        :param arguments: The arguments of the step primitive.
        :type arguments: list
        :param object_dict: The dictionary containing all the step primitives
            that have already been instantiated
        :type object_dict: dict
        :return: The corresponding Plane3D object.
        :rtype: :class:`volmdlr.faces.Plane3D`
        """
        frame3d = object_dict[arguments[1]]
        frame3d.normalize()
        frame = volmdlr.Frame3D(frame3d.origin,
                                frame3d.v, frame3d.w, frame3d.u)
        return cls(frame, arguments[0][1:-1])

    def to_step(self, current_id):
        frame = volmdlr.Frame3D(self.frame.origin, self.frame.w, self.frame.u,
                                self.frame.v)
        content, frame_id = frame.to_step(current_id)
        plane_id = frame_id + 1
        content += f"#{plane_id} = PLANE('{self.name}',#{frame_id});\n"
        return content, [plane_id]

    @classmethod
    def from_3_points(cls, *args):
        """
        Point 1 is used as origin of the plane.
        """
        point1, point2, point3 = args
        vector1 = point2 - point1
        vector2 = point3 - point1
        vector1 = vector1.to_vector()
        vector2 = vector2.to_vector()
        vector1.normalize()
        vector2.normalize()
        normal = vector1.cross(vector2)
        normal.normalize()
        frame = volmdlr.Frame3D(point1, vector1, normal.cross(vector1), normal)
        return cls(frame)

    @classmethod
    def from_normal(cls, point, normal):
        v1 = normal.deterministic_unit_normal_vector()
        v2 = v1.cross(normal)
        return cls(volmdlr.Frame3D(point, v1, v2, normal))

    @classmethod
    def from_plane_vectors(cls, plane_origin: volmdlr.Point3D, plane_x: volmdlr.Vector3D, plane_y: volmdlr.Vector3D):
        """
        Initializes a 3D plane object with a given plane origin and plane x and y vectors.

        :param plane_origin: A volmdlr.Point3D representing the origin of the plane.
        :param plane_x: A volmdlr.Vector3D representing the x-axis of the plane.
        :param plane_y: A volmdlr.Vector3D representing the y-axis of the plane.
        :return: A Plane3D object initialized from the provided plane origin and plane x and y vectors.
        """
        normal = plane_x.cross(plane_y)
        return cls(volmdlr.Frame3D(plane_origin, plane_x, plane_y, normal))

    @classmethod
    def from_points(cls, points):
        """
        Returns a 3D plane that goes through the 3 first points on the list.

        Why for more than 3 points we only do some check and never raise error?
        """
        if len(points) < 3:
            raise ValueError
        if len(points) == 3:
            return cls.from_3_points(points[0],
                                     points[1],
                                     points[2])
        points = [p.copy() for p in points]
        indexes_to_del = []
        for i, point in enumerate(points[1:]):
            if point.is_close(points[0]):
                indexes_to_del.append(i)
        for index in indexes_to_del[::-1]:
            del points[index + 1]

        origin = points[0]
        vector1 = points[1] - origin
        vector1.normalize()
        vector2_min = points[2] - origin
        vector2_min.normalize()
        dot_min = abs(vector1.dot(vector2_min))
        for point in points[3:]:
            vector2 = point - origin
            vector2.normalize()
            dot = abs(vector1.dot(vector2))
            if dot < dot_min:
                vector2_min = vector2
                dot_min = dot
        return cls.from_3_points(origin, vector1 + origin, vector2_min + origin)

    def angle_between_planes(self, plane2):
        """
        Get angle between 2 planes.

        :param plane2: the second plane.
        :return: the angle between the two planes.
        """
        angle = math.acos(self.frame.w.dot(plane2.frame.w))
        return angle

    def point_on_surface(self, point):
        """
        Return if the point belongs to the plane at a tolerance of 1e-6.

        """
        if math.isclose(self.frame.w.dot(point - self.frame.origin), 0,
                        abs_tol=1e-6):
            return True
        return False

    def point_distance(self, point3d):
        """
        Calculates the distance of a point to plane.

        :param point3d: point to verify distance.
        :return: a float, point distance to plane.
        """
        coefficient_a, coefficient_b, coefficient_c, coefficient_d = self.equation_coefficients()
        return abs(self.frame.w.dot(point3d) + coefficient_d) / math.sqrt(coefficient_a ** 2 +
                                                                          coefficient_b ** 2 + coefficient_c ** 2)

    def line_intersections(self, line):
        """
        Find the intersection with a line.

        :param line: Line to evaluate the intersection
        :type line: :class:`edges.Line`
        :return: ADD DESCRIPTION
        :rtype: List[volmdlr.Point3D]
        """
        u_vector = line.point2 - line.point1
        w_vector = line.point1 - self.frame.origin
        if math.isclose(self.frame.w.dot(u_vector), 0, abs_tol=1e-08):
            return []
        intersection_abscissea = - self.frame.w.dot(w_vector) / self.frame.w.dot(u_vector)
        return [line.point1 + intersection_abscissea * u_vector]

    def linesegment_intersections(self, linesegment: edges.LineSegment3D) \
            -> List[volmdlr.Point3D]:
        """
        Gets the intersections of a plane a line segment 3d.

        :param linesegment: other line segment.
        :return: a list with the intersecting point.
        """
        u_vector = linesegment.end - linesegment.start
        w_vector = linesegment.start - self.frame.origin
        normaldotu = self.frame.w.dot(u_vector)
        if math.isclose(normaldotu, 0, abs_tol=1e-08):
            return []
        intersection_abscissea = - self.frame.w.dot(w_vector) / normaldotu
        if intersection_abscissea < 0 or intersection_abscissea > 1:
            return []
        return [linesegment.start + intersection_abscissea * u_vector]

    def fullarc_intersections(self, fullarc: edges.FullArc3D):
        """
        Calculates the intersections between a Plane 3D and a FullArc 3D.

        :param fullarc: full arc to verify intersections.
        :return: list of intersections: List[volmdlr.Point3D].
        """
        fullarc_plane = Plane3D(fullarc.frame)
        plane_intersections = self.plane_intersection(fullarc_plane)
        if not plane_intersections:
            return []
        fullarc2d = fullarc.to_2d(fullarc.center, fullarc_plane.frame.u, fullarc_plane.frame.v)
        line2d = plane_intersections[0].to_2d(fullarc.center, fullarc_plane.frame.u, fullarc_plane.frame.v)
        fullarc2d_inters_line2d = fullarc2d.line_intersections(line2d)
        intersections = []
        for inter in fullarc2d_inters_line2d:
            intersections.append(inter.to_3d(fullarc.center, fullarc_plane.frame.u, fullarc_plane.frame.v))
        return intersections

    def equation_coefficients(self):
        """
        Returns the a,b,c,d coefficient from equation ax+by+cz+d = 0.

        """
        a, b, c = self.frame.w
        d = -self.frame.origin.dot(self.frame.w)
        return round(a, 12), round(b, 12), round(c, 12), round(d, 12)

    def plane_intersection(self, other_plane):
        """
        Computes intersection points between two Planes 3D.

        """
        if self.is_parallel(other_plane):
            return []
        line_direction = self.frame.w.cross(other_plane.frame.w)

        if line_direction.norm() < 1e-6:
            return None

        a1, b1, c1, d1 = self.equation_coefficients()
        a2, b2, c2, d2 = other_plane.equation_coefficients()
        if not math.isclose(a1 * b2 - a2 * b1, 0.0, abs_tol=1e-10):
            x0 = (b1 * d2 - b2 * d1) / (a1 * b2 - a2 * b1)
            y0 = (a2 * d1 - a1 * d2) / (a1 * b2 - a2 * b1)
            point1 = volmdlr.Point3D(x0, y0, 0)
        elif a2 * c1 != a1 * c2:
            x0 = (c2 * d1 - c1 * d2) / (a2 * c1 - a1 * c2)
            z0 = (a1 * d2 - a2 * d1) / (a2 * c1 - a1 * c2)
            point1 = volmdlr.Point3D(x0, 0, z0)
        elif c1 * b2 != b1 * c2:
            y0 = (- c2 * d1 + c1 * d2) / (b1 * c2 - c1 * b2)
            z0 = (- b1 * d2 + b2 * d1) / (b1 * c2 - c1 * b2)
            point1 = volmdlr.Point3D(0, y0, z0)
        else:
            raise NotImplementedError
        return [edges.Line3D(point1, point1 + line_direction)]

    def is_coincident(self, plane2):
        """
        Verifies if two planes are parallel and coincident.

        """
        if not isinstance(self, plane2.__class__):
            return False
        if self.is_parallel(plane2):
            if plane2.point_on_surface(self.frame.origin):
                return True
        return False

    def is_parallel(self, plane2):
        """
        Verifies if two planes are parallel.

        """
        if self.frame.w.is_colinear_to(plane2.frame.w):
            return True
        return False

    @classmethod
    def plane_betweeen_two_planes(cls, plane1, plane2):
        """
        Calculates a plane between two other planes.

        :param plane1: plane1.
        :param plane2: plane2.
        :return: resulting plane.
        """
        plane1_plane2_intersection = plane1.plane_intersection(plane2)[0]
        u = plane1_plane2_intersection.unit_direction_vector()
        v = plane1.frame.w + plane2.frame.w
        v.normalize()
        w = u.cross(v)
        point = (plane1.frame.origin + plane2.frame.origin) / 2
        return cls(volmdlr.Frame3D(point, u, w, v))

    def rotation(self, center: volmdlr.Point3D, axis: volmdlr.Vector3D, angle: float):
        """
        Plane3D rotation.

        :param center: rotation center
        :param axis: rotation axis
        :param angle: angle rotation
        :return: a new rotated Plane3D
        """
        new_frame = self.frame.rotation(center=center, axis=axis, angle=angle)
        return Plane3D(new_frame)

    def rotation_inplace(self, center: volmdlr.Point3D, axis: volmdlr.Vector3D, angle: float):
        """
        Plane3D rotation. Object is updated in-place.

        :param center: rotation center
        :param axis: rotation axis
        :param angle: rotation angle
        """
        warnings.warn("'in-place' methods are deprecated. Use a not in-place method instead.", DeprecationWarning)

        self.frame.rotation_inplace(center=center, axis=axis, angle=angle)

    def translation(self, offset: volmdlr.Vector3D):
        """
        Plane3D translation.

        :param offset: translation vector
        :return: A new translated Plane3D
        """
        new_frame = self.frame.translation(offset)
        return Plane3D(new_frame)

    def translation_inplace(self, offset: volmdlr.Vector3D):
        """
        Plane3D translation. Object is updated in-place.

        :param offset: translation vector
        """
        warnings.warn("'in-place' methods are deprecated. Use a not in-place method instead.", DeprecationWarning)

        self.frame.translation_inplace(offset)

    def frame_mapping(self, frame: volmdlr.Frame3D, side: str):
        """
        Changes frame_mapping and return a new Frame3D.

        :param frame: Frame of reference
        :type frame: `volmdlr.Frame3D`
        :param side: 'old' or 'new'
        """
        new_frame = self.frame.frame_mapping(frame, side)
        return Plane3D(new_frame, self.name)

    def frame_mapping_inplace(self, frame: volmdlr.Frame3D, side: str):
        """
        Changes frame_mapping and the object is updated in-place.

        :param frame: Frame of reference
        :type frame: `volmdlr.Frame3D`
        :param side: 'old' or 'new'
        """
        warnings.warn("'in-place' methods are deprecated. Use a not in-place method instead.", DeprecationWarning)

        new_frame = self.frame.frame_mapping(frame, side)
        self.frame.origin = new_frame.origin
        self.frame.u = new_frame.u
        self.frame.v = new_frame.v
        self.frame.w = new_frame.w

    def copy(self, deep=True, memo=None):
        """Creates a copy of the plane."""
        new_frame = self.frame.copy(deep, memo)
        return Plane3D(new_frame, self.name)

    def plot(self, ax=None, edge_style: EdgeStyle = EdgeStyle(color='grey'), length: float = 1.):
        """
        Plot the cylindrical surface in the local frame normal direction.

        :param ax: Matplotlib Axes3D object to plot on. If None, create a new figure.
        :type ax: Axes3D or None
        :param edge_style: edge styles.
        :type edge_style; EdgeStyle.
        :param length: plotted length
        :type length: float
        :return: Matplotlib Axes3D object containing the plotted wire-frame.
        :rtype: Axes3D
        """
        grid_size = 10

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.axis('equal')

        self.frame.plot(ax=ax, color=edge_style.color)
        for i in range(grid_size):
            for v1, v2 in [(self.frame.u, self.frame.v), (self.frame.v, self.frame.u)]:
                start = self.frame.origin - 0.5 * length * v1 + (-0.5 + i / (grid_size - 1)) * length * v2
                end = self.frame.origin + 0.5 * length * v1 + (-0.5 + i / (grid_size - 1)) * length * v2
                edges.LineSegment3D(start, end).plot(ax=ax, edge_style=edge_style)
        return ax

    def point2d_to_3d(self, point2d):
        """
        Converts a 2D parametric point into a 3D point on the surface.
        """
        return point2d.to_3d(self.frame.origin, self.frame.u, self.frame.v)

    def point3d_to_2d(self, point3d):
        """
        Converts a 3D point into a 2D parametric point.
        """
        return point3d.to_2d(self.frame.origin, self.frame.u, self.frame.v)

    def contour2d_to_3d(self, contour2d):
        """
        Converts a contour 2D on parametric surface into a 3D contour.
        """
        return contour2d.to_3d(self.frame.origin, self.frame.u, self.frame.v)

    def contour3d_to_2d(self, contour3d):
        """
        Converts a contour 3D into a 2D parametric contour.
        """
        return contour3d.to_2d(self.frame.origin, self.frame.u, self.frame.v)

    def bsplinecurve3d_to_2d(self, bspline_curve3d):
        control_points = [self.point3d_to_2d(p)
                          for p in bspline_curve3d.control_points]
        return [edges.BSplineCurve2D(
            bspline_curve3d.degree,
            control_points=control_points,
            knot_multiplicities=bspline_curve3d.knot_multiplicities,
            knots=bspline_curve3d.knots,
            weights=bspline_curve3d.weights,
            periodic=bspline_curve3d.periodic)]

    def bsplinecurve2d_to_3d(self, bspline_curve2d):
        """
        Converts a 2D B-Spline in parametric domain into a 3D B-Spline in spatial domain.

        :param bspline_curve2d: The B-Spline curve to perform the transformation.
        :type bspline_curve2d: edges.BSplineCurve2D
        :return: A 3D B-Spline.
        :rtype: edges.BSplineCurve3D
        """
        control_points = [self.point2d_to_3d(point)
                          for point in bspline_curve2d.control_points]
        return [edges.BSplineCurve3D(
            bspline_curve2d.degree,
            control_points=control_points,
            knot_multiplicities=bspline_curve2d.knot_multiplicities,
            knots=bspline_curve2d.knots,
            weights=bspline_curve2d.weights,
            periodic=bspline_curve2d.periodic)]

    def rectangular_cut(self, x1: float, x2: float,
                        y1: float, y2: float, name: str = ''):
        """Deprecated method, Use PlaneFace3D from_surface_rectangular_cut method."""

        raise AttributeError('Use PlaneFace3D from_surface_rectangular_cut method')


PLANE3D_OXY = Plane3D(volmdlr.OXYZ)
PLANE3D_OYZ = Plane3D(volmdlr.OYZX)
PLANE3D_OZX = Plane3D(volmdlr.OZXY)


class PeriodicalSurface(Surface3D):
    """
    Abstract class for surfaces with two-pi periodicity that creates some problems.
    """

    def point2d_to_3d(self, point2d):
        raise NotImplementedError(f'point2d_to_3d is abstract and should be implemented in {self.__class__.__name__}')

    def point3d_to_2d(self, point3d):
        """
        Abstract method. Convert a 3D point to a 2D parametric point.

        :param point3d: The 3D point to convert, represented by 3 coordinates (x, y, z).
        :type point3d: `volmdlr.Point3D`
        :return: NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError(f'point3d_to_2d is abstract and should be implemented in {self.__class__.__name__}')

    def repair_contours2d(self, outer_contour, inner_contours):
        """
        Repair contours on parametric domain.

        :param outer_contour: Outer contour 2D.
        :type inner_contours: wires.Contour2D
        :param inner_contours: List of 2D contours.
        :type inner_contours: list
        """
        new_inner_contours = []
        point1 = outer_contour.primitives[0].start
        point2 = outer_contour.primitives[-1].end

        theta1, z1 = point1
        theta2, _ = point2
        old_outer_contour_positioned = outer_contour
        new_outer_contour = old_outer_contour_positioned
        for inner_contour in inner_contours:
            theta3, z3 = inner_contour.primitives[0].start
            theta4, _ = inner_contour.primitives[-1].end

            if not inner_contour.is_ordered():

                outer_contour_theta = [theta1, theta2]
                inner_contour_theta = [theta3, theta4]

                # Contours are aligned
                if (math.isclose(theta1, theta3, abs_tol=1e-3) and math.isclose(theta2, theta4, abs_tol=1e-3)) \
                        or (math.isclose(theta1, theta4, abs_tol=1e-3) and math.isclose(theta2, theta3, abs_tol=1e-3)):
                    old_innner_contour_positioned = inner_contour

                else:
                    overlapping_theta, outer_contour_side, inner_contour_side, side = self._get_overlapping_theta(
                        outer_contour_theta,
                        inner_contour_theta)
                    line = edges.Line2D(volmdlr.Point2D(overlapping_theta, z1),
                                      volmdlr.Point2D(overlapping_theta, z3))
                    cutted_contours = inner_contour.split_by_line(line)
                    number_contours = len(cutted_contours)
                    if number_contours == 2:
                        contour1, contour2 = cutted_contours
                        increasing_theta = theta3 < theta4
                        # side = 0 --> left  side = 1 --> right
                        if (not side and increasing_theta) or (
                                side and not increasing_theta):
                            theta_offset = outer_contour_theta[outer_contour_side] - contour2.primitives[0].start.x
                            translation_vector = volmdlr.Vector2D(theta_offset, 0)
                            contour2_positionned = contour2.translation(offset=translation_vector)
                            theta_offset = contour2_positionned.primitives[-1].end.x - contour1.primitives[0].start.x
                            translation_vector = volmdlr.Vector2D(theta_offset, 0)
                            contour1_positionned = contour1.translation(offset=translation_vector)
                            primitives2d = contour2_positionned.primitives
                            primitives2d.extend(contour1_positionned.primitives)
                            old_innner_contour_positioned = wires.Wire2D(primitives2d)
                        else:
                            theta_offset = outer_contour_theta[outer_contour_side] - contour1.primitives[-1].end.x
                            translation_vector = volmdlr.Vector2D(theta_offset, 0)
                            contour1_positionned = contour1.translation(offset=translation_vector)
                            theta_offset = contour1_positionned.primitives[0].start.x - contour2.primitives[-1].end.x
                            translation_vector = volmdlr.Vector2D(theta_offset, 0)
                            contour2_positionned = contour2.translation(offset=translation_vector)
                            primitives2d = contour1_positionned.primitives
                            primitives2d.extend(contour2_positionned.primitives)
                            old_innner_contour_positioned = wires.Wire2D(primitives2d)
                        old_innner_contour_positioned = old_innner_contour_positioned.order_wire(tol=1e-4)
                    elif number_contours == 1:
                        contour = cutted_contours[0]
                        theta_offset = outer_contour_theta[outer_contour_side] -\
                                       inner_contour_theta[inner_contour_side]
                        translation_vector = volmdlr.Vector2D(theta_offset, 0)
                        old_innner_contour_positioned = contour.translation(offset=translation_vector)

                    else:
                        raise NotImplementedError
                point1 = old_outer_contour_positioned.primitives[0].start
                point2 = old_outer_contour_positioned.primitives[-1].end
                point3 = old_innner_contour_positioned.primitives[0].start
                point4 = old_innner_contour_positioned.primitives[-1].end

                outer_contour_direction = point1.x < point2.x
                inner_contour_direction = point3.x < point4.x
                if outer_contour_direction == inner_contour_direction:
                    old_innner_contour_positioned = old_innner_contour_positioned.invert()
                    point3 = old_innner_contour_positioned.primitives[0].start
                    point4 = old_innner_contour_positioned.primitives[-1].end

                closing_linesegment1 = edges.LineSegment2D(point2, point3)
                closing_linesegment2 = edges.LineSegment2D(point4, point1)
                new_outer_contour_primitives = old_outer_contour_positioned.primitives + [closing_linesegment1] + \
                                               old_innner_contour_positioned.primitives + \
                                               [closing_linesegment2]
                new_outer_contour = wires.Contour2D(primitives=new_outer_contour_primitives)
                new_outer_contour.order_contour(tol=1e-4)
            else:
                new_inner_contours.append(inner_contour)
        return new_outer_contour, new_inner_contours

    def _get_overlapping_theta(self, outer_contour_startend_theta, inner_contour_startend_theta):
        """
        Find overlapping theta domain between two contours on periodical Surfaces.
        """
        oc_xmin_index, outer_contour_xmin = min(enumerate(outer_contour_startend_theta), key=lambda x: x[1])
        oc_xmax_index, outer_contour_xman = max(enumerate(outer_contour_startend_theta), key=lambda x: x[1])
        ic_xmin_index, inner_contour_xmin = min(enumerate(inner_contour_startend_theta), key=lambda x: x[1])
        ic_xmax_index, inner_contour_xmax = max(enumerate(inner_contour_startend_theta), key=lambda x: x[1])

        # check if tetha3 or theta4 is in [theta1, theta2] interval
        overlap = outer_contour_xmin <= inner_contour_xmax and outer_contour_xman >= inner_contour_xmin

        if overlap:
            if inner_contour_xmin < outer_contour_xmin:
                overlapping_theta = outer_contour_startend_theta[oc_xmin_index]
                side = 0
                return overlapping_theta, oc_xmin_index, ic_xmin_index, side
            overlapping_theta = outer_contour_startend_theta[oc_xmax_index]
            side = 1
            return overlapping_theta, oc_xmax_index, ic_xmax_index, side

        # if not direct intersection -> find intersection at periodicity
        if inner_contour_xmin < outer_contour_xmin:
            overlapping_theta = outer_contour_startend_theta[oc_xmin_index] - 2 * math.pi
            side = 0
            return overlapping_theta, oc_xmin_index, ic_xmin_index, side
        overlapping_theta = outer_contour_startend_theta[oc_xmax_index] + 2 * math.pi
        side = 1
        return overlapping_theta, oc_xmax_index, ic_xmax_index, side

    def _reference_points(self, edge):
        """
        Helper function to return points of reference on the edge to fix some parametric periodical discontinuities.
        """
        length = edge.length()
        point_after_start = self.point3d_to_2d(edge.point_at_abscissa(0.001 * length))
        point_before_end = self.point3d_to_2d(edge.point_at_abscissa(0.98 * length))
        theta3, _ = point_after_start
        theta4, _ = point_before_end
        if abs(theta3) == math.pi or abs(theta3) == 0.5 * math.pi:
            point_after_start = self.point3d_to_2d(edge.point_at_abscissa(0.002 * length))
        if abs(theta4) == math.pi or abs(theta4) == 0.5 * math.pi:
            point_before_end = self.point3d_to_2d(edge.point_at_abscissa(0.97 * length))
        return point_after_start, point_before_end

    def _verify_start_end_angles(self, edge, theta1, theta2):
        """
        Verify if there is some incoherence with start and end angles. If so, return fixed angles.
        """
        length = edge.length()
        theta3, _ = self.point3d_to_2d(edge.point_at_abscissa(0.001 * length))
        # make sure that the reference angle is not undefined
        if abs(theta3) == math.pi:
            theta3, _ = self.point3d_to_2d(edge.point_at_abscissa(0.002 * length))

        # Verify if theta1 or theta2 point should be -pi because atan2() -> ]-pi, pi]
        # And also atan2 discontinuity in 0.5 * math.pi
        if abs(theta1) == math.pi or abs(theta1) == 0.5 * math.pi:
            theta1 = repair_start_end_angle_periodicity(theta1, theta3)
        if abs(theta2) == math.pi or abs(theta2) == 0.5 * math.pi:
            theta4, _ = self.point3d_to_2d(edge.point_at_abscissa(0.98 * length))
            # make sure that the reference angle is not undefined
            if abs(theta4) == math.pi:
                theta4, _ = self.point3d_to_2d(edge.point_at_abscissa(0.97 * length))
            theta2 = repair_start_end_angle_periodicity(theta2, theta4)

        return theta1, theta2

    def _fix_angle_discontinuity_on_discretization_points(self, points, indexes_angle_discontinuity, direction):
        i = 0 if direction == "x" else 1
        if len(indexes_angle_discontinuity) == 1:
            index_angle_discontinuity = indexes_angle_discontinuity[0]
            sign = round(points[index_angle_discontinuity - 1][i] / abs(points[index_angle_discontinuity - 1][i]), 2)
            if i == 0:
                points = [p + volmdlr.Point2D(sign * volmdlr.TWO_PI, 0) if i >= index_angle_discontinuity else p
                          for i, p in enumerate(points)]
            else:
                points = [p + volmdlr.Point2D(0, sign * volmdlr.TWO_PI) if i >= index_angle_discontinuity else p
                          for i, p in enumerate(points)]
        else:
            raise NotImplementedError
        return points

    def linesegment3d_to_2d(self, linesegment3d):
        """
        Converts the primitive from 3D spatial coordinates to its equivalent 2D primitive in the parametric space.
        """
        start = self.point3d_to_2d(linesegment3d.start)
        end = self.point3d_to_2d(linesegment3d.end)
        if start.x != end.x:
            end = volmdlr.Point2D(start.x, end.y)
        if not start.is_close(end):
            return [edges.LineSegment2D(start, end, name="parametric.linesegment")]
        return None

    def arc3d_to_2d(self, arc3d):
        """
        Converts the primitive from 3D spatial coordinates to its equivalent 2D primitive in the parametric space.
        """
        start = self.point3d_to_2d(arc3d.start)
        end = self.point3d_to_2d(arc3d.end)
        angle3d = arc3d.angle
        point_after_start, point_before_end = self._reference_points(arc3d)

        start, end = vm_parametric.arc3d_to_cylindrical_coordinates_verification(start, end, angle3d,
                                                                                 point_after_start.x,
                                                                                 point_before_end.x)
        return [edges.LineSegment2D(start, end, name="arc")]

    def fullarc3d_to_2d(self, fullarc3d):
        """
        Converts the primitive from 3D spatial coordinates to its equivalent 2D primitive in the parametric space.
        """
        start = self.point3d_to_2d(fullarc3d.start)
        end = self.point3d_to_2d(fullarc3d.end)

        point_after_start, point_before_end = self._reference_points(fullarc3d)

        start, end = vm_parametric.arc3d_to_cylindrical_coordinates_verification(start, end, volmdlr.TWO_PI,
                                                                                 point_after_start.x,
                                                                                 point_before_end.x)
        theta1, z1 = start
        # _, z2 = end
        theta3, z3 = point_after_start

        if self.frame.w.is_colinear_to(fullarc3d.normal):
            if start.is_close(end):
                start, end = vm_parametric.fullarc_to_cylindrical_coordinates_verification(start, end, theta3)
            return [edges.LineSegment2D(start, end, name="parametric.fullarc")]
        # Treating one case from Revolution Surface
        if z1 > z3:
            point1 = volmdlr.Point2D(theta1, 1)
            point2 = volmdlr.Point2D(theta1, 0)
        else:
            point1 = volmdlr.Point2D(theta1, 0)
            point2 = volmdlr.Point2D(theta1, 1)
        return [edges.LineSegment2D(point1, point2, name="parametric.fullarc")]

    def bsplinecurve3d_to_2d(self, bspline_curve3d):
        """
        Converts the primitive from 3D spatial coordinates to its equivalent 2D primitive in the parametric space.
        """
        n = len(bspline_curve3d.control_points)
        points3d = bspline_curve3d.discretization_points(number_points=n)
        points = [self.point3d_to_2d(point) for point in points3d]
        theta1, z1 = points[0]
        theta2, z2 = points[-1]
        theta1, theta2 = self._verify_start_end_angles(bspline_curve3d, theta1, theta2)
        points[0] = volmdlr.Point2D(theta1, z1)
        points[-1] = volmdlr.Point2D(theta2, z2)

        theta_list = [point.x for point in points]
        theta_discontinuity, indexes_theta_discontinuity = angle_discontinuity(theta_list)
        if theta_discontinuity:
            points = self._fix_angle_discontinuity_on_discretization_points(points,
                                                                            indexes_theta_discontinuity, "x")

        return [edges.BSplineCurve2D.from_points_interpolation(points, degree=bspline_curve3d.degree,
                                                             periodic=bspline_curve3d.periodic)]

    def arcellipse3d_to_2d(self, arcellipse3d):
        """
        Transformation of a 3D arc of ellipse to a 2D primitive in a cylindrical surface.

        """
        points = [self.point3d_to_2d(p)
                  for p in arcellipse3d.discretization_points(number_points=50)]

        theta1, z1 = points[0]
        theta2, z2 = points[-1]

        # theta3, _ = self.point3d_to_2d(arcellipse3d.point_at_abscissa(0.001 * length))
        theta3, _ = points[1]
        # make sure that the reference angle is not undefined
        if abs(theta3) == math.pi:
            theta3, _ = points[1]

        # Verify if theta1 or theta2 point should be -pi because atan2() -> ]-pi, pi]
        if abs(theta1) == math.pi:
            theta1 = vm_parametric.repair_start_end_angle_periodicity(theta1, theta3)
        if abs(theta2) == math.pi:
            theta4, _ = points[-2]
            # make sure that the reference angle is not undefined
            if abs(theta4) == math.pi:
                theta4, _ = points[-3]
            theta2 = vm_parametric.repair_start_end_angle_periodicity(theta2, theta4)

        points[0] = volmdlr.Point2D(theta1, z1)
        points[-1] = volmdlr.Point2D(theta2, z2)

        theta_list = [point.x for point in points]
        theta_discontinuity, indexes_theta_discontinuity = angle_discontinuity(theta_list)
        if theta_discontinuity:
            points = self._fix_angle_discontinuity_on_discretization_points(points,
                                                                            indexes_theta_discontinuity, "x")

        return [edges.BSplineCurve2D.from_points_interpolation(points, degree=2, name="parametric.arcellipse")]

    def fullarcellipse3d_to_2d(self, arcellipse3d):
        """
        Transformation of a 3D arc ellipse to 2D, in a cylindrical surface.

        """
        points = [self.point3d_to_2d(p)
                  for p in arcellipse3d.discretization_points(number_points=100)]
        start, end = points[0], points[-1]
        if start.is_close(end, 1e-4):
            start, end = vm_parametric.fullarc_to_cylindrical_coordinates_verification(start, end, points[2])
        theta1, z1 = start
        theta2, z2 = end
        theta1, theta2 = self._verify_start_end_angles(arcellipse3d, theta1, theta2)
        points[0] = volmdlr.Point2D(theta1, z1)
        points[-1] = volmdlr.Point2D(theta2, z2)

        theta_list = [point.x for point in points]
        theta_discontinuity, indexes_theta_discontinuity = angle_discontinuity(theta_list)
        if theta_discontinuity:
            points = self._fix_angle_discontinuity_on_discretization_points(points,
                                                                            indexes_theta_discontinuity, "x")

        return [edges.BSplineCurve2D.from_points_interpolation(points, degree=2, periodic=True,
                                                               name="parametric.fullarcellipse")]

    def bsplinecurve2d_to_3d(self, bspline_curve2d):
        """
        Is this right?.
        """
        if bspline_curve2d.name in ("parametric.arcellipse", "parametric.fullarcellipse"):
            start = self.point2d_to_3d(bspline_curve2d.start)
            middle_point = self.point2d_to_3d(bspline_curve2d.point_at_abscissa(0.5 * bspline_curve2d.length()))
            extra_point = self.point2d_to_3d(bspline_curve2d.point_at_abscissa(0.75 * bspline_curve2d.length()))
            if bspline_curve2d.name == "parametric.arcellipse":
                end = self.point2d_to_3d(bspline_curve2d.end)
                plane3d = Plane3D.from_3_points(start, middle_point, end)
                ellipse = self.concurrent_plane_intersection(plane3d)[0]
                return [edges.ArcEllipse3D(start, middle_point, end, ellipse.center, ellipse.major_dir, ellipse.normal,
                                         extra_point)]
            plane3d = Plane3D.from_3_points(start, middle_point, extra_point)
            ellipse = self.concurrent_plane_intersection(plane3d)[0]
            return [edges.FullArcEllipse3D(start, ellipse.major_axis, ellipse.minor_axis,
                                            ellipse.center, ellipse.normal, ellipse.major_dir)]
        n = len(bspline_curve2d.control_points)
        points = [self.point2d_to_3d(p)
                  for p in bspline_curve2d.discretization_points(number_points=n)]
        return [edges.BSplineCurve3D.from_points_interpolation(points, bspline_curve2d.degree,
                                                               bspline_curve2d.periodic)]

    def linesegment2d_to_3d(self, linesegment2d):
        """
        Converts a BREP line segment 2D onto a 3D primitive on the surface.
        """
        theta1, z1 = linesegment2d.start
        theta2, z2 = linesegment2d.end
        start3d = self.point2d_to_3d(linesegment2d.start)
        end3d = self.point2d_to_3d(linesegment2d.end)
        if math.isclose(theta1, theta2, abs_tol=1e-4) or linesegment2d.name == "parametic.linesegment":
            if start3d.is_close(end3d):
                return None
            return [edges.LineSegment3D(start3d, end3d)]

        if math.isclose(z1, z2, abs_tol=1e-4) or linesegment2d.name == "parametric.arc" or \
                linesegment2d.name == "parametric.fullarc":
            if math.isclose(abs(theta1 - theta2), volmdlr.TWO_PI, abs_tol=1e-4):
                return [edges.FullArc3D(center=self.frame.origin + z1 * self.frame.w,
                                        start_end=self.point2d_to_3d(linesegment2d.start),
                                        normal=self.frame.w)]

            return [edges.Arc3D(
                self.point2d_to_3d(linesegment2d.start),
                self.point2d_to_3d(volmdlr.Point2D(0.5 * (theta1 + theta2), z1)),
                self.point2d_to_3d(linesegment2d.end)
            )]
        if start3d.is_close(end3d):
            return None
        raise NotImplementedError("This case is not yet treated")


class CylindricalSurface3D(PeriodicalSurface):
    """
    The local plane is defined by (theta, z).

    :param frame: frame.w is axis, frame.u is theta=0 frame.v theta=pi/2
    :param frame:
    :param radius: Cylinder's radius
    :type radius: float
    """
    face_class = 'CylindricalFace3D'
    x_periodicity = volmdlr.TWO_PI
    y_periodicity = None

    def __init__(self, frame, radius: float, name: str = ''):
        self.frame = frame
        self.radius = radius
        PeriodicalSurface.__init__(self, name=name)

    def plot(self, ax=None, edge_style: EdgeStyle = EdgeStyle(color='grey', alpha=0.5), length: float = 1.):
        """
        Plot the cylindrical surface in the local frame normal direction.

        :param ax: Matplotlib Axes3D object to plot on. If None, create a new figure.
        :type ax: Axes3D or None
        :param edge_style: edge styles.
        :type edge_style; EdgeStyle.
        :param length: plotted length
        :type length: float
        :return: Matplotlib Axes3D object containing the plotted wire-frame.
        :rtype: Axes3D
        """
        ncircles = 10
        nlines = 30

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        self.frame.plot(ax=ax, ratio=0.5 * length)
        for i in range(nlines):
            theta = i / (nlines - 1) * volmdlr.TWO_PI
            start = self.point2d_to_3d(volmdlr.Point2D(theta, -0.5 * length))
            end = self.point2d_to_3d(volmdlr.Point2D(theta, 0.5 * length))
            edges.LineSegment3D(start, end).plot(ax=ax, edge_style=edge_style)

        for j in range(ncircles):
            circle_frame = self.frame.copy()
            circle_frame.origin += (-0.5 + j / (ncircles - 1)) * length * circle_frame.w
            circle = wires.Circle3D(circle_frame, self.radius)
            circle.plot(ax=ax, edge_style=edge_style)
        return ax

    def point2d_to_3d(self, point2d: volmdlr.Point2D):
        """
        Coverts a parametric coordinate on the surface into a 3D spatial point (x, y, z).

        :param point2d: Point at the ToroidalSuface3D
        :type point2d: `volmdlr.`Point2D`
        """

        point = volmdlr.Point3D(self.radius * math.cos(point2d.x),
                                self.radius * math.sin(point2d.x),
                                point2d.y)
        return self.frame.local_to_global_coordinates(point)

    def point3d_to_2d(self, point3d):
        """
        Returns the cylindrical coordinates volmdlr.Point2D(theta, z) of a Cartesian coordinates point (x, y, z).

        :param point3d: Point at the CylindricalSuface3D
        :type point3d: `volmdlr.`Point3D`
        """
        x, y, z = self.frame.global_to_local_coordinates(point3d)
        # Do not delete this, mathematical problem when x and y close to zero but not 0
        if abs(x) < 1e-12:
            x = 0
        if abs(y) < 1e-12:
            y = 0

        theta = math.atan2(y, x)
        if abs(theta) < 1e-9:
            theta = 0.0

        return volmdlr.Point2D(theta, z)

    @classmethod
    def from_step(cls, arguments, object_dict, **kwargs):
        """
        Converts a step primitive to a CylindricalSurface3D.

        :param arguments: The arguments of the step primitive.
        :type arguments: list
        :param object_dict: The dictionary containing all the step primitives
            that have already been instantiated
        :type object_dict: dict
        :return: The corresponding CylindricalSurface3D object.
        :rtype: :class:`volmdlr.faces.CylindricalSurface3D`
        """

        length_conversion_factor = kwargs.get("length_conversion_factor", 1)
        frame3d = object_dict[arguments[1]]
        u_vector, w_vector = frame3d.v, -frame3d.u
        u_vector.normalize()
        w_vector.normalize()
        v_vector = w_vector.cross(u_vector)
        frame_direct = volmdlr.Frame3D(frame3d.origin, u_vector, v_vector, w_vector)
        radius = float(arguments[2]) * length_conversion_factor
        return cls(frame_direct, radius, arguments[0][1:-1])

    def to_step(self, current_id):
        """
        Translate volmdlr primitive to step syntax.
        """
        frame = volmdlr.Frame3D(self.frame.origin, self.frame.w, self.frame.u,
                                self.frame.v)
        content, frame_id = frame.to_step(current_id)
        current_id = frame_id + 1
        content += f"#{current_id} = CYLINDRICAL_SURFACE('{self.name}',#{frame_id},{round(1000 * self.radius, 3)});\n"
        return content, [current_id]

    def frame_mapping(self, frame: volmdlr.Frame3D, side: str):
        """
        Changes frame_mapping and return a new CylindricalSurface3D.

        :param side: 'old' or 'new'
        """
        new_frame = self.frame.frame_mapping(frame, side)
        return CylindricalSurface3D(new_frame, self.radius,
                                    name=self.name)

    def frame_mapping_inplace(self, frame: volmdlr.Frame3D, side: str):
        """
        Changes frame_mapping and the object is updated in-place.

        :param side: 'old' or 'new'
        """
        warnings.warn("'in-place' methods are deprecated. Use a not in-place method instead.", DeprecationWarning)

        new_frame = self.frame.frame_mapping(frame, side)
        self.frame = new_frame

    def rectangular_cut(self, theta1: float, theta2: float,
                        z1: float, z2: float, name: str = ''):
        """Deprecated method, Use CylindricalFace3D from_surface_rectangular_cut method."""
        raise AttributeError('Use CylindricalFace3D from_surface_rectangular_cut method')

    def rotation(self, center: volmdlr.Point3D, axis: volmdlr.Vector3D, angle: float):
        """
        CylindricalFace3D rotation.

        :param center: rotation center.
        :param axis: rotation axis.
        :param angle: angle rotation.
        :return: a new rotated Plane3D.
        """
        new_frame = self.frame.rotation(center=center, axis=axis,
                                        angle=angle)
        return CylindricalSurface3D(new_frame, self.radius)

    def rotation_inplace(self, center: volmdlr.Point3D, axis: volmdlr.Vector3D, angle: float):
        """
        CylindricalFace3D rotation. Object is updated in-place.

        :param center: rotation center.
        :param axis: rotation axis.
        :param angle: rotation angle.
        """
        warnings.warn("'in-place' methods are deprecated. Use a not in-place method instead.", DeprecationWarning)

        self.frame.rotation_inplace(center, axis, angle)

    def translation(self, offset: volmdlr.Vector3D):
        """
        CylindricalFace3D translation.

        :param offset: translation vector.
        :return: A new translated CylindricalFace3D.
        """
        return CylindricalSurface3D(self.frame.translation(offset), self.radius)

    def translation_inplace(self, offset: volmdlr.Vector3D):
        """
        CylindricalFace3D translation. Object is updated in-place.

        :param offset: translation vector
        """
        warnings.warn("'in-place' methods are deprecated. Use a not in-place method instead.", DeprecationWarning)

        self.frame.translation_inplace(offset)

    def grid3d(self, grid2d: grid.Grid2D):
        """
        Generate 3d grid points of a Cylindrical surface, based on a Grid2D.

        """

        points_2d = grid2d.points
        points_3d = [self.point2d_to_3d(point2d) for point2d in points_2d]

        return points_3d

    def line_intersections(self, line: edges.Line3D):
        line_2d = line.to_2d(self.frame.origin, self.frame.u, self.frame.v)
        if line_2d is None:
            return []
        origin2d = self.frame.origin.to_2d(self.frame.origin, self.frame.u, self.frame.v)
        distance_line2d_to_origin = line_2d.point_distance(origin2d)
        if distance_line2d_to_origin > self.radius:
            return []
        a_prime = line_2d.point1
        b_prime = line_2d.point2
        a_prime_minus_b_prime = a_prime - b_prime
        t_param = a_prime.dot(a_prime_minus_b_prime) / a_prime_minus_b_prime.dot(a_prime_minus_b_prime)
        k_param = math.sqrt(
            (self.radius ** 2 - distance_line2d_to_origin ** 2) / a_prime_minus_b_prime.dot(a_prime_minus_b_prime))
        intersection1 = line.point1 + (t_param + k_param) * (line.direction_vector())
        intersection2 = line.point1 + (t_param - k_param) * (line.direction_vector())
        if intersection1 == intersection2:
            return [intersection1]

        return [intersection1, intersection2]

    def linesegment_intersections(self, linesegment: edges.LineSegment3D):
        line = linesegment.to_line()
        line_intersections = self.line_intersections(line)
        linesegment_intersections = [inters for inters in line_intersections if linesegment.point_belongs(inters)]
        return linesegment_intersections

    def parallel_plane_intersection(self, plane3d):
        """
        Cylinder plane intersections when plane's normal is perpendicular with the cylinder axis.

        :param plane3d: intersecting plane
        :return: list of intersecting curves
        """
        distance_plane_cylinder_axis = plane3d.point_distance(self.frame.origin)
        if distance_plane_cylinder_axis > self.radius:
            return []
        if math.isclose(self.frame.w.dot(plane3d.frame.u), 0, abs_tol=1e-6):
            line = edges.Line3D(plane3d.frame.origin, plane3d.frame.origin + plane3d.frame.u)
        else:
            line = edges.Line3D(plane3d.frame.origin, plane3d.frame.origin + plane3d.frame.v)
        line_intersections = self.line_intersections(line)
        lines = []
        for intersection in line_intersections:
            lines.append(edges.Line3D(intersection, intersection + self.frame.w))
        return lines

    def perpendicular_plane_intersection(self, plane3d):
        """
        Cylinder plane intersections when plane's normal is parallel with the cylinder axis.

        :param plane3d: intersecting plane
        :return: list of intersecting curves
        """
        line = edges.Line3D(self.frame.origin, self.frame.origin + self.frame.w)
        center3d_plane = plane3d.line_intersections(line)[0]
        circle3d = wires.Circle3D(volmdlr.Frame3D(center3d_plane, plane3d.frame.u,
                                                          plane3d.frame.v, plane3d.frame.w), self.radius)
        return [circle3d]

    def concurrent_plane_intersection(self, plane3d):
        """
        Cylinder plane intersections when plane's normal is concurrent with the cylinder axis, but not orthogonal.

        Ellipse vector equation : < r*cos(t), r*sin(t), -(1 / c)*(d + a*r*cos(t) +
        b*r*sin(t)); d = - (ax_0 + by_0 + cz_0).

        :param plane3d: intersecting plane.
        :return: list of intersecting curves.
        """
        line = edges.Line3D(self.frame.origin, self.frame.origin + self.frame.w)
        center3d_plane = plane3d.line_intersections(line)[0]
        plane_coefficient_a, plane_coefficient_b, plane_coefficient_c, plane_coefficient_d = \
            plane3d.equation_coefficients()
        ellipse_0 = volmdlr.Point3D(
            self.radius * math.cos(0),
            self.radius * math.sin(0),
            - (1 / plane_coefficient_c) * (plane_coefficient_d + plane_coefficient_a * self.radius * math.cos(0) +
                                           plane_coefficient_b * self.radius * math.sin(0)))
        ellipse_pi_by_2 = volmdlr.Point3D(
            self.radius * math.cos(math.pi / 2),
            self.radius * math.sin(math.pi / 2),
            - (1 / plane_coefficient_c) * (
                    plane_coefficient_d + plane_coefficient_a * self.radius * math.cos(math.pi / 2)
                    + plane_coefficient_b * self.radius * math.sin(math.pi / 2)))
        axis_1 = center3d_plane.point_distance(ellipse_0)
        axis_2 = center3d_plane.point_distance(ellipse_pi_by_2)
        if axis_1 > axis_2:
            major_axis = axis_1
            minor_axis = axis_2
            major_dir = ellipse_0 - center3d_plane
        else:
            major_axis = axis_2
            minor_axis = axis_1
            major_dir = ellipse_pi_by_2 - center3d_plane
        return [wires.Ellipse3D(major_axis, minor_axis, center3d_plane, plane3d.frame.w, major_dir)]

    def plane_intersection(self, plane3d):
        """
        Cylinder intersections with a plane.

        :param plane3d: intersecting plane.
        :return: list of intersecting curves.
        """
        if math.isclose(abs(plane3d.frame.w.dot(self.frame.w)), 0, abs_tol=1e-6):
            return self.parallel_plane_intersection(plane3d)
        if math.isclose(abs(plane3d.frame.w.dot(self.frame.w)), 1, abs_tol=1e-6):
            return self.perpendicular_plane_intersection(plane3d)
        return self.concurrent_plane_intersection(plane3d)

    def is_coincident(self, surface3d):
        """
        Verifies if two CylindricalSurfaces are coincident.

        :param surface3d: surface to verify.
        :return: True if they are coincident, False otherwise.
        """
        if not isinstance(self, surface3d.__class__):
            return False
        if math.isclose(abs(self.frame.w.dot(surface3d.frame.w)), 1.0, abs_tol=1e-6) and \
                self.radius == surface3d.radius:
            return True
        return False

    def point_on_surface(self, point3d):
        """
        Verifies if a given point is on the CylindricalSurface3D.

        :param point3d: point to verify.
        :return: True if point on surface, False otherwise.
        """
        new_point = self.frame.global_to_local_coordinates(point3d)
        if math.isclose(new_point.x ** 2 + new_point.y ** 2, self.radius ** 2, abs_tol=1e-6):
            return True
        return False


class ToroidalSurface3D(PeriodicalSurface):
    """
    The local plane is defined by (theta, phi).

    Theta is the angle around the big (R) circle and phi around the small (r).

    :param frame: Tore's frame: origin is the center, u is pointing at theta=0.
    :param tore_radius: Tore's radius.
    :param r: Circle to revolute radius.

    See Also Definitions of R and r according to https://en.wikipedia.org/wiki/Torus.

    """
    face_class = 'ToroidalFace3D'
    x_periodicity = volmdlr.TWO_PI
    y_periodicity = volmdlr.TWO_PI

    def __init__(self, frame: volmdlr.Frame3D, tore_radius: float, small_radius: float, name: str = ''):
        self.frame = frame
        self.tore_radius = tore_radius
        self.small_radius = small_radius
        PeriodicalSurface.__init__(self, name=name)

        self._bbox = None

    @property
    def bounding_box(self):
        """
        Returns the surface bounding box.
        """
        if not self._bbox:
            self._bbox = self._bounding_box()
        return self._bbox

    def _bounding_box(self):
        distance = self.tore_radius + self.small_radius
        point1 = self.frame.origin + \
                 self.frame.u * distance + self.frame.v * distance + self.frame.w * self.small_radius
        point2 = self.frame.origin + \
                 self.frame.u * distance + self.frame.v * distance - self.frame.w * self.small_radius
        point3 = self.frame.origin + \
                 self.frame.u * distance - self.frame.v * distance + self.frame.w * self.small_radius
        point4 = self.frame.origin + \
                 self.frame.u * distance - self.frame.v * distance - self.frame.w * self.small_radius
        point5 = self.frame.origin - \
                 self.frame.u * distance + self.frame.v * distance + self.frame.w * self.small_radius
        point6 = self.frame.origin - \
                 self.frame.u * distance + self.frame.v * distance - self.frame.w * self.small_radius
        point7 = self.frame.origin - \
                 self.frame.u * distance - self.frame.v * distance + self.frame.w * self.small_radius
        point8 = self.frame.origin - \
                 self.frame.u * distance - self.frame.v * distance - self.frame.w * self.small_radius

        return volmdlr.core.BoundingBox.from_points(
            [point1, point2, point3, point4, point5, point6, point7, point8])

    def point2d_to_3d(self, point2d: volmdlr.Point2D):
        """
        Coverts a parametric coordinate on the surface into a 3D spatial point (x, y, z).

        :param point2d: Point at the ToroidalSuface3D
        :type point2d: `volmdlr.`Point2D`
        """
        theta, phi = point2d
        x = (self.tore_radius + self.small_radius * math.cos(phi)) * math.cos(theta)
        y = (self.tore_radius + self.small_radius * math.cos(phi)) * math.sin(theta)
        z = self.small_radius * math.sin(phi)
        return self.frame.local_to_global_coordinates(volmdlr.Point3D(x, y, z))

    def point3d_to_2d(self, point3d):
        """
        Transform a 3D spatial point (x, y, z) into a 2D spherical parametric point (theta, phi).
        """
        x, y, z = self.frame.global_to_local_coordinates(point3d)
        z = min(self.small_radius, max(-self.small_radius, z))

        # Do not delete this, mathematical problem when x and y close to zero (should be zero) but not 0
        # Generally this is related to uncertainty of step files.

        if abs(x) < 1e-12:
            x = 0
        if abs(y) < 1e-12:
            y = 0

        zr = z / self.small_radius
        phi = math.asin(zr)
        if abs(phi) < 1e-9:
            phi = 0

        u = self.tore_radius + math.sqrt((self.small_radius ** 2) - (z ** 2))
        u1, u2 = round(x / u, 5), round(y / u, 5)
        theta = math.atan2(u2, u1)

        vector_to_tube_center = volmdlr.Vector3D(self.tore_radius * math.cos(theta),
                                                 self.tore_radius * math.sin(theta), 0)
        vector_from_tube_center_to_point = volmdlr.Vector3D(x, y, z) - vector_to_tube_center
        phi2 = volmdlr.geometry.vectors3d_angle(vector_to_tube_center, vector_from_tube_center_to_point)

        if phi >= 0 and phi2 > 0.5 * math.pi:
            phi = math.pi - phi
        elif phi < 0 and phi2 > 0.5 * math.pi:
            phi = -math.pi - phi
        if abs(theta) < 1e-9:
            theta = 0.0
        if abs(phi) < 1e-9:
            phi = 0.0
        return volmdlr.Point2D(theta, phi)

    @classmethod
    def from_step(cls, arguments, object_dict, **kwargs):
        """
        Converts a step primitive to a ToroidalSurface3D.

        :param arguments: The arguments of the step primitive.
        :type arguments: list
        :param object_dict: The dictionary containing all the step primitives
            that have already been instantiated.
        :type object_dict: dict
        :return: The corresponding ToroidalSurface3D object.
        :rtype: :class:`volmdlr.faces.ToroidalSurface3D`
        """

        length_conversion_factor = kwargs.get("length_conversion_factor", 1)

        frame3d = object_dict[arguments[1]]
        u_vector, w_vector = frame3d.v, -frame3d.u
        u_vector.normalize()
        w_vector.normalize()
        v_vector = w_vector.cross(u_vector)
        frame_direct = volmdlr.Frame3D(frame3d.origin, u_vector, v_vector, w_vector)
        rcenter = float(arguments[2]) * length_conversion_factor
        rcircle = float(arguments[3]) * length_conversion_factor
        return cls(frame_direct, rcenter, rcircle, arguments[0][1:-1])

    def to_step(self, current_id):
        frame = volmdlr.Frame3D(self.frame.origin, self.frame.w, self.frame.u,
                                self.frame.v)
        content, frame_id = frame.to_step(current_id)
        current_id = frame_id + 1
        content += f"#{current_id} = TOROIDAL_SURFACE('{self.name}',#{frame_id}," \
                   f"{round(1000 * self.tore_radius, 3)},{round(1000 * self.small_radius, 3)});\n"
        return content, [current_id]

    def frame_mapping(self, frame: volmdlr.Frame3D, side: str):
        """
        Changes frame_mapping and return a new ToroidalSurface3D.

        :param frame: The new frame to map to.
        :type frame: `volmdlr.Frame3D
        :param side: Indicates whether the frame should be mapped to the 'old' or 'new' frame.
            Acceptable values are 'old' or 'new'.
        :type side: str
        """
        new_frame = self.frame.frame_mapping(frame, side)
        return ToroidalSurface3D(new_frame, self.tore_radius, self.small_radius, name=self.name)

    def frame_mapping_inplace(self, frame: volmdlr.Frame3D, side: str):
        """
        Changes frame_mapping and the object is updated in-place.

        :param frame: The new frame to map to.
        :type frame: `volmdlr.Frame3D
        :param side: Indicates whether the frame should be mapped to the 'old' or 'new' frame.
            Acceptable values are 'old' or 'new'.
        :type side: str
        """
        warnings.warn("'in-place' methods are deprecated. Use a not in-place method instead.", DeprecationWarning)

        new_frame = self.frame.frame_mapping(frame, side)
        self.frame = new_frame

    def rectangular_cut(self, theta1: float, theta2: float, phi1: float, phi2: float, name: str = ""):
        """Deprecated method, Use ToroidalFace3D from_surface_rectangular_cut method."""
        raise AttributeError('Use ToroidalFace3D from_surface_rectangular_cut method')

    def linesegment2d_to_3d(self, linesegment2d):
        """
        Converts the parametric boundary representation into a 3D primitive.
        """
        theta1, phi1 = linesegment2d.start
        theta2, phi2 = linesegment2d.end
        if math.isclose(theta1, theta2, abs_tol=1e-4):
            if math.isclose(abs(phi1 - phi2), volmdlr.TWO_PI, abs_tol=1e-4):
                u_vector = self.frame.u.rotation(self.frame.origin, self.frame.w, angle=theta1)
                v_vector = self.frame.u.rotation(self.frame.origin, self.frame.w, angle=theta1)
                center = self.frame.origin + self.tore_radius * u_vector
                return [edges.FullArc3D(center=center,
                                      start_end=center + self.small_radius * u_vector,
                                      normal=v_vector)]
            return [edges.Arc3D(
                self.point2d_to_3d(linesegment2d.start),
                self.point2d_to_3d(volmdlr.Point2D(theta1, 0.5 * (phi1 + phi2))),
                self.point2d_to_3d(linesegment2d.end),
            )]
        if math.isclose(phi1, phi2, abs_tol=1e-4):
            if math.isclose(abs(theta1 - theta2), volmdlr.TWO_PI, abs_tol=1e-4):
                center = self.frame.origin + self.small_radius * math.sin(phi1) * self.frame.w
                start_end = center + self.frame.u * (self.small_radius + self.tore_radius)
                return [edges.FullArc3D(center=center,
                                      start_end=start_end,
                                      normal=self.frame.w)]
            return [edges.Arc3D(
                self.point2d_to_3d(linesegment2d.start),
                self.point2d_to_3d(volmdlr.Point2D(0.5 * (theta1 + theta2), phi1)),
                self.point2d_to_3d(linesegment2d.end),
            )]
        raise NotImplementedError('Ellipse?')

    def bsplinecurve2d_to_3d(self, bspline_curve2d):
        """
        Converts the parametric boundary representation into a 3D primitive.
        """
        n = len(bspline_curve2d.control_points)
        points = [self.point2d_to_3d(p)
                  for p in bspline_curve2d.discretization_points(number_points=n)]
        return [edges.BSplineCurve3D.from_points_interpolation(
            points, bspline_curve2d.degree, bspline_curve2d.periodic)]

    def fullarc3d_to_2d(self, fullarc3d):
        """
        Converts the primitive from 3D spatial coordinates to its equivalent 2D primitive in the parametric space.
        """
        start = self.point3d_to_2d(fullarc3d.start)
        end = self.point3d_to_2d(fullarc3d.end)

        length = fullarc3d.length()
        angle3d = fullarc3d.angle
        point_after_start = self.point3d_to_2d(fullarc3d.point_at_abscissa(0.001 * length))
        point_before_end = self.point3d_to_2d(fullarc3d.point_at_abscissa(0.98 * length))

        start, end = vm_parametric.arc3d_to_spherical_coordinates_verification(start, end, angle3d,
                                                                               [point_after_start, point_before_end],
                                                                               [self.x_periodicity,
                                                                                self.y_periodicity])
        theta1, phi1 = start
        # theta2, phi2 = end
        theta3, phi3 = point_after_start
        # theta4, phi4 = point_before_end
        if self.frame.w.is_colinear_to(fullarc3d.normal, abs_tol=1e-4):
            point1 = start
            if theta1 > theta3:
                point2 = volmdlr.Point2D(theta1 - volmdlr.TWO_PI, phi1)
            elif theta1 < theta3:
                point2 = volmdlr.Point2D(theta1 + volmdlr.TWO_PI, phi1)
            return [edges.LineSegment2D(point1, point2)]
        point1 = start
        if phi1 > phi3:
            point2 = volmdlr.Point2D(theta1, phi1 - volmdlr.TWO_PI)
        elif phi1 < phi3:
            point2 = volmdlr.Point2D(theta1, phi1 + volmdlr.TWO_PI)
        return [edges.LineSegment2D(point1, point2)]

    def arc3d_to_2d(self, arc3d):
        start = self.point3d_to_2d(arc3d.start)
        end = self.point3d_to_2d(arc3d.end)

        length = arc3d.length()
        angle3d = arc3d.angle
        point_after_start = self.point3d_to_2d(arc3d.point_at_abscissa(0.001 * length))
        point_before_end = self.point3d_to_2d(arc3d.point_at_abscissa(0.98 * length))

        start, end = vm_parametric.arc3d_to_spherical_coordinates_verification(start, end, angle3d,
                                                                               [point_after_start, point_before_end],
                                                                               [self.x_periodicity,
                                                                                self.y_periodicity])

        return [edges.LineSegment2D(start, end)]

    def bsplinecurve3d_to_2d(self, bspline_curve3d):
        """
        Converts the primitive from 3D spatial coordinates to its equivalent 2D primitive in the parametric space.
        """
        length = bspline_curve3d.length()
        theta3, phi3 = self.point3d_to_2d(bspline_curve3d.point_at_abscissa(0.001 * length))
        theta4, phi4 = self.point3d_to_2d(bspline_curve3d.point_at_abscissa(0.98 * length))
        n = len(bspline_curve3d.control_points)
        points3d = bspline_curve3d.discretization_points(number_points=n)
        points = [self.point3d_to_2d(p) for p in points3d]
        theta1, phi1 = points[0]
        theta2, phi2 = points[-1]

        # Verify if theta1 or theta2 point should be -pi because atan2() -> ]-pi, pi]
        if abs(theta1) == math.pi:
            theta1 = repair_start_end_angle_periodicity(theta1, theta3)
        if abs(theta2) == math.pi:
            theta2 = repair_start_end_angle_periodicity(theta2, theta4)

        # Verify if phi1 or phi2 point should be -pi because phi -> ]-pi, pi]
        if abs(phi1) == math.pi:
            phi1 = repair_start_end_angle_periodicity(phi1, phi3)
        if abs(phi2) == math.pi:
            phi2 = repair_start_end_angle_periodicity(phi2, phi4)

        points[0] = volmdlr.Point2D(theta1, phi1)
        points[-1] = volmdlr.Point2D(theta2, phi2)

        theta_list = [point.x for point in points]
        phi_list = [point.y for point in points]
        theta_discontinuity, indexes_theta_discontinuity = angle_discontinuity(theta_list)
        phi_discontinuity, indexes_phi_discontinuity = angle_discontinuity(phi_list)

        if theta_discontinuity:
            points = self._fix_angle_discontinuity_on_discretization_points(points,
                                                                            indexes_theta_discontinuity, "x")
        if phi_discontinuity:
            points = self._fix_angle_discontinuity_on_discretization_points(points,
                                                                            indexes_phi_discontinuity, "y")

        return [edges.BSplineCurve2D.from_points_interpolation(
            points, bspline_curve3d.degree, bspline_curve3d.periodic)]

    def triangulation(self):
        """
        Triangulation.

        :rtype: display.DisplayMesh3D
        """
        face = self.rectangular_cut(0, volmdlr.TWO_PI, 0, volmdlr.TWO_PI)
        return face.triangulation()

    def translation(self, offset: volmdlr.Vector3D):
        """
        ToroidalSurface3D translation.

        :param offset: translation vector
        :return: A new translated ToroidalSurface3D
        """
        return ToroidalSurface3D(self.frame.translation(
            offset), self.tore_radius, self.small_radius)

    def translation_inplace(self, offset: volmdlr.Vector3D):
        """
        ToroidalSurface3D translation. Object is updated in-place.

        :param offset: translation vector.
        """
        warnings.warn("'in-place' methods are deprecated. Use a not in-place method instead.", DeprecationWarning)

        self.frame.translation_inplace(offset)

    def rotation(self, center: volmdlr.Point3D, axis: volmdlr.Vector3D, angle: float):
        """
        ToroidalSurface3D rotation.

        :param center: rotation center.
        :param axis: rotation axis.
        :param angle: angle rotation.
        :return: a new rotated ToroidalSurface3D.
        """
        new_frame = self.frame.rotation(center=center, axis=axis,
                                        angle=angle)
        return self.__class__(new_frame, self.tore_radius, self.small_radius)

    def rotation_inplace(self, center: volmdlr.Point3D, axis: volmdlr.Vector3D, angle: float):
        """
        ToroidalSurface3D rotation. Object is updated in-place.

        :param center: rotation center.
        :param axis: rotation axis.
        :param angle: rotation angle.
        """
        warnings.warn("'in-place' methods are deprecated. Use a not in-place method instead.", DeprecationWarning)

        self.frame.rotation_inplace(center, axis, angle)

    def plot(self, ax=None, color='grey', alpha=0.5):
        """Plot torus arcs."""
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        self.frame.plot(ax=ax)
        number_arcs = 50
        for i in range(number_arcs):
            theta = i / number_arcs * volmdlr.TWO_PI
            t_points = []
            for j in range(number_arcs):
                phi = j / number_arcs * volmdlr.TWO_PI
                t_points.append(self.point2d_to_3d(volmdlr.Point2D(theta, phi)))
            ax = wires.ClosedPolygon3D(t_points).plot(ax=ax, edge_style=EdgeStyle(color=color, alpha=alpha))

        return ax

    def point_projection(self, point3d):
        """
        Returns the projection of the point on the toroidal surface.

        :param point3d: Point to project.
        :type point3d: volmdlr.Point3D
        :return: A point on the surface
        :rtype: volmdlr.Point3D
        """
        x, y, z = self.frame.global_to_local_coordinates(point3d)

        if abs(x) < 1e-12:
            x = 0
        if abs(y) < 1e-12:
            y = 0

        theta = math.atan2(y, x)

        vector_to_tube_center = volmdlr.Vector3D(self.tore_radius * math.cos(theta),
                                                 self.tore_radius * math.sin(theta), 0)
        vector_from_tube_center_to_point = volmdlr.Vector3D(x, y, z) - vector_to_tube_center
        phi = volmdlr.geometry.vectors3d_angle(vector_to_tube_center, vector_from_tube_center_to_point)
        if z < 0:
            phi = 2 * math.pi - phi
        if abs(theta) < 1e-9:
            theta = 0.0
        if abs(phi) < 1e-9:
            phi = 0.0
        return self.point2d_to_3d(volmdlr.Point2D(theta, phi))


class ConicalSurface3D(PeriodicalSurface):
    """
    The local plane is defined by (theta, z).

    :param frame: Cone's frame to position it: frame.w is axis of cone frame. Origin is at the angle of the cone.
    :param semi_angle: cone's semi-angle.
    """
    face_class = 'ConicalFace3D'
    x_periodicity = volmdlr.TWO_PI
    y_periodicity = None

    def __init__(self, frame: volmdlr.Frame3D, semi_angle: float,
                 name: str = ''):
        self.frame = frame
        self.semi_angle = semi_angle
        PeriodicalSurface.__init__(self, name=name)

    def plot(self, ax=None, color='grey', alpha=0.5, **kwargs):
        z = kwargs.get("z", 0.5)
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        self.frame.plot(ax=ax, ratio=z)
        x = z * math.tan(self.semi_angle)
        # point1 = self.frame.local_to_global_coordinates(volmdlr.Point3D(-x, 0, -z))
        point1 = self.frame.origin
        point2 = self.frame.local_to_global_coordinates(volmdlr.Point3D(x, 0, z))
        generatrix = edges.LineSegment3D(point1, point2)
        for i in range(37):
            theta = i / 36. * volmdlr.TWO_PI
            wire = generatrix.rotation(self.frame.origin, self.frame.w, theta)
            wire.plot(ax=ax, edge_style=EdgeStyle(color=color, alpha=alpha))
        return ax

    @classmethod
    def from_step(cls, arguments, object_dict, **kwargs):
        """
        Converts a step primitive to a ConicalSurface3D.

        :param arguments: The arguments of the step primitive.
        :type arguments: list
        :param object_dict: The dictionary containing all the step primitives
            that have already been instantiated.
        :type object_dict: dict
        :return: The corresponding ConicalSurface3D object.
        :rtype: :class:`volmdlr.faces.ConicalSurface3D`
        """

        length_conversion_factor = kwargs.get("length_conversion_factor", 1)
        angle_conversion_factor = kwargs.get("angle_conversion_factor", 1)

        frame3d = object_dict[arguments[1]]
        u, w = frame3d.v, frame3d.u
        u.normalize()
        w.normalize()
        v = w.cross(u)
        radius = float(arguments[2]) * length_conversion_factor
        semi_angle = float(arguments[3]) * angle_conversion_factor
        origin = frame3d.origin - radius / math.tan(semi_angle) * w
        frame_direct = volmdlr.Frame3D(origin, u, v, w)
        return cls(frame_direct, semi_angle, arguments[0][1:-1])

    def to_step(self, current_id):
        frame = volmdlr.Frame3D(self.frame.origin, self.frame.w, self.frame.u,
                                self.frame.v)
        content, frame_id = frame.to_step(current_id)
        current_id = frame_id + 1
        content += f"#{current_id} = CONICAL_SURFACE('{self.name}',#{frame_id},{0.},{round(self.semi_angle, 3)});\n"
        return content, [current_id]

    def frame_mapping(self, frame: volmdlr.Frame3D, side: str):
        """
        Changes frame_mapping and return a new ConicalSurface3D.

        :param side: 'old' or 'new'
        """
        new_frame = self.frame.frame_mapping(frame, side)
        return ConicalSurface3D(new_frame, self.semi_angle, name=self.name)

    def frame_mapping_inplace(self, frame: volmdlr.Frame3D, side: str):
        """
        Changes frame_mapping and the object is updated in-place.

        :param side:'old' or 'new'
        """
        warnings.warn("'in-place' methods are deprecated. Use a not in-place method instead.", DeprecationWarning)

        new_frame = self.frame.frame_mapping(frame, side)
        self.frame = new_frame

    def point2d_to_3d(self, point2d: volmdlr.Point2D):
        """
        Coverts a parametric coordinate on the surface into a 3D spatial point (x, y, z).

        :param point2d: Point at the ConicalSuface3D
        :type point2d: `volmdlr.`Point2D`
        """
        theta, z = point2d
        radius = math.tan(self.semi_angle) * z
        new_point = volmdlr.Point3D(radius * math.cos(theta),
                                    radius * math.sin(theta),
                                    z)
        return self.frame.local_to_global_coordinates(new_point)

    def point3d_to_2d(self, point3d: volmdlr.Point3D):
        """
        Returns the cylindrical coordinates volmdlr.Point2D(theta, z) of a Cartesian coordinates point (x, y, z).

        :param point3d: Point at the CylindricalSuface3D.
        :type point3d: :class:`volmdlr.`Point3D`
        """
        x, y, z = self.frame.global_to_local_coordinates(point3d)
        # Do not delete this, mathematical problem when x and y close to zero (should be zero) but not 0
        # Generally this is related to uncertainty of step files.
        if abs(x) < 1e-12:
            x = 0
        if abs(y) < 1e-12:
            y = 0
        theta = math.atan2(y, x)
        if abs(theta) < 1e-9:
            theta = 0.0
        return volmdlr.Point2D(theta, z)

    def rectangular_cut(self, theta1: float, theta2: float,
                        z1: float, z2: float, name: str = ''):
        """Deprecated method, Use ConicalFace3D from_surface_rectangular_cut method."""
        raise AttributeError("ConicalSurface3D.rectangular_cut is deprecated."
                             "Use the class_method from_surface_rectangular_cut in ConicalFace3D instead")

    def linesegment3d_to_2d(self, linesegment3d):
        """
        Converts the primitive from 3D spatial coordinates to its equivalent 2D primitive in the parametric space.
        """
        start = self.point3d_to_2d(linesegment3d.start)
        end = self.point3d_to_2d(linesegment3d.end)
        if start.x != end.x and start.is_close(volmdlr.Point2D(0, 0)):
            start = volmdlr.Point2D(end.x, 0)
        elif start.x != end.x and end == volmdlr.Point2D(0, 0):
            end = volmdlr.Point2D(start.x, 0)
        elif start.x != end.x:
            end = volmdlr.Point2D(start.x, end.y)
        if not start.is_close(end):
            return [edges.LineSegment2D(start, end)]
        return [edges.BSplineCurve2D.from_points_interpolation([start, end], 1, False)]

    def linesegment2d_to_3d(self, linesegment2d):
        if linesegment2d.name == "construction":
            return None
        theta1, z1 = linesegment2d.start
        theta2, z2 = linesegment2d.end

        if math.isclose(theta1, theta2, abs_tol=1e-4):
            return [edges.LineSegment3D(
                self.point2d_to_3d(linesegment2d.start),
                self.point2d_to_3d(linesegment2d.end),
            )]
        if math.isclose(z1, z2, abs_tol=1e-4) and math.isclose(z1, 0.,
                                                               abs_tol=1e-6):
            return []
        if math.isclose(z1, z2, abs_tol=1e-4):
            if abs(theta1 - theta2) == volmdlr.TWO_PI:
                return [edges.FullArc3D(center=self.frame.origin + z1 * self.frame.w,
                                      start_end=self.point2d_to_3d(linesegment2d.start),
                                      normal=self.frame.w)]

            return [edges.Arc3D(
                self.point2d_to_3d(linesegment2d.start),
                self.point2d_to_3d(
                    volmdlr.Point2D(0.5 * (theta1 + theta2), z1)),
                self.point2d_to_3d(linesegment2d.end))
            ]
        raise NotImplementedError('Ellipse?')

    def contour3d_to_2d(self, contour3d):
        """
        Transforms a Contour3D into a Contour2D in the parametric domain of the surface.

        :param contour3d: The contour to be transformed.
        :type contour3d: :class:`wires.Contour3D`
        :return: A 2D contour object.
        :rtype: :class:`wires.Contour2D`
        """
        primitives2d = self.primitives3d_to_2d(contour3d.primitives)

        wire2d = wires.Wire2D(primitives2d)
        delta_x = abs(wire2d.primitives[0].start.x - wire2d.primitives[-1].end.x)
        if math.isclose(delta_x, volmdlr.TWO_PI, abs_tol=1e-3) and wire2d.is_ordered():
            if len(primitives2d) > 1:
                # very specific conical case due to the singularity in the point z = 0 on parametric domain.
                if primitives2d[-2].start.y == 0.0:
                    primitives2d = self.repair_primitives_periodicity(primitives2d)
            return wires.Contour2D(primitives2d)
        # Fix contour
        primitives2d = self.repair_primitives_periodicity(primitives2d)
        return wires.Contour2D(primitives2d)

    def translation(self, offset: volmdlr.Vector3D):
        """
        ConicalSurface3D translation.

        :param offset: translation vector.
        :return: A new translated ConicalSurface3D.
        """
        return self.__class__(self.frame.translation(offset),
                              self.semi_angle)

    def translation_inplace(self, offset: volmdlr.Vector3D):
        """
        ConicalSurface3D translation. Object is updated in-place.

        :param offset: translation vector.
        """
        warnings.warn("'in-place' methods are deprecated. Use a not in-place method instead.", DeprecationWarning)

        self.frame.translation_inplace(offset)

    def rotation(self, center: volmdlr.Point3D,
                 axis: volmdlr.Vector3D, angle: float):
        """
        ConicalSurface3D rotation.

        :param center: rotation center.
        :param axis: rotation axis.
        :param angle: angle rotation.
        :return: a new rotated ConicalSurface3D.
        """
        new_frame = self.frame.rotation(center=center, axis=axis, angle=angle)
        return self.__class__(new_frame, self.semi_angle)

    def rotation_inplace(self, center: volmdlr.Point3D,
                         axis: volmdlr.Vector3D, angle: float):
        """
        ConicalSurface3D rotation. Object is updated in-place.

        :param center: rotation center.
        :param axis: rotation axis.
        :param angle: rotation angle.
        """
        warnings.warn("'in-place' methods are deprecated. Use a not in-place method instead.", DeprecationWarning)

        self.frame.rotation_inplace(center, axis, angle)

    def repair_primitives_periodicity(self, primitives2d):
        """
        Repairs the continuity of the 2D contour while using contour3d_to_2d on periodic surfaces.

        :param primitives2d: The primitives in parametric surface domain.
        :type primitives2d: list
        :return: A list of primitives.
        :rtype: list
        """
        # Search for a primitive that can be used as reference for repairing periodicity
        pos = vm_parametric.find_index_defined_brep_primitive_on_periodical_surface(primitives2d,
                                                                                    [self.x_periodicity,
                                                                                     self.y_periodicity])
        if pos != 0:
            primitives2d = primitives2d[pos:] + primitives2d[:pos]

        i = 1
        while i < len(primitives2d):
            previous_primitive = primitives2d[i - 1]
            delta = previous_primitive.end - primitives2d[i].start
            if not math.isclose(delta.norm(), 0, abs_tol=1e-5):
                if primitives2d[i].end.is_close(primitives2d[i - 1].end, tol=1e-4) and \
                        math.isclose(primitives2d[i].length(), volmdlr.TWO_PI, abs_tol=1e-4):
                    primitives2d[i] = primitives2d[i].reverse()
                elif delta.norm() and math.isclose(abs(previous_primitive.end.y), 0, abs_tol=1e-6):
                    primitives2d.insert(i, edges.LineSegment2D(previous_primitive.end, primitives2d[i].start,
                                                             name="construction"))
                    i += 1
                else:
                    primitives2d[i] = primitives2d[i].translation(delta)
            # treat very specific case of conical surfaces when the previous primitive and the primitive are a
            # linesegment3d with singularity
            elif math.isclose(primitives2d[i].start.y, 0.0, abs_tol=1e-6) and \
                    math.isclose(primitives2d[i].start.x, primitives2d[i].end.x, abs_tol=1e-6) and \
                    math.isclose(primitives2d[i].start.x, previous_primitive.end.x, abs_tol=1e-6):

                if primitives2d[i + 1].end.x < primitives2d[i].end.x:
                    theta_offset = volmdlr.TWO_PI
                elif primitives2d[i + 1].end.x > primitives2d[i].end.x:
                    theta_offset = -volmdlr.TWO_PI
                primitive1 = edges.LineSegment2D(previous_primitive.end,
                                               previous_primitive.end + volmdlr.Point2D(theta_offset, 0),
                                               name="construction")
                primitive2 = primitives2d[i].translation(volmdlr.Vector2D(theta_offset, 0))
                primitive3 = primitives2d[i + 1].translation(volmdlr.Vector2D(theta_offset, 0))
                primitives2d[i] = primitive1
                primitives2d.insert(i + 1, primitive2)
                primitives2d[i + 2] = primitive3
                i += 1
            i += 1
        if not primitives2d[0].start.is_close(primitives2d[-1].end) \
                and primitives2d[0].start.y == 0.0 and primitives2d[-1].end.y == 0.0:
            primitives2d.append(edges.LineSegment2D(primitives2d[-1].end, primitives2d[0].start))

        return primitives2d

    def face_from_base_and_vertex(self, contour: wires.Contour3D, vertex: volmdlr.Point3D, name: str = ''):

        raise AttributeError(f'Use method from ConicalFace3D{volmdlr.faces.ConicalFace3D.face_from_base_and_vertex}')


class SphericalSurface3D(PeriodicalSurface):
    """
    Defines a spherical surface.

    :param frame: Sphere's frame to position it
    :type frame: volmdlr.Frame3D
    :param radius: Sphere's radius
    :type radius: float
    """
    face_class = 'SphericalFace3D'
    x_periodicity = volmdlr.TWO_PI
    y_periodicity = math.pi

    def __init__(self, frame, radius, name=''):
        self.frame = frame
        self.radius = radius
        PeriodicalSurface.__init__(self, name=name)

        # Hidden Attributes
        self._bbox = None

    @property
    def bounding_box(self):

        if not self._bbox:
            self._bbox = self._bounding_box()
        return self._bbox

    def _bounding_box(self):
        points = [self.frame.origin + volmdlr.Point3D(-self.radius,
                                                      -self.radius,
                                                      -self.radius),
                  self.frame.origin + volmdlr.Point3D(self.radius,
                                                      self.radius,
                                                      self.radius),

                  ]
        return volmdlr.core.BoundingBox.from_points(points)

    def contour2d_to_3d(self, contour2d):
        primitives3d = []
        for primitive2d in contour2d.primitives:
            method_name = f'{primitive2d.__class__.__name__.lower()}_to_3d'
            if hasattr(self, method_name):
                try:
                    primitives_list = getattr(self, method_name)(primitive2d)
                    if primitives_list:
                        primitives3d.extend(primitives_list)
                    else:
                        continue
                except AttributeError:
                    print(f'Class {self.__class__.__name__} does not implement {method_name}'
                          f'with {primitive2d.__class__.__name__}')
            else:
                raise AttributeError(f'Class {self.__class__.__name__} does not implement {method_name}')

        return wires.Contour3D(primitives3d)

    @classmethod
    def from_step(cls, arguments, object_dict, **kwargs):
        """
        Converts a step primitive to a SphericalSurface3D.

        :param arguments: The arguments of the step primitive.
        :type arguments: list
        :param object_dict: The dictionary containing all the step primitives
            that have already been instantiated.
        :type object_dict: dict
        :return: The corresponding SphericalSurface3D object.
        :rtype: :class:`volmdlr.faces.SphericalSurface3D`
        """
        length_conversion_factor = kwargs.get("length_conversion_factor", 1)

        frame3d = object_dict[arguments[1]]
        u_vector, w_vector = frame3d.v, frame3d.u
        u_vector.normalize()
        w_vector.normalize()
        v_vector = w_vector.cross(u_vector)
        frame_direct = volmdlr.Frame3D(frame3d.origin, u_vector, v_vector, w_vector)
        radius = float(arguments[2]) * length_conversion_factor
        return cls(frame_direct, radius, arguments[0][1:-1])

    def point2d_to_3d(self, point2d):
        """
        Coverts a parametric coordinate on the surface into a 3D spatial point (x, y, z).

        source: https://mathcurve.com/surfaces/sphere
        # -pi<theta<pi, -pi/2<phi<pi/2

        :param point2d: Point at the CylindricalSuface3D.
        :type point2d: `volmdlr.`Point2D`
        """
        theta, phi = point2d
        x = self.radius * math.cos(phi) * math.cos(theta)
        y = self.radius * math.cos(phi) * math.sin(theta)
        z = self.radius * math.sin(phi)
        return self.frame.local_to_global_coordinates(volmdlr.Point3D(x, y, z))

    def point3d_to_2d(self, point3d):
        """
        Transform a 3D spatial point (x, y, z) into a 2D spherical parametric point (theta, phi).
        """
        x, y, z = self.frame.global_to_local_coordinates(point3d)
        z = min(self.radius, max(-self.radius, z))

        if z == -0.0:
            z = 0.0

        # Do not delete this, mathematical problem when x and y close to zero (should be zero) but not 0
        # Generally this is related to uncertainty of step files.
        if abs(x) < 1e-12:
            x = 0
        if abs(y) < 1e-12:
            y = 0

        theta = math.atan2(y, x)
        if abs(theta) < 1e-10:
            theta = 0

        z_over_r = z / self.radius
        phi = math.asin(z_over_r)
        if abs(phi) < 1e-10:
            phi = 0

        return volmdlr.Point2D(theta, phi)

    def linesegment2d_to_3d(self, linesegment2d):
        if linesegment2d.name == "construction":
            return []
        start = self.point2d_to_3d(linesegment2d.start)
        interior = self.point2d_to_3d(0.5 * (linesegment2d.start + linesegment2d.end))
        end = self.point2d_to_3d(linesegment2d.end)
        if start.is_close(end) or linesegment2d.length() == 2 * math.pi:
            u_vector = start - self.frame.origin
            u_vector.normalize()
            v_vector = interior - self.frame.origin
            v_vector.normalize()
            normal = u_vector.cross(v_vector)
            return [edges.FullArc3D(self.frame.origin, start, normal)]
        return [edges.Arc3D(start, interior, end)]

    def contour3d_to_2d(self, contour3d):
        """
        Transforms a Contour3D into a Contour2D in the parametric domain of the surface.

        :param contour3d: The contour to be transformed.
        :type contour3d: :class:`wires.Contour3D`
        :return: A 2D contour object.
        :rtype: :class:`wires.Contour2D`
        """
        primitives2d = []

        # Transform the contour's primitives to parametric domain
        for primitive3d in contour3d.primitives:
            method_name = f'{primitive3d.__class__.__name__.lower()}_to_2d'
            if hasattr(self, method_name):
                primitives = getattr(self, method_name)(primitive3d)

                if primitives is None:
                    continue
                primitives2d.extend(primitives)
            else:
                raise NotImplementedError(
                    f'Class {self.__class__.__name__} does not implement {method_name}')
        # Fix contour
        if self.x_periodicity or self.y_periodicity:
            primitives2d = self.repair_primitives_periodicity(primitives2d)
        return wires.Contour2D(primitives2d)

    def is_lat_long_curve(self, theta_list, phi_list):
        """
        Checks if a curve defined on the sphere is a latitude/longitude curve.

        Returns True if it is, False otherwise.
        """
        # Check if curve is a longitude curve (phi is constant)
        if all(math.isclose(theta, theta_list[0], abs_tol=1e-4) for theta in theta_list[1:]):
            return True
        # Check if curve is a latitude curve (theta is constant)
        if all(math.isclose(phi, phi_list[0], abs_tol=1e-4) for phi in phi_list[1:]):
            return True
        return False

    def arc3d_to_2d(self, arc3d):
        """
        Converts the primitive from 3D spatial coordinates to its equivalent 2D primitive in the parametric space.
        """
        start = self.point3d_to_2d(arc3d.start)
        end = self.point3d_to_2d(arc3d.end)
        theta_i, phi_i = self.point3d_to_2d(arc3d.interior)
        theta1, phi1 = start
        theta2, phi2 = end

        point_after_start, point_before_end = self._reference_points(arc3d)
        theta3, _ = point_after_start
        theta4, _ = point_before_end

        # Fix sphere singularity point
        if math.isclose(abs(phi1), 0.5 * math.pi, abs_tol=1e-5) and theta1 == 0.0 \
                and math.isclose(theta3, theta_i, abs_tol=1e-6) and math.isclose(theta4, theta_i, abs_tol=1e-6):
            theta1 = theta_i
            start = volmdlr.Point2D(theta1, phi1)
        if math.isclose(abs(phi2), 0.5 * math.pi, abs_tol=1e-5) and theta2 == 0.0 \
                and math.isclose(theta3, theta_i, abs_tol=1e-6) and math.isclose(theta4, theta_i, abs_tol=1e-6):
            theta2 = theta_i
            end = volmdlr.Point2D(theta2, phi2)

        start, end = vm_parametric.arc3d_to_spherical_coordinates_verification(start, end, arc3d.angle,
                                                                               [point_after_start, point_before_end],
                                                                               [self.x_periodicity,
                                                                                self.y_periodicity])
        if start == end:  # IS THIS POSSIBLE ?
            return [edges.LineSegment2D(start, start + volmdlr.TWO_PI * volmdlr.X2D)]
        if self.is_lat_long_curve([theta1, theta_i, theta2], [phi1, phi_i, phi2]):
            return [edges.LineSegment2D(start, end)]

        return self.arc3d_to_2d_with_singularity(arc3d, start, end, point_after_start)

    def arc3d_to_2d_with_singularity(self, arc3d, start, end, point_after_start):
        # trying to treat when the arc starts at theta1 passes at the singularity at |phi| = 0.5*math.pi
        # and ends at theta2 = theta1 + math.pi
        theta1, phi1 = start
        theta2, phi2 = end
        theta3, phi3 = point_after_start

        half_pi = 0.5 * math.pi
        point_positive_singularity = self.point2d_to_3d(volmdlr.Point2D(theta1, half_pi))
        point_negative_singularity = self.point2d_to_3d(volmdlr.Point2D(theta1, -half_pi))
        positive_singularity = arc3d.point_belongs(point_positive_singularity, 1e-4)
        negative_singularity = arc3d.point_belongs(point_negative_singularity, 1e-4)
        interior = self.point3d_to_2d(arc3d.interior)
        if positive_singularity and negative_singularity:
            thetai = interior.x
            is_trigo = phi1 < phi3
            if is_trigo and abs(phi1) > half_pi:
                half_pi = 0.5 * math.pi
            elif is_trigo and abs(phi1) < half_pi:
                half_pi = - 0.5 * math.pi
            elif not is_trigo and abs(phi1) > half_pi:
                half_pi = - 0.5 * math.pi
            elif is_trigo and abs(phi1) < half_pi:
                half_pi = 0.5 * math.pi
            return [edges.LineSegment2D(volmdlr.Point2D(theta1, phi1), volmdlr.Point2D(theta1, -half_pi)),
                    edges.LineSegment2D(volmdlr.Point2D(theta1, -half_pi), volmdlr.Point2D(thetai, -half_pi),
                                      name="construction"),
                    edges.LineSegment2D(volmdlr.Point2D(thetai, -half_pi), volmdlr.Point2D(thetai, half_pi)),
                    edges.LineSegment2D(volmdlr.Point2D(thetai, half_pi), volmdlr.Point2D(theta2, half_pi),
                                      name="construction"),
                    edges.LineSegment2D(volmdlr.Point2D(theta2, half_pi), volmdlr.Point2D(theta2, phi2))
                    ]

        if (positive_singularity or negative_singularity) and \
                math.isclose(abs(theta2 - theta1), math.pi, abs_tol=1e-4):
            if abs(phi1) == 0.5 * math.pi:
                return [edges.LineSegment2D(volmdlr.Point2D(theta3, phi1), volmdlr.Point2D(theta2, phi2))]
            if theta1 == math.pi and theta2 != math.pi:
                theta1 = -math.pi
            if theta2 == math.pi and theta1 != math.pi:
                theta2 = -math.pi

            return [edges.LineSegment2D(volmdlr.Point2D(theta1, phi1), volmdlr.Point2D(theta1, half_pi)),
                    edges.LineSegment2D(volmdlr.Point2D(theta1, half_pi), volmdlr.Point2D(theta2, half_pi),
                                      name="construction"),
                    edges.LineSegment2D(volmdlr.Point2D(theta2, half_pi), volmdlr.Point2D(theta2, phi2))
                    ]

        # maybe this is incomplete and not exact
        angle3d = arc3d.angle
        number_points = math.ceil(angle3d * 50) + 1  # 50 points per radian
        number_points = max(number_points, 5)
        points3d = arc3d.discretization_points(number_points=number_points)
        points = [self.point3d_to_2d(p) for p in points3d]

        points[0] = start  # to take into account all the previous verification
        points[-1] = end  # to take into account all the previous verification

        if theta3 < theta1 < theta2:
            points = [p - volmdlr.Point2D(self.x_periodicity, 0) if p.x > 0 else p for p in points]
        elif theta3 > theta1 > theta2:
            points = [p + volmdlr.Point2D(self.x_periodicity, 0) if p.x < 0 else p for p in points]

        return [edges.BSplineCurve2D.from_points_interpolation(points, 2)]

    def bsplinecurve3d_to_2d(self, bspline_curve3d):
        """
        Converts the primitive from 3D spatial coordinates to its equivalent 2D primitive in the parametric space.
        """
        length = bspline_curve3d.length()
        n = len(bspline_curve3d.control_points)
        points3d = bspline_curve3d.discretization_points(number_points=n)
        points = [self.point3d_to_2d(point) for point in points3d]
        theta1, phi1 = points[0]
        theta2, phi2 = points[-1]

        theta3, _ = self.point3d_to_2d(bspline_curve3d.point_at_abscissa(0.001 * length))
        # make sure that the reference angle is not undefined
        if abs(theta3) == math.pi:
            theta3, _ = self.point3d_to_2d(bspline_curve3d.point_at_abscissa(0.002 * length))

        # Verify if theta1 or theta2 point should be -pi because atan2() -> ]-pi, pi]
        if abs(theta1) == math.pi:
            theta1 = repair_start_end_angle_periodicity(theta1, theta3)
        if abs(theta2) == math.pi:
            theta4, _ = self.point3d_to_2d(bspline_curve3d.point_at_abscissa(0.98 * length))
            # make sure that the reference angle is not undefined
            if abs(theta4) == math.pi:
                theta4, _ = self.point3d_to_2d(bspline_curve3d.point_at_abscissa(0.97 * length))
            theta2 = repair_start_end_angle_periodicity(theta2, theta4)

        points[0] = volmdlr.Point2D(theta1, phi1)
        points[-1] = volmdlr.Point2D(theta2, phi2)

        theta_list = [point.x for point in points]
        theta_discontinuity, indexes_theta_discontinuity = angle_discontinuity(theta_list)
        if theta_discontinuity:
            points = self._fix_angle_discontinuity_on_discretization_points(points, indexes_theta_discontinuity, "x")

        return [edges.BSplineCurve2D.from_points_interpolation(points, degree=bspline_curve3d.degree,
                                                             periodic=bspline_curve3d.periodic)]

    def bsplinecurve2d_to_3d(self, bspline_curve2d):
        # TODO: this is incomplete, a bspline_curve2d can be also a bspline_curve3d
        i = round(0.5 * len(bspline_curve2d.points))
        start = self.point2d_to_3d(bspline_curve2d.points[0])
        interior = self.point2d_to_3d(bspline_curve2d.points[i])
        end = self.point2d_to_3d(bspline_curve2d.points[-1])
        arc3d = edges.Arc3D(start, interior, end)
        flag = True
        points3d = [self.point2d_to_3d(p) for p in bspline_curve2d.points]
        for point in points3d:
            if not arc3d.point_belongs(point, 1e-4):
                flag = False
                break
        if flag:
            return [arc3d]

        return [edges.BSplineCurve3D.from_points_interpolation(points3d, degree=bspline_curve2d.degree,
                                                             periodic=bspline_curve2d.periodic)]

    def fullarc3d_to_2d(self, fullarc3d):
        """
        Converts the primitive from 3D spatial coordinates to its equivalent 2D primitive in the parametric space.
        """
        # TODO: On a spherical surface we can have fullarc3d in any plane
        length = fullarc3d.length()

        theta1, phi1 = self.point3d_to_2d(fullarc3d.start)
        theta2, phi2 = self.point3d_to_2d(fullarc3d.end)
        theta3, phi3 = self.point3d_to_2d(fullarc3d.point_at_abscissa(0.001 * length))
        theta4, _ = self.point3d_to_2d(fullarc3d.point_at_abscissa(0.98 * length))

        if self.frame.w.is_colinear_to(fullarc3d.normal):
            point1 = volmdlr.Point2D(theta1, phi1)
            if theta1 > theta3:
                point2 = volmdlr.Point2D(theta1 - volmdlr.TWO_PI, phi2)
            elif theta1 < theta3:
                point2 = volmdlr.Point2D(theta1 + volmdlr.TWO_PI, phi2)
            return [edges.LineSegment2D(point1, point2)]

        if math.isclose(self.frame.w.dot(fullarc3d.normal), 0, abs_tol=1e-4):
            if theta1 > theta3:
                theta_plus_pi = theta1 - math.pi
            else:
                theta_plus_pi = theta1 + math.pi
            if phi1 > phi3:
                half_pi = 0.5 * math.pi
            else:
                half_pi = -0.5 * math.pi
            if abs(phi1) == 0.5 * math.pi:
                return [edges.LineSegment2D(volmdlr.Point2D(theta3, phi1), volmdlr.Point2D(theta3, -half_pi)),
                        edges.LineSegment2D(volmdlr.Point2D(theta4, -half_pi), volmdlr.Point2D(theta4, phi2))]

            return [edges.LineSegment2D(volmdlr.Point2D(theta1, phi1), volmdlr.Point2D(theta1, -half_pi)),
                    edges.LineSegment2D(volmdlr.Point2D(theta_plus_pi, -half_pi),
                                      volmdlr.Point2D(theta_plus_pi, half_pi)),
                    edges.LineSegment2D(volmdlr.Point2D(theta1, half_pi), volmdlr.Point2D(theta1, phi2))]

        points = [self.point3d_to_2d(p) for p in fullarc3d.discretization_points(angle_resolution=25)]

        # Verify if theta1 or theta2 point should be -pi because atan2() -> ]-pi, pi]
        theta1 = vm_parametric.repair_start_end_angle_periodicity(theta1, theta3)
        theta2 = vm_parametric.repair_start_end_angle_periodicity(theta2, theta4)

        points[0] = volmdlr.Point2D(theta1, phi1)
        points[-1] = volmdlr.Point2D(theta2, phi2)

        if theta3 < theta1 < theta2:
            points = [p - volmdlr.Point2D(volmdlr.TWO_PI, 0) if p.x > 0 else p for p in points]
        elif theta3 > theta1 > theta2:
            points = [p + volmdlr.Point2D(volmdlr.TWO_PI, 0) if p.x < 0 else p for p in points]

        return [edges.BSplineCurve2D.from_points_interpolation(points, 2)]

    def plot(self, ax=None, color='grey', alpha=0.5):
        """Plot sphere arcs."""
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        self.frame.plot(ax=ax)
        for i in range(20):
            theta = i / 20. * volmdlr.TWO_PI
            t_points = []
            for j in range(20):
                phi = j / 20. * volmdlr.TWO_PI
                t_points.append(self.point2d_to_3d(volmdlr.Point2D(theta, phi)))
            ax = wires.ClosedPolygon3D(t_points).plot(ax=ax, edge_style=EdgeStyle(color=color, alpha=alpha))

        return ax

    def rectangular_cut(self, theta1, theta2, phi1, phi2, name=''):
        """Deprecated method, Use ShericalFace3D from_surface_rectangular_cut method."""
        raise AttributeError('Use ShericalFace3D from_surface_rectangular_cut method')

    def triangulation(self):
        face = self.rectangular_cut(0, volmdlr.TWO_PI, -0.5 * math.pi, 0.5 * math.pi)
        return face.triangulation()

    def repair_primitives_periodicity(self, primitives2d):
        """
        Repairs the continuity of the 2D contour while using contour3d_to_2d on periodic surfaces.

        :param primitives2d: The primitives in parametric surface domain.
        :type primitives2d: list
        :return: A list of primitives.
        :rtype: list
        """
        # # Search for a primitive that can be used as reference for repairing periodicity
        i = 1
        while i < len(primitives2d):
            previous_primitive = primitives2d[i - 1]
            delta = previous_primitive.end - primitives2d[i].start
            if not math.isclose(delta.norm(), 0, abs_tol=1e-5):
                if primitives2d[i].end == primitives2d[i - 1].end and \
                        primitives2d[i].length() == volmdlr.TWO_PI:
                    primitives2d[i] = primitives2d[i].reverse()
                elif math.isclose(abs(previous_primitive.end.y), 0.5 * math.pi, abs_tol=1e-6):
                    primitives2d.insert(i, edges.LineSegment2D(previous_primitive.end, primitives2d[i].start,
                                                             name="construction"))
                else:
                    primitives2d[i] = primitives2d[i].translation(delta)
            i += 1
        #     return primitives2d
        # primitives2d = repair(primitives2d)
        last_end = primitives2d[-1].end
        first_start = primitives2d[0].start
        if not last_end.is_close(first_start, tol=1e-3):
            last_end_3d = self.point2d_to_3d(last_end)
            first_start_3d = self.point2d_to_3d(first_start)
            if last_end_3d.is_close(first_start_3d, 1e-6) and \
                    not math.isclose(abs(last_end.y), 0.5 * math.pi, abs_tol=1e-5):
                if first_start.x > last_end.x:
                    half_pi = -0.5 * math.pi
                else:
                    half_pi = 0.5 * math.pi
                lines = [edges.LineSegment2D(last_end, volmdlr.Point2D(last_end.x, half_pi), name="construction"),
                         edges.LineSegment2D(volmdlr.Point2D(last_end.x, half_pi),
                                           volmdlr.Point2D(first_start.x, half_pi), name="construction"),
                         edges.LineSegment2D(volmdlr.Point2D(first_start.x, half_pi),
                                           first_start, name="construction")
                         ]
                primitives2d.extend(lines)
            else:
                primitives2d.append(edges.LineSegment2D(last_end, first_start))
        return primitives2d

    def rotation(self, center: volmdlr.Point3D, axis: volmdlr.Vector3D, angle: float):
        """
        Spherical Surface 3D rotation.

        :param center: rotation center
        :param axis: rotation axis
        :param angle: angle rotation
        :return: a new rotated Spherical Surface 3D
        """
        new_frame = self.frame.rotation(center=center, axis=axis, angle=angle)
        return SphericalSurface3D(new_frame, self.radius)

    def translation(self, offset: volmdlr.Vector3D):
        """
        Spherical Surface 3D translation.

        :param offset: translation vector
        :return: A new translated Spherical Surface 3D
        """
        new_frame = self.frame.translation(offset)
        return SphericalSurface3D(new_frame, self.radius)

    def frame_mapping(self, frame: volmdlr.Frame3D, side: str):
        """
        Changes Spherical Surface 3D's frame and return a new Spherical Surface 3D.

        :param frame: Frame of reference
        :type frame: `volmdlr.Frame3D`
        :param side: 'old' or 'new'
        """
        new_frame = self.frame.frame_mapping(frame, side)
        return SphericalSurface3D(new_frame, self.radius)


class RuledSurface3D(Surface3D):
    """
    Defines a ruled surface between two wires.

    :param wire1: Wire
    :type wire1: :class:`vmw.Wire3D`
    :param wire2: Wire
    :type wire2: :class:`wires.Wire3D`
    """
    face_class = 'RuledFace3D'

    def __init__(self, wire1: wires.Wire3D, wire2: wires.Wire3D, name: str = ''):
        self.wire1 = wire1
        self.wire2 = wire2
        self.length1 = wire1.length()
        self.length2 = wire2.length()
        Surface3D.__init__(self, name=name)

    def point2d_to_3d(self, point2d: volmdlr.Point2D):
        x, y = point2d
        point1 = self.wire1.point_at_abscissa(x * self.length1)
        point2 = self.wire2.point_at_abscissa(x * self.length2)
        joining_line = edges.LineSegment3D(point1, point2)
        point = joining_line.point_at_abscissa(y * joining_line.length())
        return point

    def point3d_to_2d(self, point3d):
        raise NotImplementedError

    def rectangular_cut(self, x1: float, x2: float,
                        y1: float, y2: float, name: str = ''):
        """Deprecated method, Use RuledFace3D from_surface_rectangular_cut method."""
        raise NotImplementedError('Use RuledFace3D from_surface_rectangular_cut method')


class ExtrusionSurface3D(Surface3D):
    """
    Defines a surface of revolution.

    An extrusion surface is a surface that is a generic cylindrical surface generated by the linear
    extrusion of a curve, generally an Ellipse or a B-Spline curve.

    :param edge: edge.
    :type edge: Union[:class:`vmw.Wire3D`, :class:`vmw.Contour3D`]
    :param axis_point: Axis placement
    :type axis_point: :class:`volmdlr.Point3D`
    :param axis: Axis of revolution
    :type axis: :class:`volmdlr.Vector3D`
    """
    face_class = 'ExtrusionFace3D'
    y_periodicity = None

    def __init__(self, edge: Union[edges.FullArcEllipse3D, edges.BSplineCurve3D],
                 direction: volmdlr.Vector3D, name: str = ''):
        self.edge = edge
        direction.normalize()
        self.direction = direction
        self.frame = volmdlr.Frame3D.from_point_and_vector(edge.start, direction, volmdlr.Z3D)
        self._x_periodicity = False

        Surface3D.__init__(self, name=name)

    @property
    def x_periodicity(self):
        if self._x_periodicity:
            return self._x_periodicity
        start = self.edge.start
        end = self.edge.end
        if start.is_close(end, 1e-4):
            return 1
        return None

    @x_periodicity.setter
    def x_periodicity(self, value):
        self._x_periodicity = value

    def point2d_to_3d(self, point2d: volmdlr.Point2D):
        """
        Transform a parametric (u, v) point into a 3D Cartesian point (x, y, z).

        # u = [0, 1] and v = z
        """
        u, v = point2d
        if abs(u) < 1e-7:
            u = 0.0
        if abs(v) < 1e-7:
            v = 0.

        point_at_curve_global = self.edge.point_at_abscissa(u * self.edge.length())
        point_at_curve_local = self.frame.global_to_local_coordinates(point_at_curve_global)
        # x, y, z = point_at_curve_local
        point_local = point_at_curve_local.translation(volmdlr.Vector3D(0, 0, v))
        return self.frame.local_to_global_coordinates(point_local)

    def point3d_to_2d(self, point3d):
        """
        Transform a 3D Cartesian point (x, y, z) into a parametric (u, v) point.
        """
        x, y, z = self.frame.global_to_local_coordinates(point3d)
        if abs(x) < 1e-7:
            x = 0.0
        if abs(y) < 1e-7:
            y = 0.0
        if abs(z) < 1e-7:
            z = 0.0
        v = z
        point_at_curve_local = volmdlr.Point3D(x, y, 0)
        point_at_curve_global = self.frame.local_to_global_coordinates(point_at_curve_local)

        u = self.edge.abscissa(point_at_curve_global) / self.edge.length()

        u = min(u, 1.0)
        return volmdlr.Point2D(u, v)

    def rectangular_cut(self, x1: float = 0.0, x2: float = 1.0,
                        y1: float = 0.0, y2: float = 1.0, name: str = ''):
        """Deprecated method, Use ExtrusionFace3D from_surface_rectangular_cut method."""
        raise AttributeError('Use ExtrusionFace3D from_surface_rectangular_cut method')

    def plot(self, ax=None, color='grey', alpha=0.5, z: float = 0.5):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        for i in range(21):
            step = i / 20. * z
            wire = self.edge.translation(step * self.frame.w)
            wire.plot(ax=ax, color=color, alpha=alpha)

        return ax

    @classmethod
    def from_step(cls, arguments, object_dict, **kwargs):
        name = arguments[0][1:-1]
        edge = object_dict[arguments[1]]
        if edge.__class__ is wires.Ellipse3D:
            start_end = edge.center + edge.major_axis * edge.major_dir
            fullarcellipse = edges.FullArcEllipse3D(start_end, edge.major_axis, edge.minor_axis,
                                                  edge.center, edge.normal, edge.major_dir, edge.name)
            edge = fullarcellipse
            direction = -object_dict[arguments[2]]
            surface = cls(edge=edge, direction=direction, name=name)
            surface.x_periodicity = 1

        else:
            direction = object_dict[arguments[2]]
            surface = cls(edge=edge, direction=direction, name=name)
        return surface

    def arcellipse3d_to_2d(self, arcellipse3d):
        """
        Transformation of an arc-ellipse 3d to 2d, in a cylindrical surface.

        """
        if isinstance(self.edge, edges.FullArcEllipse3D):
            start2d = self.point3d_to_2d(arcellipse3d.start)
            end2d = self.point3d_to_2d(arcellipse3d.end)
            return [edges.LineSegment2D(start2d, end2d)]
        points = [self.point3d_to_2d(p)
                  for p in arcellipse3d.discretization_points(number_points=15)]

        bsplinecurve2d = edges.BSplineCurve2D.from_points_interpolation(points, degree=2)
        return [bsplinecurve2d]

    def fullarcellipse3d_to_2d(self, fullarcellipse3d):
        length = fullarcellipse3d.length()
        start = self.point3d_to_2d(fullarcellipse3d.start)
        end = self.point3d_to_2d(fullarcellipse3d.end)

        u3, _ = self.point3d_to_2d(fullarcellipse3d.point_at_abscissa(0.01 * length))
        if u3 > 0.5:
            p1 = volmdlr.Point2D(1, start.y)
            p2 = volmdlr.Point2D(0, end.y)
        elif u3 < 0.5:
            p1 = volmdlr.Point2D(0, start.y)
            p2 = volmdlr.Point2D(1, end.y)
        else:
            raise NotImplementedError
        return [edges.LineSegment2D(p1, p2)]

    def linesegment2d_to_3d(self, linesegment2d):
        """
        Converts a BREP line segment 2D onto a 3D primitive on the surface.
        """
        start3d = self.point2d_to_3d(linesegment2d.start)
        end3d = self.point2d_to_3d(linesegment2d.end)
        u1, z1 = linesegment2d.start
        u2, z2 = linesegment2d.end
        if math.isclose(u1, u2, abs_tol=1e-4):
            return [edges.LineSegment3D(start3d, end3d)]
        if math.isclose(z1, z2, abs_tol=1e-4):
            if math.isclose(abs(u1 - u2), 1.0, abs_tol=1e-4):
                primitive = self.edge.translation(self.direction * z1)
                return [primitive]
            primitive = self.edge.translation(self.direction * z1)
            primitive = primitive.trim(start3d, end3d)
            return [primitive]
        n = 10
        degree = 3
        points = [self.point2d_to_3d(point2d) for point2d in linesegment2d.discretization_points(number_points=n)]
        periodic = points[0].is_close(points[-1])
        return [edges.BSplineCurve3D.from_points_interpolation(points, degree, periodic)]

    def bsplinecurve3d_to_2d(self, bspline_curve3d):
        n = len(bspline_curve3d.control_points)
        points = [self.point3d_to_2d(point)
                  for point in bspline_curve3d.discretization_points(number_points=n)]
        start = points[0]
        end = points[-1]
        if not start.is_close(end):
            linesegment = edges.LineSegment2D(start, end)
            flag = True
            for point in points:
                if not linesegment.point_belongs(point):
                    flag = False
                    break
            if flag:
                return [linesegment]

        # Is this always True?
        n = len(bspline_curve3d.control_points)
        points = [self.point3d_to_2d(p)
                  for p in bspline_curve3d.discretization_points(number_points=n)]
        return [edges.BSplineCurve2D.from_points_interpolation(
            points, bspline_curve3d.degree, bspline_curve3d.periodic)]


class RevolutionSurface3D(PeriodicalSurface):
    """
    Defines a surface of revolution.

    :param wire: Wire.
    :type wire: Union[:class:`vmw.Wire3D`, :class:`vmw.Contour3D`]
    :param axis_point: Axis placement
    :type axis_point: :class:`volmdlr.Point3D`
    :param axis: Axis of revolution
    :type axis: :class:`volmdlr.Vector3D`
    """
    face_class = 'RevolutionFace3D'
    x_periodicity = volmdlr.TWO_PI
    y_periodicity = None

    def __init__(self, wire: Union[wires.Wire3D, wires.Contour3D],
                 axis_point: volmdlr.Point3D, axis: volmdlr.Vector3D, name: str = ''):
        self.wire = wire
        self.axis_point = axis_point
        self.axis = axis

        point1 = wire.point_at_abscissa(0)
        if point1 == axis_point:
            point1 = wire.point_at_abscissa(0.1 * wire.length())
        vector1 = point1 - axis_point
        w_vector = axis
        w_vector.normalize()
        u_vector = vector1 - vector1.vector_projection(w_vector)
        u_vector.normalize()
        v_vector = w_vector.cross(u_vector)
        self.frame = volmdlr.Frame3D(origin=axis_point, u=u_vector, v=v_vector, w=w_vector)

        PeriodicalSurface.__init__(self, name=name)

    def point2d_to_3d(self, point2d: volmdlr.Point2D):
        """
        Transform a parametric (u, v) point into a 3D Cartesian point (x, y, z).

        u = [0, 2pi] and v = [0, 1] into a
        """
        u, v = point2d
        point_at_curve = self.wire.point_at_abscissa(v * self.wire.length())
        point = point_at_curve.rotation(self.axis_point, self.axis, u)
        return point

    def point3d_to_2d(self, point3d):
        """
        Transform a 3D Cartesian point (x, y, z) into a parametric (u, v) point.
        """
        x, y, _ = self.frame.global_to_local_coordinates(point3d)
        if abs(x) < 1e-12:
            x = 0
        if abs(y) < 1e-12:
            y = 0
        u = math.atan2(y, x)

        point_at_curve = point3d.rotation(self.axis_point, self.axis, -u)
        v = self.wire.abscissa(point_at_curve) / self.wire.length()
        return volmdlr.Point2D(u, v)

    def rectangular_cut(self, x1: float, x2: float,
                        y1: float, y2: float, name: str = ''):
        """Deprecated method, Use RevolutionFace3D from_surface_rectangular_cut method."""
        raise AttributeError('Use RevolutionFace3D from_surface_rectangular_cut method')

    def plot(self, ax=None, color='grey', alpha=0.5, number_curves: int = 20):
        """
        Plot rotated Revolution surface generatrix.

        :param number_curves: Number of curves to display.
        :type number_curves: int
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        for i in range(number_curves + 1):
            theta = i / number_curves * volmdlr.TWO_PI
            wire = self.wire.rotation(self.axis_point, self.axis, theta)
            wire.plot(ax=ax, edge_style=EdgeStyle(color=color, alpha=alpha))

        return ax

    @classmethod
    def from_step(cls, arguments, object_dict, **kwargs):
        """
        Converts a step primitive to a RevolutionSurface3D.

        :param arguments: The arguments of the step primitive.
        :type arguments: list
        :param object_dict: The dictionary containing all the step primitives
            that have already been instantiated.
        :type object_dict: dict
        :return: The corresponding RevolutionSurface3D object.
        :rtype: :class:`volmdlr.faces.RevolutionSurface3D`
        """
        name = arguments[0][1:-1]
        contour3d = object_dict[arguments[1]]
        axis_point, axis = object_dict[arguments[2]]
        return cls(wire=contour3d, axis_point=axis_point, axis=axis, name=name)

    def fullarc3d_to_2d(self, fullarc3d):
        """
        Converts the primitive from 3D spatial coordinates to its equivalent 2D primitive in the parametric space.
        """
        length = fullarc3d.length()

        start = self.point3d_to_2d(fullarc3d.start)
        end = self.point3d_to_2d(fullarc3d.end)

        theta3, _ = self.point3d_to_2d(fullarc3d.point_at_abscissa(0.001 * length))
        theta4, _ = self.point3d_to_2d(fullarc3d.point_at_abscissa(0.98 * length))

        # make sure that the references points are not undefined
        if abs(theta3) == math.pi:
            theta3, _ = self.point3d_to_2d(fullarc3d.point_at_abscissa(0.002 * length))
        if abs(theta4) == math.pi:
            theta4, _ = self.point3d_to_2d(fullarc3d.point_at_abscissa(0.97 * length))

        start, end = vm_parametric.arc3d_to_cylindrical_coordinates_verification(start, end, volmdlr.TWO_PI, theta3,
                                                                                 theta4)

        theta1, z1 = start
        _, z2 = end

        point1 = volmdlr.Point2D(theta1, z1)
        if theta1 > theta3:
            point2 = volmdlr.Point2D(theta1 + volmdlr.TWO_PI, z2)
        elif theta1 < theta3:
            point2 = volmdlr.Point2D(theta1 - volmdlr.TWO_PI, z2)
        return [edges.LineSegment2D(point1, point2)]


class BSplineSurface3D(Surface3D):
    """
    A class representing a 3D B-spline surface.

    A B-spline surface is a smooth surface defined by a set of control points and
    a set of basis functions called B-spline basis functions. The shape of the
    surface is determined by the position of the control points and can be
    modified by moving the control points.

    :param degree_u: The degree of the B-spline curve in the u direction.
    :type degree_u: int
    :param degree_v: The degree of the B-spline curve in the v direction.
    :type degree_v: int
    :param control_points: A list of 3D control points that define the shape of
        the surface.
    :type control_points: List[`volmdlr.Point3D`]
    :param nb_u: The number of control points in the u direction.
    :type nb_u: int
    :param nb_v: The number of control points in the v direction.
    :type nb_v: int
    :param u_multiplicities: A list of multiplicities for the knots in the u direction.
        The multiplicity of a knot is the number of times it appears in the knot vector.
    :type u_multiplicities: List[int]
    :param v_multiplicities: A list of multiplicities for the knots in the v direction.
        The multiplicity of a knot is the number of times it appears in the knot vector.
    :type v_multiplicities: List[int]
    :param u_knots: A list of knots in the u direction. The knots are real numbers that
        define the position of the control points along the u direction.
    :type u_knots: List[float]
    :param v_knots: A list of knots in the v direction. The knots are real numbers that
        define the position of the control points along the v direction.
    :type v_knots: List[float]
    :param weights: (optional) A list of weights for the control points. The weights
        can be used to adjust the influence of each control point on the shape of the
        surface. Default is None.
    :type weights: List[float]
    :param name: (optional) A name for the surface. Default is an empty string.
    :type name: str
    """
    face_class = "BSplineFace3D"
    _non_serializable_attributes = ["surface", "curves"]

    def __init__(self, degree_u: int, degree_v: int, control_points: List[volmdlr.Point3D], nb_u: int, nb_v: int,
                 u_multiplicities: List[int], v_multiplicities: List[int], u_knots: List[float], v_knots: List[float],
                 weights: List[float] = None, name: str = ''):
        self.control_points = control_points
        self.degree_u = degree_u
        self.degree_v = degree_v
        self.nb_u = nb_u
        self.nb_v = nb_v

        u_knots = edges.standardize_knot_vector(u_knots)
        v_knots = edges.standardize_knot_vector(v_knots)
        self.u_knots = u_knots
        self.v_knots = v_knots
        self.u_multiplicities = u_multiplicities
        self.v_multiplicities = v_multiplicities
        self.weights = weights

        self.control_points_table = []
        points_row = []
        i = 1
        for point in control_points:
            points_row.append(point)
            if i == nb_v:
                self.control_points_table.append(points_row)
                points_row = []
                i = 1
            else:
                i += 1
        if weights is None:
            surface = BSpline.Surface()
            points = [(control_points[i][0], control_points[i][1],
                       control_points[i][2]) for i in range(len(control_points))]

        else:
            surface = NURBS.Surface()
            points = [(control_points[i][0] * weights[i], control_points[i][1] * weights[i],
                       control_points[i][2] * weights[i], weights[i]) for i in range(len(control_points))]
        surface.degree_u = degree_u
        surface.degree_v = degree_v
        surface.set_ctrlpts(points, nb_u, nb_v)
        knot_vector_u = []
        for i, u_knot in enumerate(u_knots):
            knot_vector_u.extend([u_knot] * u_multiplicities[i])
        knot_vector_v = []
        for i, v_knot in enumerate(v_knots):
            knot_vector_v.extend([v_knot] * v_multiplicities[i])
        surface.knotvector_u = knot_vector_u
        surface.knotvector_v = knot_vector_v
        surface.delta = 0.05

        self.surface = surface
        self.curves = extract_curves(surface, extract_u=True, extract_v=True)
        Surface3D.__init__(self, name=name)

        # Hidden Attributes
        self._displacements = None
        self._grids2d = None
        self._grids2d_deformed = None
        self._bbox = None

        self._x_periodicity = False  # Use False instead of None because None is a possible value of x_periodicity
        self._y_periodicity = False

    @property
    def x_periodicity(self):
        """
        Evaluates the periodicity of the surface in u direction.
        """
        if self._x_periodicity is False:
            u = self.curves['u']
            a, b = self.surface.domain[0]
            u0 = u[0]
            point_at_a = u0.evaluate_single(a)
            point_at_b = u0.evaluate_single(b)
            if npy.linalg.norm(npy.array(point_at_b) - npy.array(point_at_a)) < 1e-6:
                self._x_periodicity = self.surface.range[0]
            else:
                self._x_periodicity = None
        return self._x_periodicity

    @property
    def y_periodicity(self):
        """
        Evaluates the periodicity of the surface in v direction.
        """
        if self._y_periodicity is False:
            v = self.curves['v']
            c, d = self.surface.domain[1]
            v0 = v[0]
            point_at_c = v0.evaluate_single(c)
            point_at_d = v0.evaluate_single(d)
            if npy.linalg.norm(npy.array(point_at_d) - npy.array(point_at_c)) < 1e-6:
                self._y_periodicity = self.surface.range[1]
            else:
                self._y_periodicity = None
        return self._y_periodicity

    @property
    def bounding_box(self):
        if not self._bbox:
            self._bbox = self._bounding_box()
        return self._bbox

    def _bounding_box(self):
        """
        Computes the bounding box of the surface.

        """
        min_bounds, max_bounds = self.surface.bbox
        xmin, ymin, zmin = min_bounds
        xmax, ymax, zmax = max_bounds
        return volmdlr.core.BoundingBox(xmin, xmax, ymin, ymax, zmin, zmax)

    def control_points_matrix(self, coordinates):
        """
        Define control points like a matrix, for each coordinate: x:0, y:1, z:2.

        """

        points = npy.empty((self.nb_u, self.nb_v))
        for i in range(0, self.nb_u):
            for j in range(0, self.nb_v):
                points[i][j] = self.control_points_table[i][j][coordinates]
        return points

    # Knots_vector
    def knots_vector_u(self):
        """
        Compute the global knot vector (u direction) based on knot elements and multiplicities.

        """

        knots = self.u_knots
        multiplicities = self.u_multiplicities

        knots_vec = []
        for i, knot in enumerate(knots):
            for _ in range(0, multiplicities[i]):
                knots_vec.append(knot)
        return knots_vec

    def knots_vector_v(self):
        """
        Compute the global knot vector (v direction) based on knot elements and multiplicities.

        """

        knots = self.v_knots
        multiplicities = self.v_multiplicities

        knots_vec = []
        for i, knot in enumerate(knots):
            for _ in range(0, multiplicities[i]):
                knots_vec.append(knot)
        return knots_vec

    def basis_functions_u(self, u, k, i):
        """
        Compute basis functions Bi in u direction for u=u and degree=k.

        """

        # k = self.degree_u
        knots_vector_u = self.knots_vector_u()

        if k == 0:
            return 1.0 if knots_vector_u[i] <= u < knots_vector_u[i + 1] else 0.0
        if knots_vector_u[i + k] == knots_vector_u[i]:
            param_c1 = 0.0
        else:
            param_c1 = (u -knots_vector_u[i]) / (knots_vector_u[i + k] - knots_vector_u[i])\
                       * self.basis_functions_u(u, k - 1, i)
        if knots_vector_u[i + k + 1] == knots_vector_u[i + 1]:
            param_c2 = 0.0
        else:
            param_c2 = (knots_vector_u[i + k + 1] - u) / (knots_vector_u[i + k + 1] - knots_vector_u[i + 1]) *\
                       self.basis_functions_u(u, k - 1, i + 1)
        return param_c1 + param_c2

    def basis_functions_v(self, v, k, i):
        """
        Compute basis functions Bi in v direction for v=v and degree=k.

        """

        # k = self.degree_u
        knots = self.knots_vector_v()

        if k == 0:
            return 1.0 if knots[i] <= v < knots[i + 1] else 0.0
        if knots[i + k] == knots[i]:
            param_c1 = 0.0
        else:
            param_c1 = (v - knots[i]) / (knots[i + k] - knots[i]) * self.basis_functions_v(v, k - 1, i)
        if knots[i + k + 1] == knots[i + 1]:
            param_c2 = 0.0
        else:
            param_c2 = (knots[i + k + 1] - v) / (knots[i + k + 1] - knots[i + 1]) * self.basis_functions_v(v, k - 1,
                                                                                                           i + 1)
        return param_c1 + param_c2

    def blending_vector_u(self, u):
        """
        Compute a vector of basis_functions in u direction for u=u.
        """

        blending_vect = npy.empty((1, self.nb_u))
        for j in range(0, self.nb_u):
            blending_vect[0][j] = self.basis_functions_u(u, self.degree_u, j)

        return blending_vect

    def blending_vector_v(self, v):
        """
        Compute a vector of basis_functions in v direction for v=v.

        """

        blending_vect = npy.empty((1, self.nb_v))
        for j in range(0, self.nb_v):
            blending_vect[0][j] = self.basis_functions_v(v, self.degree_v, j)

        return blending_vect

    def blending_matrix_u(self, u):
        """
        Compute a matrix of basis_functions in u direction for a vector u like [0,1].

        """

        blending_mat = npy.empty((len(u), self.nb_u))
        for i, u_i in enumerate(u):
            for j in range(self.nb_u):
                blending_mat[i][j] = self.basis_functions_u(u_i, self.degree_u, j)
        return blending_mat

    def blending_matrix_v(self, v):
        """
        Compute a matrix of basis_functions in v direction for a vector v like [0,1].

        """

        blending_mat = npy.empty((len(v), self.nb_v))
        for i, v_i in enumerate(v):
            for j in range(self.nb_v):
                blending_mat[i][j] = self.basis_functions_v(v_i, self.degree_v, j)
        return blending_mat

    def point2d_to_3d(self, point2d: volmdlr.Point2D):
        u, v = point2d
        u = min(max(u, 0), 1)
        v = min(max(v, 0), 1)
        return volmdlr.Point3D(*evaluate_single((u, v), self.surface.data,
                                                self.surface.rational,
                                                self.surface.evaluator._span_func))
        # uses derivatives for performance because it's already compiled
        # return volmdlr.Point3D(*self.derivatives(u, v, 0)[0][0])
        # return volmdlr.Point3D(*self.surface.evaluate_single((x, y)))

    def point3d_to_2d(self, point3d: volmdlr.Point3D, tol=1e-5):
        """
        Evaluates the parametric coordinates (u, v) of a 3D point (x, y, z).

        :param point3d: A 3D point to be evaluated.
        :type point3d: :class:`volmdlr.Point3D`
        :param tol: Tolerance to accept the results.
        :type tol: float
        :return: The parametric coordinates (u, v) of the point.
        :rtype: :class:`volmdlr.Point2D`
        """

        def f(x):
            return point3d.point_distance(self.point2d_to_3d(volmdlr.Point2D(x[0], x[1])))

        def fun(x):
            derivatives = self.derivatives(x[0], x[1], 1)
            r = derivatives[0][0] - point3d
            f_value = r.norm() + 1e-32
            jacobian = npy.array([r.dot(derivatives[1][0]) / f_value, r.dot(derivatives[0][1]) / f_value])
            return f_value, jacobian

        min_bound_x, max_bound_x = self.surface.domain[0]
        min_bound_y, max_bound_y = self.surface.domain[1]

        delta_bound_x = max_bound_x - min_bound_x
        delta_bound_y = max_bound_y - min_bound_y
        x0s = [((min_bound_x + max_bound_x) / 2, (min_bound_y + max_bound_y) / 2),
               ((min_bound_x + max_bound_x) / 2, min_bound_y + delta_bound_y / 10),
               ((min_bound_x + max_bound_x) / 2, max_bound_y - delta_bound_y / 10),
               ((min_bound_x + max_bound_x) / 4, min_bound_y + delta_bound_y / 10),
               (max_bound_x - delta_bound_x / 4, min_bound_y + delta_bound_y / 10),
               ((min_bound_x + max_bound_x) / 4, max_bound_y - delta_bound_y / 10),
               (max_bound_x - delta_bound_x / 4, max_bound_y - delta_bound_y / 10),
               (min_bound_x + delta_bound_x / 10, min_bound_y + delta_bound_y / 10),
               (min_bound_x + delta_bound_x / 10, max_bound_y - delta_bound_y / 10),
               (max_bound_x - delta_bound_x / 10, min_bound_y + delta_bound_y / 10),
               (max_bound_x - delta_bound_x / 10, max_bound_y - delta_bound_y / 10)]

        # Sort the initial conditions
        x0s.sort(key=f)

        # Find the parametric coordinates of the point
        results = []
        for x0 in x0s:
            res = minimize(fun, x0=npy.array(x0), jac=True,
                           bounds=[(min_bound_x, max_bound_x),
                                   (min_bound_y, max_bound_y)])
            if res.fun <= tol:
                return volmdlr.Point2D(*res.x)

            results.append((res.x, res.fun))

        return volmdlr.Point2D(*min(results, key=lambda r: r[1])[0])

    def linesegment2d_to_3d(self, linesegment2d):
        # TODO: this is a non exact method!
        lth = linesegment2d.length()
        points = [self.point2d_to_3d(
            linesegment2d.point_at_abscissa(i * lth / 20.)) for i in range(21)]
        if points[0].is_close(points[-1]):
            return None
        linesegment = edges.LineSegment3D(points[0], points[-1])
        flag_arc = False
        flag = all(linesegment.point_belongs(point, abs_tol=1e-4) for point in points)
        if not flag:
            interior = self.point2d_to_3d(linesegment2d.point_at_abscissa(0.5 * lth))
            arc = edges.Arc3D(points[0], interior, points[-1])
            flag_arc = all(arc.point_belongs(point, abs_tol=1e-4) for point in points)

        periodic = False
        if self.x_periodicity is not None and \
                math.isclose(lth, self.x_periodicity, abs_tol=1e-6) and \
                math.isclose(linesegment2d.start.y, linesegment2d.end.y,
                             abs_tol=1e-6):
            periodic = True
        elif self.y_periodicity is not None and \
                math.isclose(lth, self.y_periodicity, abs_tol=1e-6) and \
                math.isclose(linesegment2d.start.x, linesegment2d.end.x,
                             abs_tol=1e-6):
            periodic = True

        if flag and not flag_arc:
            # All the points are on the same LineSegment3D
            primitives = [linesegment]
        elif flag_arc:
            primitives = [arc]
        else:
            primitives = [edges.BSplineCurve3D.from_points_interpolation(
                points, min(self.degree_u, self.degree_v), periodic=periodic)]
        return primitives

    def linesegment3d_to_2d(self, linesegment3d):
        """
        A line segment on a BSplineSurface3D will be in any case a line in 2D?.

        """
        start = self.point3d_to_2d(linesegment3d.start)
        end = self.point3d_to_2d(linesegment3d.end)
        if self.x_periodicity:
            if start.x != end.x:
                end = volmdlr.Point2D(start.x, end.y)
            if not start.is_close(end):
                return [edges.LineSegment2D(start, end)]
            return None
        if self.y_periodicity:
            if start.y != end.y:
                end = volmdlr.Point2D(end.x, start.y)
            if not start.is_close(end):
                return [edges.LineSegment2D(start, end)]
            return None
        if start.is_close(end):
            return None
        return [edges.LineSegment2D(start, end)]

    def _repair_periodic_boundary_points(self, curve3d, points_2d, direction_periodicity):
        """
        Verifies points at boundary on a periodic BSplineSurface3D.

        :param points_2d: List of `volmdlr.Point2D` after transformation from 3D Cartesian coordinates
        :type points_2d: List[volmdlr.Point2D]
        :param direction_periodicity: should be 'x' if x_periodicity or 'y' if y periodicity
        :type direction_periodicity: str
        """
        lth = curve3d.length()
        start = points_2d[0]
        end = points_2d[-1]
        points = points_2d
        pt_after_start = self.point3d_to_2d(curve3d.point_at_abscissa(0.1 * lth))
        pt_before_end = self.point3d_to_2d(curve3d.point_at_abscissa(0.9 * lth))
        # pt_after_start = points[1]
        # pt_before_end = points[-2]

        if direction_periodicity == 'x':
            i = 0
        else:
            i = 1
        min_bound, max_bound = self.surface.domain[i]
        delta = max_bound + min_bound

        if math.isclose(start[i], min_bound, abs_tol=1e-4) and pt_after_start[i] > 0.5 * delta:
            start[i] = max_bound
        elif math.isclose(start[i], max_bound, abs_tol=1e-4) and pt_after_start[i] < 0.5 * delta:
            start[i] = min_bound

        if math.isclose(end[i], min_bound, abs_tol=1e-4) and pt_before_end[i] > 0.5 * delta:
            end[i] = max_bound
        elif math.isclose(end[i], max_bound, abs_tol=1e-4) and pt_before_end[i] < 0.5 * delta:
            end[i] = min_bound

        points[0] = start
        points[-1] = end

        boundary = [(math.isclose(p[i], max_bound, abs_tol=1e-4) or math.isclose(p[i], min_bound, abs_tol=1e-4)) for
                    p in points]
        if all(boundary):
            # if the line is at the boundary of the surface domain, we take the first point as reference
            t_param = max_bound if math.isclose(points[0][i], max_bound, abs_tol=1e-4) else min_bound
            if direction_periodicity == 'x':
                points = [volmdlr.Point2D(t_param, p[1]) for p in points]
            else:
                points = [volmdlr.Point2D(p[0], t_param) for p in points]

        return points

    def bsplinecurve3d_to_2d(self, bspline_curve3d):
        """
        Converts the primitive from 3D spatial coordinates to its equivalent 2D primitive in the parametric space.
        """
        # TODO: enhance this, it is a non exact method!
        # TODO: bsplinecurve can be periodic but not around the bsplinesurface
        flag = False
        if not bspline_curve3d.points[0].is_close(bspline_curve3d.points[-1]):
            bsc_linesegment = edges.LineSegment3D(bspline_curve3d.points[0],
                                                bspline_curve3d.points[-1])
            flag = True
            for point in bspline_curve3d.points:
                if not bsc_linesegment.point_belongs(point):
                    flag = False
                    break

        if self.x_periodicity and not self.y_periodicity \
                and bspline_curve3d.periodic:
            point1 = self.point3d_to_2d(bspline_curve3d.points[0])
            p1_sup = self.point3d_to_2d(bspline_curve3d.points[0])
            new_x = point1.x - p1_sup.x + self.x_periodicity
            new_x = new_x if 0 <= new_x else 0
            reverse = False
            if new_x < 0:
                new_x = 0
            elif math.isclose(new_x, self.x_periodicity, abs_tol=1e-5):
                new_x = 0
                reverse = True

            linesegments = [
                edges.LineSegment2D(
                    volmdlr.Point2D(new_x, point1.y),
                    volmdlr.Point2D(self.x_periodicity, point1.y))]
            if reverse:
                linesegments[0] = linesegments[0].reverse()

        elif self.y_periodicity and not self.x_periodicity \
                and bspline_curve3d.periodic:
            point1 = self.point3d_to_2d(bspline_curve3d.points[0])
            p1_sup = self.point3d_to_2d(bspline_curve3d.points[0])
            new_y = point1.y - p1_sup.y + self.y_periodicity
            new_y = new_y if 0 <= new_y else 0
            reverse = False
            if new_y < 0:
                new_y = 0
            elif math.isclose(new_y, self.y_periodicity, abs_tol=1e-5):
                new_y = 0
                reverse = True

            linesegments = [
                edges.LineSegment2D(
                    volmdlr.Point2D(point1.x, new_y),
                    volmdlr.Point2D(point1.x, self.y_periodicity))]
            if reverse:
                linesegments[0] = linesegments[0].reverse()

        elif self.x_periodicity and self.y_periodicity \
                and bspline_curve3d.periodic:
            raise NotImplementedError

        if flag:
            x_perio = self.x_periodicity if self.x_periodicity is not None \
                else 1.
            y_perio = self.y_periodicity if self.y_periodicity is not None \
                else 1.

            point1 = self.point3d_to_2d(bspline_curve3d.points[0])
            point2 = self.point3d_to_2d(bspline_curve3d.points[-1])

            if point1.is_close(point2):
                print('BSplineCruve3D skipped because it is too small')
                linesegments = None
            else:
                p1_sup = self.point3d_to_2d(bspline_curve3d.points[0])
                p2_sup = self.point3d_to_2d(bspline_curve3d.points[-1])
                if self.x_periodicity and point1.point_distance(p1_sup) > 1e-5:
                    point1.x -= p1_sup.x - x_perio
                    point2.x -= p2_sup.x - x_perio
                if self.y_periodicity and point1.point_distance(p1_sup) > 1e-5:
                    point1.y -= p1_sup.y - y_perio
                    point2.y -= p2_sup.y - y_perio
                linesegments = [edges.LineSegment2D(point1, point2)]
            # How to check if end of surface overlaps start or the opposite ?
        else:
            lth = bspline_curve3d.length()
            if lth > 1e-5:
                n = len(bspline_curve3d.control_points)
                points = [self.point3d_to_2d(p) for p in bspline_curve3d.discretization_points(number_points=n)]

                if self.x_periodicity:
                    points = self._repair_periodic_boundary_points(bspline_curve3d, points, 'x')
                    if bspline_curve3d.periodic and points[0].is_close(points[-1]):
                        u_min, u_max = bspline_curve3d.curve.domain
                        if math.isclose(points[0].x, u_min, abs_tol=1e-6):
                            should_be_umax = (u_max - points[1].x) < (points[1].x - u_min)
                            if should_be_umax:
                                points[0] = volmdlr.Point2D(u_max, points[0].y)
                            else:
                                points[-1] = volmdlr.Point2D(u_max, points[-1].y)
                        elif math.isclose(points[0].x, u_max, abs_tol=1e-6):
                            should_be_umin = (u_max - points[1].x) > (points[1].x - u_min)
                            if should_be_umin:
                                points[0] = volmdlr.Point2D(u_min, points[0].y)
                            else:
                                points[-1] = volmdlr.Point2D(u_min, points[-1].y)
                if self.y_periodicity:
                    points = self._repair_periodic_boundary_points(bspline_curve3d, points, 'y')
                    if bspline_curve3d.periodic and points[0].is_close(points[-1]):
                        u_min, u_max = bspline_curve3d.curve.domain
                        if math.isclose(points[0].y, u_min, abs_tol=1e-6):
                            should_be_umax = (u_max - points[1].y) < (points[1].y - u_min)
                            if should_be_umax:
                                points[0] = volmdlr.Point2D(points[0].x, u_max)
                            else:
                                points[-1] = volmdlr.Point2D(points[-1].x, u_max)
                        elif math.isclose(points[0].y, u_max, abs_tol=1e-6):
                            should_be_umin = (u_max - points[1].y) > (points[1].y - u_min)
                            if should_be_umin:
                                points[0] = volmdlr.Point2D(points[0].x, u_min)
                            else:
                                points[-1] = volmdlr.Point2D(points[-1].x, u_min)

                if not points[0].is_close(points[-1]) and not bspline_curve3d.periodic:
                    linesegment = edges.LineSegment2D(points[0], points[-1])
                    flag_line = True
                    for point in points:
                        if not linesegment.point_belongs(point, abs_tol=1e-4):
                            flag_line = False
                            break
                    if flag_line:
                        return [linesegment]

                if self.x_periodicity:
                    points = self._repair_periodic_boundary_points(bspline_curve3d, points, 'x')

                if self.y_periodicity:
                    points = self._repair_periodic_boundary_points(bspline_curve3d, points, 'y')

                return [edges.BSplineCurve2D.from_points_interpolation(
                    points=points, degree=bspline_curve3d.degree, periodic=bspline_curve3d.periodic)]

            if 1e-6 < lth <= 1e-5:
                linesegments = [edges.LineSegment2D(
                    self.point3d_to_2d(bspline_curve3d.start),
                    self.point3d_to_2d(bspline_curve3d.end))]
            else:
                print('BSplineCruve3D skipped because it is too small')
                linesegments = None

        return linesegments

    def bsplinecurve2d_to_3d(self, bspline_curve2d):
        """
        Converts the parametric boundary representation into a 3D primitive.
        """
        if bspline_curve2d.name == "parametric.arc":
            start = self.point2d_to_3d(bspline_curve2d.start)
            interior = self.point2d_to_3d(bspline_curve2d.evaluate_single(0.5))
            end = self.point2d_to_3d(bspline_curve2d.end)
            return [edges.Arc3D(start, interior, end)]

        number_points = len(bspline_curve2d.control_points)
        points = [self.point2d_to_3d(point)
                  for point in bspline_curve2d.discretization_points(number_points=number_points)]
        return [edges.BSplineCurve3D.from_points_interpolation(
            points, bspline_curve2d.degree, bspline_curve2d.periodic)]

    def arc3d_to_2d(self, arc3d):
        """
        Converts the primitive from 3D spatial coordinates to its equivalent 2D primitive in the parametric space.
        """
        number_points = max(self.nb_u, self.nb_v)
        degree = max(self.degree_u, self.degree_v)
        points = [self.point3d_to_2d(point3d) for point3d in arc3d.discretization_points(number_points=number_points)]
        start = points[0]
        end = points[-1]
        min_bound_x, max_bound_x = self.surface.domain[0]
        min_bound_y, max_bound_y = self.surface.domain[1]
        if self.x_periodicity:
            points = self._repair_periodic_boundary_points(arc3d, points, 'x')
            start = points[0]
            end = points[-1]
            if start.is_close(end):
                if math.isclose(start.x, min_bound_x, abs_tol=1e-4):
                    end.x = max_bound_x
                else:
                    end.x = min_bound_x
        if self.y_periodicity:
            points = self._repair_periodic_boundary_points(arc3d, points, 'y')
            start = points[0]
            end = points[-1]
            if start.is_close(end):
                if math.isclose(start.y, min_bound_y, abs_tol=1e-4):
                    end.y = max_bound_y
                else:
                    end.y = min_bound_y
        if start.is_close(end):
            return []
        linesegment = edges.LineSegment2D(start, end, name="parametric.arc")
        flag = True
        for point in points:
            if not linesegment.point_belongs(point):
                flag = False
                break
        if flag:
            return [linesegment]
        return [edges.BSplineCurve2D.from_points_interpolation(points, degree, name="parametric.arc")]

    def arc2d_to_3d(self, arc2d):
        number_points = math.ceil(arc2d.angle * 7) + 1  # 7 points per radian
        length = arc2d.length()
        points = [self.point2d_to_3d(arc2d.point_at_abscissa(i * length / (number_points - 1)))
                  for i in range(number_points)]
        return [edges.BSplineCurve3D.from_points_interpolation(
            points, max(self.degree_u, self.degree_v))]

    def rectangular_cut(self, u1: float, u2: float,
                        v1: float, v2: float, name: str = ''):
        """Deprecated method, Use BSplineFace3D from_surface_rectangular_cut method."""
        raise AttributeError("BSplineSurface3D.rectangular_cut is deprecated."
                             " Use the class_method from_surface_rectangular_cut in BSplineFace3D instead")

    def rotation(self, center: volmdlr.Vector3D,
                 axis: volmdlr.Vector3D, angle: float):
        """
        BSplineSurface3D rotation.

        :param center: rotation center
        :param axis: rotation axis
        :param angle: angle rotation
        :return: a new rotated BSplineSurface3D
        """
        new_control_points = [p.rotation(center, axis, angle)
                              for p in self.control_points]
        new_bsplinesurface3d = BSplineSurface3D(self.degree_u, self.degree_v,
                                                new_control_points, self.nb_u,
                                                self.nb_v,
                                                self.u_multiplicities,
                                                self.v_multiplicities,
                                                self.u_knots, self.v_knots,
                                                self.weights, self.name)
        return new_bsplinesurface3d

    def rotation_inplace(self, center: volmdlr.Vector3D,
                         axis: volmdlr.Vector3D, angle: float):
        """
        BSplineSurface3D rotation. Object is updated in-place.

        :param center: rotation center.
        :type center: `volmdlr.Vector3D`
        :param axis: rotation axis.
        :type axis: `volmdlr.Vector3D`
        :param angle: rotation angle.
        :type angle: float
        :return: None, BSplineSurface3D is updated in-place
        :rtype: None
        """
        warnings.warn("'in-place' methods are deprecated. Use a not in-place method instead.", DeprecationWarning)

        new_bsplinesurface3d = self.rotation(center, axis, angle)
        self.control_points = new_bsplinesurface3d.control_points
        self.surface = new_bsplinesurface3d.surface

    def translation(self, offset: volmdlr.Vector3D):
        """
        BSplineSurface3D translation.

        :param offset: translation vector
        :return: A new translated BSplineSurface3D
        """
        new_control_points = [p.translation(offset) for p in
                              self.control_points]
        new_bsplinesurface3d = BSplineSurface3D(self.degree_u, self.degree_v,
                                                new_control_points, self.nb_u,
                                                self.nb_v,
                                                self.u_multiplicities,
                                                self.v_multiplicities,
                                                self.u_knots, self.v_knots,
                                                self.weights, self.name)

        return new_bsplinesurface3d

    def translation_inplace(self, offset: volmdlr.Vector3D):
        """
        BSplineSurface3D translation. Object is updated in-place.

        :param offset: translation vector
        """
        warnings.warn("'in-place' methods are deprecated. Use a not in-place method instead.", DeprecationWarning)

        new_bsplinesurface3d = self.translation(offset)
        self.control_points = new_bsplinesurface3d.control_points
        self.surface = new_bsplinesurface3d.surface

    def frame_mapping(self, frame: volmdlr.Frame3D, side: str):
        """
        Changes frame_mapping and return a new BSplineSurface3D.

        side = 'old' or 'new'
        """
        new_control_points = [p.frame_mapping(frame, side) for p in
                              self.control_points]
        new_bsplinesurface3d = BSplineSurface3D(self.degree_u, self.degree_v,
                                                new_control_points, self.nb_u,
                                                self.nb_v,
                                                self.u_multiplicities,
                                                self.v_multiplicities,
                                                self.u_knots, self.v_knots,
                                                self.weights, self.name)
        return new_bsplinesurface3d

    def frame_mapping_inplace(self, frame: volmdlr.Frame3D, side: str):
        """
        Changes frame_mapping and the object is updated in-place.

        side = 'old' or 'new'
        """
        warnings.warn("'in-place' methods are deprecated. Use a not in-place method instead.", DeprecationWarning)

        new_bsplinesurface3d = self.frame_mapping(frame, side)
        self.control_points = new_bsplinesurface3d.control_points
        self.surface = new_bsplinesurface3d.surface

    def plot(self, ax=None, color='grey', alpha=0.5):
        u_curves = [edges.BSplineCurve3D.from_geomdl_curve(u) for u in self.curves['u']]
        v_curves = [edges.BSplineCurve3D.from_geomdl_curve(v) for v in self.curves['v']]
        if ax is None:
            ax = plt.figure().add_subplot(111, projection='3d')
        for u in u_curves:
            u.plot(ax=ax, edge_style=EdgeStyle(color=color, alpha=alpha))
        for v in v_curves:
            v.plot(ax=ax, edge_style=EdgeStyle(color=color, alpha=alpha))
        for point in self.control_points:
            point.plot(ax, color=color, alpha=alpha)
        return ax

    def simplify_surface(self):
        """
        Verifies if BSplineSurface3D could be a Plane3D.

        :return: A planar surface if possible, otherwise, returns self.
        """
        points = [self.control_points[0]]
        vector_list = []
        for point in self.control_points[1:]:
            vector = point - points[0]
            is_colinear = any(vector.is_colinear_to(other_vector) for other_vector in vector_list)
            if not point_in_list(point, points) and not is_colinear:
                points.append(point)
                vector_list.append(vector)
                if len(points) == 3:
                    plane3d = Plane3D.from_3_points(*points)
                    if all(plane3d.point_on_surface(point) for point in self.control_points):
                        return plane3d
                    break
        return self

    @classmethod
    def from_step(cls, arguments, object_dict, **kwargs):
        """
        Converts a step primitive to a BSplineSurface3D.

        :param arguments: The arguments of the step primitive.
        :type arguments: list
        :param object_dict: The dictionary containing all the step primitives
            that have already been instantiated.
        :type object_dict: dict
        :return: The corresponding BSplineSurface3D object.
        :rtype: :class:`volmdlr.faces.BSplineSurface3D`
        """
        name = arguments[0][1:-1]
        degree_u = int(arguments[1])
        degree_v = int(arguments[2])
        points_sets = arguments[3][1:-1].split("),")
        points_sets = [elem + ")" for elem in points_sets[:-1]] + [
            points_sets[-1]]
        control_points = []
        for points_set in points_sets:
            points = [object_dict[int(i[1:])] for i in
                      points_set[1:-1].split(",")]
            nb_v = len(points)
            control_points.extend(points)
        nb_u = int(len(control_points) / nb_v)
        surface_form = arguments[4]
        if arguments[5] == '.F.':
            u_closed = False
        elif arguments[5] == '.T.':
            u_closed = True
        else:
            raise ValueError
        if arguments[6] == '.F.':
            v_closed = False
        elif arguments[6] == '.T.':
            v_closed = True
        else:
            raise ValueError
        self_intersect = arguments[7]
        u_multiplicities = [int(i) for i in arguments[8][1:-1].split(",")]
        v_multiplicities = [int(i) for i in arguments[9][1:-1].split(",")]
        u_knots = [float(i) for i in arguments[10][1:-1].split(",")]
        v_knots = [float(i) for i in arguments[11][1:-1].split(",")]
        knot_spec = arguments[12]

        if 13 in range(len(arguments)):
            weight_data = [
                float(i) for i in
                arguments[13][1:-1].replace("(", "").replace(")", "").split(",")
            ]
        else:
            weight_data = None

        bsplinesurface = cls(degree_u, degree_v, control_points, nb_u, nb_v,
                             u_multiplicities, v_multiplicities, u_knots,
                             v_knots, weight_data, name)
        if not bsplinesurface.x_periodicity and not bsplinesurface.y_periodicity:
            bsplinesurface = bsplinesurface.simplify_surface()
        # if u_closed:
        #     bsplinesurface.x_periodicity = bsplinesurface.get_x_periodicity()
        # if v_closed:
        #     bsplinesurface.y_periodicity = bsplinesurface.get_y_periodicity()
        return bsplinesurface

    def to_step(self, current_id):
        content = ''
        point_matrix_ids = '('
        for points in self.control_points_table:
            point_ids = '('
            for point in points:
                point_content, point_id = point.to_step(current_id)
                content += point_content
                point_ids += f'#{point_id},'
                current_id = point_id + 1
            point_ids = point_ids[:-1]
            point_ids += '),'
            point_matrix_ids += point_ids
        point_matrix_ids = point_matrix_ids[:-1]
        point_matrix_ids += ')'

        u_close = '.T.' if self.x_periodicity else '.F.'
        v_close = '.T.' if self.y_periodicity else '.F.'

        content += f"#{current_id} = B_SPLINE_SURFACE_WITH_KNOTS('{self.name}',{self.degree_u},{self.degree_v}," \
                   f"{point_matrix_ids},.UNSPECIFIED.,{u_close},{v_close},.F.,{tuple(self.u_multiplicities)}," \
                   f"{tuple(self.v_multiplicities)},{tuple(self.u_knots)},{tuple(self.v_knots)},.UNSPECIFIED.);\n"
        return content, [current_id]

    def grid3d(self, grid2d: grid.Grid2D):
        """
        Generate 3d grid points of a Bspline surface, based on a Grid2D.

        """

        if not self._grids2d:
            self._grids2d = grid2d

        points_2d = grid2d.points
        points_3d = [self.point2d_to_3d(point2d) for point2d in points_2d]

        return points_3d

    def grid2d_deformed(self, grid2d: grid.Grid2D):
        """
        Dimension and deform a Grid2D points based on a Bspline surface.

        """

        points_2d = grid2d.points
        points_3d = self.grid3d(grid2d)

        points_x, points_y = grid2d.points_xy

        # Parameters
        index_x = {}  # grid point position(i,j), x coordinates position in X(unknown variable)
        index_y = {}  # grid point position(i,j), y coordinates position in X(unknown variable)
        index_points = {}  # grid point position(j,i), point position in points_2d (or points_3d)
        k_index, p_index = 0, 0
        for i in range(0, points_x):
            for j in range(0, points_y):
                index_x.update({(j, i): k_index})
                index_y.update({(j, i): k_index + 1})
                index_points.update({(j, i): p_index})
                k_index = k_index + 2
                p_index = p_index + 1

        equation_points = []  # points combination to compute distances between 2D and 3D grid points
        for i in range(0, points_y):  # row from (0,i)
            for j in range(1, points_x):
                equation_points.append(((0, i), (j, i)))
        for i in range(0, points_x):  # column from (i,0)
            for j in range(1, points_y):
                equation_points.append(((i, 0), (i, j)))
        for i in range(0, points_y):  # row
            for j in range(0, points_x - 1):
                equation_points.append(((j, i), (j + 1, i)))
        for i in range(0, points_x):  # column
            for j in range(0, points_x - 1):
                equation_points.append(((i, j), (i, j + 1)))
        for i in range(0, points_y - 1):  # diagonal
            for j in range(0, points_x - 1):
                equation_points.append(((j, i), (j + 1, i + 1)))

        for i in range(0, points_y):  # row 2segments (before.point.after)
            for j in range(1, points_x - 1):
                equation_points.append(((j - 1, i), (j + 1, i)))

        for i in range(0, points_x):  # column 2segments (before.point.after)
            for j in range(1, points_y - 1):
                equation_points.append(((i, j - 1), (i, j + 1)))

        # geodesic distances between 3D grid points (based on points combination [equation_points])
        geodesic_distances = []
        for point in equation_points:
            geodesic_distances.append((self.geodesic_distance(
                points_3d[index_points[point[0]]], points_3d[index_points[point[1]]])) ** 2)

        # System of nonlinear equations
        def non_linear_equations(xparam):
            vector_f = npy.empty(len(equation_points) + 2)
            idx = 0
            for idx, point_ in enumerate(equation_points):
                vector_f[idx] = abs((xparam[index_x[point_[0]]] ** 2 +
                                     xparam[index_x[point_[1]]] ** 2 +
                                     xparam[index_y[point_[0]]] ** 2 +
                                     xparam[index_y[point_[1]]] ** 2 -
                                     2 *
                                     xparam[index_x[point_[0]]] *
                                     xparam[index_x[point_[1]]] -
                                     2 *
                                     xparam[index_y[point_[0]]] *
                                     xparam[index_y[point_[1]]] -
                                     geodesic_distances[idx]) /
                                    geodesic_distances[idx])

            vector_f[idx + 1] = xparam[0] * 1000
            vector_f[idx + 2] = xparam[1] * 1000

            return vector_f

        # Solution with "least_squares"
        x_init = []  # initial guess (2D grid points)
        for point in points_2d:
            x_init.append(point[0])
            x_init.append(point[1])
        z = least_squares(non_linear_equations, x_init)

        points_2d_deformed = [volmdlr.Point2D(z.x[i], z.x[i + 1])
                              for i in range(0, len(z.x), 2)]  # deformed 2d grid points

        grid2d_deformed = grid.Grid2D.from_points(points=points_2d_deformed,
                                                          points_dim_1=points_x,
                                                          direction=grid2d.direction)

        self._grids2d_deformed = grid2d_deformed

        return points_2d_deformed

    def grid2d_deformation(self, grid2d: grid.Grid2D):
        """
        Compute the deformation/displacement (dx/dy) of a Grid2D based on a Bspline surface.

        """

        if not self._grids2d_deformed:
            self.grid2d_deformed(grid2d)

        displacement = self._grids2d_deformed.displacement_compared_to(grid2d)
        self._displacements = displacement

        return displacement

    def point2d_parametric_to_dimension(self, point2d: volmdlr.Point3D, grid2d: grid.Grid2D):
        """
        Convert a point 2d from the parametric to the dimensioned frame.

        """

        # Check if the 0<point2d.x<1 and 0<point2d.y<1
        if point2d.x < 0:
            point2d.x = 0
        elif point2d.x > 1:
            point2d.x = 1
        if point2d.y < 0:
            point2d.y = 0
        elif point2d.y > 1:
            point2d.y = 1

        if self._grids2d == grid2d:
            points_2d = self._grids2d.points
        else:
            points_2d = grid2d.points
            self._grids2d = grid2d

        if self._displacements is not None:
            displacement = self._displacements
        else:
            displacement = self.grid2d_deformation(grid2d)

        points_x, points_y = grid2d.points_xy

        # Parameters
        index_points = {}  # grid point position(j,i), point position in points_2d (or points_3d)
        p_index = 0
        for i in range(0, points_x):
            for j in range(0, points_y):
                index_points.update({(j, i): p_index})
                p_index = p_index + 1

        # Form function "Finite Elements"
        def form_function(s_param, t_param):
            empty_n = npy.empty(4)
            empty_n[0] = (1 - s_param) * (1 - t_param) / 4
            empty_n[1] = (1 + s_param) * (1 - t_param) / 4
            empty_n[2] = (1 + s_param) * (1 + t_param) / 4
            empty_n[3] = (1 - s_param) * (1 + t_param) / 4
            return empty_n

        finite_elements_points = []  # 2D grid points index that define one element
        for j in range(0, points_y - 1):
            for i in range(0, points_x - 1):
                finite_elements_points.append(((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)))
        finite_elements = []  # finite elements defined with closed polygon
        for point in finite_elements_points:
            finite_elements.append(
                wires.ClosedPolygon2D((points_2d[index_points[point[0]]],
                                               points_2d[index_points[point[1]]],
                                               points_2d[index_points[point[2]]],
                                               points_2d[index_points[point[3]]])))
        k = 0
        for k, point in enumerate(finite_elements_points):
            if (wires.Contour2D(finite_elements[k].primitives).point_belongs(
                    point2d)  # finite_elements[k].point_belongs(point2d)
                    or wires.Contour2D(finite_elements[k].primitives).point_over_contour(point2d)
                    or ((points_2d[index_points[point[0]]][0] < point2d.x <
                         points_2d[index_points[point[1]]][0])
                        and point2d.y == points_2d[index_points[point[0]]][1])
                    or ((points_2d[index_points[point[1]]][1] < point2d.y <
                         points_2d[index_points[point[2]]][1])
                        and point2d.x == points_2d[index_points[point[1]]][0])
                    or ((points_2d[index_points[point[3]]][0] < point2d.x <
                         points_2d[index_points[point[2]]][0])
                        and point2d.y == points_2d[index_points[point[1]]][1])
                    or ((points_2d[index_points[point[0]]][1] < point2d.y <
                         points_2d[index_points[point[3]]][1])
                        and point2d.x == points_2d[index_points[point[0]]][0])):
                break

        x0 = points_2d[index_points[finite_elements_points[k][0]]][0]
        y0 = points_2d[index_points[finite_elements_points[k][0]]][1]
        x1 = points_2d[index_points[finite_elements_points[k][1]]][0]
        y2 = points_2d[index_points[finite_elements_points[k][2]]][1]
        x = point2d.x
        y = point2d.y
        s_param = 2 * ((x - x0) / (x1 - x0)) - 1
        t_param = 2 * ((y - y0) / (y2 - y0)) - 1

        n = form_function(s_param, t_param)
        dx = npy.array([displacement[index_points[finite_elements_points[k][0]]][0],
                        displacement[index_points[finite_elements_points[k][1]]][0],
                        displacement[index_points[finite_elements_points[k][2]]][0],
                        displacement[index_points[finite_elements_points[k][3]]][0]])
        dy = npy.array([displacement[index_points[finite_elements_points[k][0]]][1],
                        displacement[index_points[finite_elements_points[k][1]]][1],
                        displacement[index_points[finite_elements_points[k][2]]][1],
                        displacement[index_points[finite_elements_points[k][3]]][1]])

        return volmdlr.Point2D(point2d.x + npy.transpose(n).dot(dx), point2d.y + npy.transpose(n).dot(dy))

    def point3d_to_2d_with_dimension(self, point3d: volmdlr.Point3D, grid2d: grid.Grid2D):
        """
        Compute the point2d of a point3d, on a Bspline surface, in the dimensioned frame.
        """

        point2d = self.point3d_to_2d(point3d)

        point2d_with_dimension = self.point2d_parametric_to_dimension(point2d, grid2d)

        return point2d_with_dimension

    def point2d_with_dimension_to_parametric_frame(self, point2d, grid2d: grid.Grid2D):
        """
        Convert a point 2d from the dimensioned to the parametric frame.

        """

        if self._grids2d != grid2d:
            self._grids2d = grid2d
        if not self._grids2d_deformed:
            self.grid2d_deformed(grid2d)

        points_2d = grid2d.points
        points_2d_deformed = self._grids2d_deformed.points
        points_x, points_y = grid2d.points_xy

        # Parameters
        index_points = {}  # grid point position(j,i), point position in points_2d (or points_3d)
        p_index = 0
        for i in range(0, points_x):
            for j in range(0, points_y):
                index_points.update({(j, i): p_index})
                p_index = p_index + 1

        finite_elements_points = []  # 2D grid points index that define one element
        for j in range(0, points_y - 1):
            for i in range(0, points_x - 1):
                finite_elements_points.append(((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)))
        finite_elements = []  # finite elements defined with closed polygon  DEFORMED
        for point in finite_elements_points:
            finite_elements.append(
                wires.ClosedPolygon2D((points_2d_deformed[index_points[point[0]]],
                                               points_2d_deformed[index_points[point[1]]],
                                               points_2d_deformed[index_points[point[2]]],
                                               points_2d_deformed[index_points[point[3]]])))

        finite_elements_initial = []  # finite elements defined with closed polygon  INITIAL
        for point in finite_elements_points:
            finite_elements_initial.append(
                wires.ClosedPolygon2D((points_2d[index_points[point[0]]],
                                               points_2d[index_points[point[1]]],
                                               points_2d[index_points[point[2]]],
                                               points_2d[index_points[point[3]]])))
        k = 0
        for k, point in enumerate(finite_elements_points):
            if (finite_elements[k].point_belongs(point2d)
                    or ((points_2d_deformed[index_points[point[0]]][0] < point2d.x <
                         points_2d_deformed[index_points[point[1]]][0])
                        and point2d.y == points_2d_deformed[index_points[point[0]]][1])
                    or ((points_2d_deformed[index_points[finite_elements_points[k][1]]][1] < point2d.y <
                         points_2d_deformed[index_points[finite_elements_points[k][2]]][1])
                        and point2d.x == points_2d_deformed[index_points[point[1]]][0])
                    or ((points_2d_deformed[index_points[point[3]]][0] < point2d.x <
                         points_2d_deformed[index_points[point[2]]][0])
                        and point2d.y == points_2d_deformed[index_points[point[1]]][1])
                    or ((points_2d_deformed[index_points[point[0]]][1] < point2d.y <
                         points_2d_deformed[index_points[point[3]]][1])
                        and point2d.x == points_2d_deformed[index_points[point[0]]][0])
                    or finite_elements[k].primitives[0].point_belongs(point2d) or finite_elements[k].primitives[
                        1].point_belongs(point2d)
                    or finite_elements[k].primitives[2].point_belongs(point2d) or finite_elements[k].primitives[
                        3].point_belongs(point2d)):
                break

        frame_deformed = volmdlr.Frame2D(finite_elements[k].center_of_mass(),
                                         volmdlr.Vector2D(finite_elements[k].primitives[1].middle_point()[0] -
                                                          finite_elements[k].center_of_mass()[0],
                                                          finite_elements[k].primitives[1].middle_point()[1] -
                                                          finite_elements[k].center_of_mass()[1]),
                                         volmdlr.Vector2D(finite_elements[k].primitives[0].middle_point()[0] -
                                                          finite_elements[k].center_of_mass()[0],
                                                          finite_elements[k].primitives[0].middle_point()[1] -
                                                          finite_elements[k].center_of_mass()[1]))

        point2d_frame_deformed = volmdlr.Point2D(point2d.frame_mapping(frame_deformed, 'new')[0],
                                                 point2d.frame_mapping(frame_deformed, 'new')[1])

        frame_inital = volmdlr.Frame2D(finite_elements_initial[k].center_of_mass(),
                                       volmdlr.Vector2D(finite_elements_initial[k].primitives[1].middle_point()[0] -
                                                        finite_elements_initial[k].center_of_mass()[0],
                                                        finite_elements_initial[k].primitives[1].middle_point()[1] -
                                                        finite_elements_initial[k].center_of_mass()[1]),
                                       volmdlr.Vector2D(finite_elements_initial[k].primitives[0].middle_point()[0] -
                                                        finite_elements_initial[k].center_of_mass()[0],
                                                        finite_elements_initial[k].primitives[0].middle_point()[1] -
                                                        finite_elements_initial[k].center_of_mass()[1]))

        point2d = point2d_frame_deformed.frame_mapping(frame_inital, 'old')
        if point2d.x < 0:
            point2d.x = 0
        elif point2d.x > 1:
            point2d.x = 1
        if point2d.y < 0:
            point2d.y = 0
        elif point2d.y > 1:
            point2d.y = 1

        return point2d

    def point2d_with_dimension_to_3d(self, point2d, grid2d: grid.Grid2D):
        """
        Compute the point 3d, on a Bspline surface, of a point 2d define in the dimensioned frame.

        """

        point2d_01 = self.point2d_with_dimension_to_parametric_frame(point2d, grid2d)

        return self.point2d_to_3d(point2d_01)

    def linesegment2d_parametric_to_dimension(self, linesegment2d, grid2d: grid.Grid2D):
        """
        Convert a linesegment2d from the parametric to the dimensioned frame.

        """

        points = linesegment2d.discretization_points(number_points=20)
        points_dim = [
            self.point2d_parametric_to_dimension(
                point, grid2d) for point in points]

        return edges.BSplineCurve2D.from_points_interpolation(
            points_dim, max(self.degree_u, self.degree_v))

    def linesegment3d_to_2d_with_dimension(self, linesegment3d, grid2d: grid.Grid2D):
        """
        Compute the linesegment2d of a linesegment3d, on a Bspline surface, in the dimensioned frame.

        """

        linesegment2d = self.linesegment3d_to_2d(linesegment3d)
        bsplinecurve2d_with_dimension = self.linesegment2d_parametric_to_dimension(linesegment2d, grid2d)

        return bsplinecurve2d_with_dimension

    def linesegment2d_with_dimension_to_parametric_frame(self, linesegment2d):
        """
        Convert a linesegment2d from the dimensioned to the parametric frame.

        """

        try:
            linesegment2d = edges.LineSegment2D(
                self.point2d_with_dimension_to_parametric_frame(linesegment2d.start, self._grids2d),
                self.point2d_with_dimension_to_parametric_frame(linesegment2d.end, self._grids2d))
        except NotImplementedError:
            return None

        return linesegment2d

    def linesegment2d_with_dimension_to_3d(self, linesegment2d):
        """
        Compute the linesegment3d, on a Bspline surface, of a linesegment2d defined in the dimensioned frame.

        """

        linesegment2d_01 = self.linesegment2d_with_dimension_to_parametric_frame(linesegment2d)
        linesegment3d = self.linesegment2d_to_3d(linesegment2d_01)

        return linesegment3d

    def bsplinecurve2d_parametric_to_dimension(self, bsplinecurve2d, grid2d: grid.Grid2D):
        """
        Convert a bsplinecurve2d from the parametric to the dimensioned frame.

        """

        # check if bsplinecurve2d is in a list
        if isinstance(bsplinecurve2d, list):
            bsplinecurve2d = bsplinecurve2d[0]
        points = bsplinecurve2d.control_points
        points_dim = []

        for point in points:
            points_dim.append(self.point2d_parametric_to_dimension(point, grid2d))

        bsplinecurve2d_with_dimension = edges.BSplineCurve2D(bsplinecurve2d.degree, points_dim,
                                                           bsplinecurve2d.knot_multiplicities,
                                                           bsplinecurve2d.knots,
                                                           bsplinecurve2d.weights,
                                                           bsplinecurve2d.periodic)

        return bsplinecurve2d_with_dimension

    def bsplinecurve3d_to_2d_with_dimension(self, bsplinecurve3d, grid2d: grid.Grid2D):
        """
        Compute the bsplinecurve2d of a bsplinecurve3d, on a Bspline surface, in the dimensioned frame.

        """

        bsplinecurve2d_01 = self.bsplinecurve3d_to_2d(bsplinecurve3d)
        bsplinecurve2d_with_dimension = self.bsplinecurve2d_parametric_to_dimension(
            bsplinecurve2d_01, grid2d)

        return bsplinecurve2d_with_dimension

    def bsplinecurve2d_with_dimension_to_parametric_frame(self, bsplinecurve2d):
        """
        Convert a bsplinecurve2d from the dimensioned to the parametric frame.

        """

        points_dim = bsplinecurve2d.control_points
        points = []
        for point in points_dim:
            points.append(
                self.point2d_with_dimension_to_parametric_frame(point, self._grids2d))

        bsplinecurve2d = edges.BSplineCurve2D(bsplinecurve2d.degree, points,
                                            bsplinecurve2d.knot_multiplicities,
                                            bsplinecurve2d.knots,
                                            bsplinecurve2d.weights,
                                            bsplinecurve2d.periodic)
        return bsplinecurve2d

    def bsplinecurve2d_with_dimension_to_3d(self, bsplinecurve2d):
        """
        Compute the bsplinecurve3d, on a Bspline surface, of a bsplinecurve2d defined in the dimensioned frame.

        """

        bsplinecurve2d_01 = self.bsplinecurve2d_with_dimension_to_parametric_frame(bsplinecurve2d)
        bsplinecurve3d = self.bsplinecurve2d_to_3d(bsplinecurve2d_01)

        return bsplinecurve3d

    def arc2d_parametric_to_dimension(self, arc2d, grid2d: grid.Grid2D):
        """
        Convert an arc 2d from the parametric to the dimensioned frame.

        """

        number_points = math.ceil(arc2d.angle * 7) + 1
        length = arc2d.length()
        points = [self.point2d_parametric_to_dimension(arc2d.point_at_abscissa(
            i * length / (number_points - 1)), grid2d) for i in range(number_points)]

        return edges.BSplineCurve2D.from_points_interpolation(
            points, max(self.degree_u, self.degree_v))

    def arc3d_to_2d_with_dimension(self, arc3d, grid2d: grid.Grid2D):
        """
        Compute the arc 2d of an arc 3d, on a Bspline surface, in the dimensioned frame.

        """

        bsplinecurve2d = self.arc3d_to_2d(arc3d)[0]  # it's a bsplinecurve2d
        arc2d_with_dimension = self.bsplinecurve2d_parametric_to_dimension(bsplinecurve2d, grid2d)

        return arc2d_with_dimension  # it's a bsplinecurve2d-dimension

    def arc2d_with_dimension_to_parametric_frame(self, arc2d):
        """
        Convert an arc 2d from the dimensioned to the parametric frame.

        """

        number_points = math.ceil(arc2d.angle * 7) + 1
        length = arc2d.length()

        points = [self.point2d_with_dimension_to_parametric_frame(arc2d.point_at_abscissa(
            i * length / (number_points - 1)), self._grids2d) for i in range(number_points)]

        return edges.BSplineCurve2D.from_points_interpolation(points, max(self.degree_u, self.degree_v))

    def arc2d_with_dimension_to_3d(self, arc2d):
        """
        Compute the arc 3d, on a Bspline surface, of an arc 2d in the dimensioned frame.

        """

        arc2d_01 = self.arc2d_with_dimension_to_parametric_frame(arc2d)
        arc3d = self.arc2d_to_3d(arc2d_01)

        return arc3d  # it's a bsplinecurve3d

    def contour2d_parametric_to_dimension(self, contour2d: wires.Contour2D,
                                          grid2d: grid.Grid2D):
        """
        Convert a contour 2d from the parametric to the dimensioned frame.

        """

        primitives2d_dim = []

        for primitive2d in contour2d.primitives:
            method_name = f'{primitive2d.__class__.__name__.lower()}_parametric_to_dimension'

            if hasattr(self, method_name):
                primitives = getattr(self, method_name)(primitive2d, grid2d)
                if primitives:
                    primitives2d_dim.append(primitives)

            else:
                raise NotImplementedError(
                    f'Class {self.__class__.__name__} does not implement {method_name}')

        return wires.Contour2D(primitives2d_dim)

    def contour3d_to_2d_with_dimension(self, contour3d: wires.Contour3D,
                                       grid2d: grid.Grid2D):
        """
        Compute the Contour 2d of a Contour 3d, on a Bspline surface, in the dimensioned frame.

        """

        contour2d_01 = self.contour3d_to_2d(contour3d)

        return self.contour2d_parametric_to_dimension(contour2d_01, grid2d)

    def contour2d_with_dimension_to_parametric_frame(self, contour2d):
        """
        Convert a contour 2d from the dimensioned to the parametric frame.

        """

        # TODO: check and avoid primitives with start=end
        primitives2d = []

        for primitive2d in contour2d.primitives:
            method_name = f'{primitive2d.__class__.__name__.lower()}_with_dimension_to_parametric_frame'

            if hasattr(self, method_name):
                primitives = getattr(self, method_name)(primitive2d)
                if primitives:
                    primitives2d.append(primitives)

            else:
                raise NotImplementedError(
                    f'Class {self.__class__.__name__} does not implement {method_name}')

        # #Avoid to have primitives with start=end
        # start_points = []
        # for i in range(0, len(new_start_points)-1):
        #     if new_start_points[i] != new_start_points[i+1]:
        #         start_points.append(new_start_points[i])
        # if new_start_points[-1] != new_start_points[0]:
        #     start_points.append(new_start_points[-1])

        return wires.Contour2D(primitives2d)

    def contour2d_with_dimension_to_3d(self, contour2d):
        """
        Compute the contour3d, on a Bspline surface, of a contour2d define in the dimensioned frame.

        """

        contour01 = self.contour2d_with_dimension_to_parametric_frame(contour2d)

        return self.contour2d_to_3d(contour01)

    @classmethod
    def from_geomdl_surface(cls, surface):
        """
        Create a volmdlr BSpline_Surface3D from a geomdl's one.

        """

        control_points = []
        for point in surface.ctrlpts:
            control_points.append(volmdlr.Point3D(point[0], point[1], point[2]))

        (u_knots, u_multiplicities) = knots_vector_inv(surface.knotvector_u)
        (v_knots, v_multiplicities) = knots_vector_inv(surface.knotvector_v)

        bspline_surface = cls(degree_u=surface.degree_u,
                              degree_v=surface.degree_v,
                              control_points=control_points,
                              nb_u=surface.ctrlpts_size_u,
                              nb_v=surface.ctrlpts_size_v,
                              u_multiplicities=u_multiplicities,
                              v_multiplicities=v_multiplicities,
                              u_knots=u_knots,
                              v_knots=v_knots)

        return bspline_surface

    @classmethod
    def points_fitting_into_bspline_surface(cls, points_3d, size_u, size_v, degree_u, degree_v):
        """
        Bspline Surface interpolation through 3d points.

        Parameters
        ----------
        points_3d : volmdlr.Point3D
            data points
        size_u : int
            number of data points on the u-direction.
        size_v : int
            number of data points on the v-direction.
        degree_u : int
            degree of the output surface for the u-direction.
        degree_v : int
            degree of the output surface for the v-direction.

        Returns
        -------
        B-spline surface

        """

        points = []
        for point in points_3d:
            points.append((point.x, point.y, point.z))

        surface = interpolate_surface(points, size_u, size_v, degree_u, degree_v)

        return cls.from_geomdl_surface(surface)

    @classmethod
    def points_approximate_into_bspline_surface(cls, points_3d, size_u, size_v, degree_u, degree_v, **kwargs):
        """
        Bspline Surface approximate through 3d points.

        Parameters
        ----------
        points_3d : volmdlr.Point3D
            data points
        size_u : int
            number of data points on the u-direction.
        size_v : int
            number of data points on the v-direction.
        degree_u : int
            degree of the output surface for the u-direction.
        degree_v : int
            degree of the output surface for the v-direction.

        Keyword Arguments:
            * ``ctrlpts_size_u``: number of control points on the u-direction. *Default: size_u - 1*
            * ``ctrlpts_size_v``: number of control points on the v-direction. *Default: size_v - 1*

        Returns
        -------
        B-spline surface: volmdlr.faces.BSplineSurface3D

        """

        # Keyword arguments
        # number of data points, r + 1 > number of control points, n + 1
        num_cpts_u = kwargs.get('ctrlpts_size_u', size_u - 1)
        # number of data points, s + 1 > number of control points, m + 1
        num_cpts_v = kwargs.get('ctrlpts_size_v', size_v - 1)

        points = [tuple([*point]) for point in points_3d]

        surface = approximate_surface(points, size_u, size_v, degree_u, degree_v,
                                      ctrlpts_size_u=num_cpts_u, num_cpts_v=num_cpts_v)

        return cls.from_geomdl_surface(surface)

    @classmethod
    def from_cylindrical_faces(cls, cylindrical_faces, degree_u, degree_v,
                               points_x: int = 10, points_y: int = 10):
        """
        Define a bspline surface from a list of cylindrical faces.

        Parameters
        ----------
        cylindrical_faces : List[volmdlr.faces.CylindricalFace3D]
            faces 3d
        degree_u : int
            degree of the output surface for the u-direction
        degree_v : int
            degree of the output surface for the v-direction
        points_x : int
            number of points in x-direction
        points_y : int
            number of points in y-direction

        Returns
        -------
        B-spline surface

        """
        if len(cylindrical_faces) < 1:
            raise NotImplementedError
        if len(cylindrical_faces) == 1:
            return cls.from_cylindrical_face(cylindrical_faces[0], degree_u, degree_v, points_x=50, points_y=50)
        bspline_surfaces = []
        direction = cylindrical_faces[0].adjacent_direction(cylindrical_faces[1])

        if direction == 'x':
            bounding_rectangle_0 = cylindrical_faces[0].surface2d.outer_contour.bounding_rectangle
            ymin = bounding_rectangle_0[2]
            ymax = bounding_rectangle_0[3]
            for face in cylindrical_faces:
                bounding_rectangle = face.surface2d.outer_contour.bounding_rectangle
                ymin = min(ymin, bounding_rectangle[2])
                ymax = max(ymax, bounding_rectangle[3])
            for face in cylindrical_faces:
                bounding_rectangle = face.surface2d.outer_contour.bounding_rectangle

                points_3d = face.surface3d.grid3d(
                    grid.Grid2D.from_properties(
                        x_limits=(bounding_rectangle[0], bounding_rectangle[1]),
                        y_limits=(ymin, ymax),
                        points_nbr=(points_x, points_y)))

                bspline_surfaces.append(
                    cls.points_fitting_into_bspline_surface(
                        points_3d, points_x, points_y, degree_u, degree_v))

        elif direction == 'y':
            bounding_rectangle_0 = cylindrical_faces[0].surface2d.outer_contour.bounding_rectangle
            xmin = bounding_rectangle_0[0]
            xmax = bounding_rectangle_0[1]
            for face in cylindrical_faces:
                bounding_rectangle = face.surface2d.outer_contour.bounding_rectangle
                xmin = min(xmin, bounding_rectangle[0])
                xmax = max(xmax, bounding_rectangle[1])
            for face in cylindrical_faces:
                bounding_rectangle = face.surface2d.outer_contour.bounding_rectangle

                points_3d = face.surface3d.grid3d(
                    grid.Grid2D.from_properties(
                        x_limits=(xmin, xmax),
                        y_limits=(bounding_rectangle[2], bounding_rectangle[3]),
                        points_nbr=(points_x, points_y)))

                bspline_surfaces.append(
                    cls.points_fitting_into_bspline_surface(
                        points_3d, points_x, points_y, degree_u, degree_v))

        to_be_merged = bspline_surfaces[0]
        for i in range(0, len(bspline_surfaces) - 1):
            merged = to_be_merged.merge_with(bspline_surfaces[i + 1])
            to_be_merged = merged

        bspline_surface = to_be_merged

        return bspline_surface

    @classmethod
    def from_cylindrical_face(cls, cylindrical_face, degree_u, degree_v,
                              **kwargs):  # points_x: int = 50, points_y: int = 50
        """
        Define a bspline surface from a cylindrical face.

        Parameters
        ----------
        cylindrical_face : volmdlr.faces.CylindricalFace3D
            face 3d
        degree_u : int
            degree of the output surface for the u-direction.
        degree_v : int
            degree of the output surface for the v-direction.
        points_x : int
            number of points in x-direction
        points_y : int
            number of points in y-direction

        Returns
        -------
        B-spline surface

        """

        points_x = kwargs['points_x']
        points_y = kwargs['points_y']
        bounding_rectangle = cylindrical_face.surface2d.outer_contour.bounding_rectangle
        points_3d = cylindrical_face.surface3d.grid3d(
            grid.Grid2D.from_properties(x_limits=(bounding_rectangle[0],
                                                          bounding_rectangle[1]),
                                                y_limits=(bounding_rectangle[2],
                                                          bounding_rectangle[3]),
                                                points_nbr=(points_x, points_y)))

        return cls.points_fitting_into_bspline_surface(points_3d, points_x, points_x, degree_u, degree_v)

    def intersection_with(self, other_bspline_surface3d):
        """
        Compute intersection points between two Bspline surfaces.

        return u,v parameters for intersection points for both surfaces
        """

        def fun(param):
            return (self.point2d_to_3d(volmdlr.Point2D(param[0], param[1])) -
                    other_bspline_surface3d.point2d_to_3d(volmdlr.Point2D(param[2], param[3]))).norm()

        x = npy.linspace(0, 1, 10)
        x_init = []
        for xi in x:
            for yi in x:
                x_init.append((xi, yi, xi, yi))

        u1, v1, u2, v2 = [], [], [], []
        solutions = []
        for x0 in x_init:
            z = least_squares(fun, x0=x0, bounds=([0, 1]))
            # print(z.cost)
            if z.fun < 1e-5:
                solution = z.x
                if solution not in solutions:
                    solutions.append(solution)
                    u1.append(solution[0])
                    v1.append(solution[1])
                    u2.append(solution[2])
                    v2.append(solution[3])

        # uv1 = [[min(u1),max(u1)],[min(v1),max(v1)]]
        # uv2 = [[min(u2),max(u2)],[min(v2),max(v2)]]

        return (u1, v1), (u2, v2)  # (uv1, uv2)

    def plane_intersection(self, plane3d):
        """
        Compute intersection points between a Bspline surface and a plane 3d.

        """

        def fun(param):
            return ((self.surface.evaluate_single((param[0], param[1]))[0]) * plane3d.equation_coefficients()[0] +
                    (self.surface.evaluate_single((param[0], param[1]))[1]) * plane3d.equation_coefficients()[1] +
                    (self.surface.evaluate_single((param[0], param[1]))[2]) * plane3d.equation_coefficients()[2] +
                    plane3d.equation_coefficients()[3])

        x = npy.linspace(0, 1, 20)
        x_init = []
        for xi in x:
            for yi in x:
                x_init.append((xi, yi))

        intersection_points = []

        for x0 in x_init:
            z = least_squares(fun, x0=x0, bounds=([0, 1]))
            if z.fun < 1e-20:
                solution = z.x
                intersection_points.append(volmdlr.Point3D(self.surface.evaluate_single((solution[0], solution[1]))[0],
                                                           self.surface.evaluate_single((solution[0], solution[1]))[1],
                                                           self.surface.evaluate_single((solution[0], solution[1]))[
                                                               2]))
        return intersection_points

    def error_with_point3d(self, point3d):
        """
        Compute the error/distance between the Bspline surface and a point 3d.

        """

        def fun(x):
            return (point3d - self.point2d_to_3d(volmdlr.Point2D(x[0], x[1]))).norm()

        cost = []

        for x0 in [(0, 0), (0, 1), (1, 0), (1, 1), (0.5, 0.5)]:
            z = least_squares(fun, x0=x0, bounds=([0, 1]))
            cost.append(z.fun)

        return min(cost)

    def error_with_edge3d(self, edge3d):
        """
        Compute the error/distance between the Bspline surface and an edge 3d.

        it's the mean of the start and end points errors'
        """

        return (self.error_with_point3d(edge3d.start) + self.error_with_point3d(edge3d.end)) / 2

    def nearest_edges3d(self, contour3d, threshold: float):
        """
        Compute the nearest edges of a contour 3d to a Bspline_surface3d based on a threshold.

        """

        nearest = []
        for primitive in contour3d.primitives:
            if self.error_with_edge3d(primitive) <= threshold:
                nearest.append(primitive)
        nearest_primitives = wires.Wire3D(nearest)

        return nearest_primitives

    def edge3d_to_2d_with_dimension(self, edge3d, grid2d: grid.Grid2D):
        """
        Compute the edge 2d of an edge 3d, on a Bspline surface, in the dimensioned frame.

        """
        method_name = f'{edge3d.__class__.__name__.lower()}_to_2d_with_dimension'

        if hasattr(self, method_name):
            edge2d_dim = getattr(self, method_name)(edge3d, grid2d)
            if edge2d_dim:
                return edge2d_dim
            raise NotImplementedError
        raise NotImplementedError(
            f'Class {self.__class__.__name__} does not implement {method_name}')

    def wire3d_to_2d(self, wire3d):
        """
        Compute the 2d of a wire 3d, on a Bspline surface.

        """

        contour = self.contour3d_to_2d(wire3d)

        return wires.Wire2D(contour.primitives)

    def wire3d_to_2d_with_dimension(self, wire3d):
        """
        Compute the 2d of a wire 3d, on a Bspline surface, in the dimensioned frame.

        """

        contour = self.contour3d_to_2d_with_dimension(wire3d, self._grids2d)

        return wires.Wire2D(contour.primitives)

    def split_surface_u(self, u: float):
        """
        Splits the surface at the input parametric coordinate on the u-direction.

        :param u: Parametric coordinate u chosen between 0 and 1
        :type u: float
        :return: Two split surfaces
        :rtype: List[:class:`volmdlr.faces.BSplineSurface3D`]
        """

        surfaces_geo = split_surface_u(self.surface, u)
        surfaces = [BSplineSurface3D.from_geomdl_surface(surface) for surface in surfaces_geo]
        return surfaces

    def split_surface_v(self, v: float):
        """
        Splits the surface at the input parametric coordinate on the v-direction.

        :param v: Parametric coordinate v chosen between 0 and 1
        :type v: float
        :return: Two split surfaces
        :rtype: List[:class:`volmdlr.faces.BSplineSurface3D`]
        """

        surfaces_geo = split_surface_v(self.surface, v)
        surfaces = [BSplineSurface3D.from_geomdl_surface(surface) for surface in surfaces_geo]
        return surfaces

    def split_surface_with_bspline_curve(self, bspline_curve3d: edges.BSplineCurve3D):
        """
        Cuts the surface into two pieces with a bspline curve.

        :param bspline_curve3d: A BSplineCurve3d used for cutting
        :type bspline_curve3d: :class:`edges.BSplineCurve3D`
        :return: Two split surfaces
        :rtype: List[:class:`volmdlr.faces.BSplineSurface3D`]
        """

        surfaces = []
        bspline_curve2d = self.bsplinecurve3d_to_2d(bspline_curve3d)[0]
        # if type(bspline_curve2d) == list:
        #     points = [bspline_curve2d[0].start]
        #     for edge in bspline_curve2d:
        #         points.append(edge.end)
        #     bspline_curve2d = edges.BSplineCurve2D.from_points_approximation(points, 2, ctrlpts_size = 5)
        contour = volmdlr.faces.BSplineFace3D.from_surface_rectangular_cut(self, 0, 1, 0, 1).surface2d.outer_contour
        contours = contour.cut_by_bspline_curve(bspline_curve2d)

        du, dv = bspline_curve2d.end - bspline_curve2d.start
        resolution = 8

        for contour in contours:
            u_min, u_max, v_min, v_max = contour.bounding_rectangle.bounds()
            if du > dv:
                delta_u = u_max - u_min
                nlines_x = int(delta_u * resolution)
                lines_x = [edges.Line2D(volmdlr.Point2D(u_min, v_min),
                                      volmdlr.Point2D(u_min, v_max))]
                for i in range(nlines_x):
                    u = u_min + (i + 1) / (nlines_x + 1) * delta_u
                    lines_x.append(edges.Line2D(volmdlr.Point2D(u, v_min),
                                              volmdlr.Point2D(u, v_max)))
                lines_x.append(edges.Line2D(volmdlr.Point2D(u_max, v_min),
                                          volmdlr.Point2D(u_max, v_max)))
                lines = lines_x

            else:
                delta_v = v_max - v_min
                nlines_y = int(delta_v * resolution)
                lines_y = [edges.Line2D(volmdlr.Point2D(v_min, v_min),
                                      volmdlr.Point2D(v_max, v_min))]
                for i in range(nlines_y):
                    v = v_min + (i + 1) / (nlines_y + 1) * delta_v
                    lines_y.append(edges.Line2D(volmdlr.Point2D(v_min, v),
                                              volmdlr.Point2D(v_max, v)))
                lines_y.append(edges.Line2D(volmdlr.Point2D(v_min, v_max),
                                          volmdlr.Point2D(v_max, v_max)))
                lines = lines_y

            pt0 = volmdlr.O2D
            points = []

            for line in lines:
                inter = contour.line_intersections(line)
                if inter:
                    pt_ = set()
                    for point_intersection in inter:
                        pt_.add(point_intersection[0])
                else:
                    raise NotImplementedError

                pt_ = sorted(pt_, key=pt0.point_distance)
                pt0 = pt_[0]
                edge = edges.LineSegment2D(pt_[0], pt_[1])

                points.extend(edge.discretization_points(number_points=10))

            points3d = []
            for point in points:
                points3d.append(self.point2d_to_3d(point))

            size_u, size_v, degree_u, degree_v = 10, 10, self.degree_u, self.degree_v
            surfaces.append(
                BSplineSurface3D.points_fitting_into_bspline_surface(points3d, size_u, size_v, degree_u, degree_v))

        return surfaces

    def point_belongs(self, point3d):
        """
        Check if a point 3d belongs to the bspline_surface or not.

        """

        def fun(param):
            p3d = self.point2d_to_3d(volmdlr.Point2D(param[0], param[1]))
            return point3d.point_distance(p3d)

        x = npy.linspace(0, 1, 5)
        x_init = []
        for xi in x:
            for yi in x:
                x_init.append((xi, yi))

        for x0 in x_init:
            z = least_squares(fun, x0=x0, bounds=([0, 1]))
            if z.fun < 1e-10:
                return True
        return False

    def is_intersected_with(self, other_bspline_surface3d):
        """
        Check if the two surfaces are intersected or not.

        return True, when there are more 50points on the intersection zone.

        """

        # intersection_results = self.intersection_with(other_bspline_surface3d)
        # if len(intersection_results[0][0]) >= 50:
        #     return True
        # else:
        #     return False

        def fun(param):
            return (self.point2d_to_3d(volmdlr.Point2D(param[0], param[1])) -
                    other_bspline_surface3d.point2d_to_3d(volmdlr.Point2D(param[2], param[3]))).norm()

        x = npy.linspace(0, 1, 10)
        x_init = []
        for xi in x:
            for yi in x:
                x_init.append((xi, yi, xi, yi))

        i = 0
        for x0 in x_init:
            z = least_squares(fun, x0=x0, bounds=([0, 1]))
            if z.fun < 1e-5:
                i += 1
                if i >= 50:
                    return True
        return False

    def merge_with(self, other_bspline_surface3d, abs_tol: float = 1e-6):
        """
        Merges two adjacent surfaces based on their faces.

        :param other_bspline_surface3d: Other adjacent surface
        :type other_bspline_surface3d: :class:`volmdlr.faces.BSplineSurface3D`
        :param abs_tol: tolerance.
        :type abs_tol: float.

        :return: Merged surface
        :rtype: :class:`volmdlr.faces.BSplineSurface3D`
        """

        bspline_face3d = volmdlr.faces.BSplineFace3D.from_surface_rectangular_cut(self, 0, 1, 0, 1)
        other_bspline_face3d = volmdlr.faces.BSplineFace3D.from_surface_rectangular_cut(
            other_bspline_surface3d, 0, 1, 0, 1)

        bsplines = [self, other_bspline_surface3d]
        bsplines_new = bsplines

        center = [bspline_face3d.surface2d.outer_contour.center_of_mass(),
                  other_bspline_face3d.surface2d.outer_contour.center_of_mass()]
        grid2d_direction = (bspline_face3d.pair_with(other_bspline_face3d))[1]

        if (not bspline_face3d.outer_contour3d.is_sharing_primitives_with(
                other_bspline_face3d.outer_contour3d, abs_tol)
                and self.is_intersected_with(other_bspline_surface3d)):
            # find primitives to split with
            contour1 = bspline_face3d.outer_contour3d
            contour2 = other_bspline_face3d.outer_contour3d

            distances = []
            for prim1 in contour1.primitives:
                dis = []
                for prim2 in contour2.primitives:
                    point1 = (prim1.start + prim1.end) / 2
                    point2 = (prim2.start + prim2.end) / 2
                    dis.append(point1.point_distance(point2))
                distances.append(dis)

            i = distances.index((min(distances)))
            j = distances[i].index(min(distances[i]))

            curves = [contour2.primitives[j], contour1.primitives[i]]

            # split surface
            for i, bspline in enumerate(bsplines):
                surfaces = bspline.split_surface_with_bspline_curve(curves[i])

                errors = []
                for surface in surfaces:
                    errors.append(surface.error_with_point3d(bsplines[i].point2d_to_3d(center[i])))

                bsplines_new[i] = surfaces[errors.index(min(errors))]

            grid2d_direction = (
                bsplines_new[0].rectangular_cut(
                    0, 1, 0, 1).pair_with(
                    bsplines_new[1].rectangular_cut(
                        0, 1, 0, 1)))[1]

        # grid3d
        number_points = 10
        points3d = []
        is_true = (bspline_face3d.outer_contour3d.is_sharing_primitives_with(
            other_bspline_face3d.outer_contour3d, abs_tol) or self.is_intersected_with(other_bspline_surface3d))

        for i, bspline in enumerate(bsplines_new):
            grid3d = bspline.grid3d(grid.Grid2D.from_properties(x_limits=(0, 1),
                                                                y_limits=(0, 1),
                                                                points_nbr=(number_points, number_points),
                                                                direction=grid2d_direction[i]))

            if is_true and i == 1:
                points3d.extend(grid3d[number_points:number_points * number_points])
            else:
                points3d.extend(grid3d)

        # fitting
        size_u, size_v, degree_u, degree_v = (number_points * 2) - 1, number_points, 3, 3

        merged_surface = BSplineSurface3D.points_fitting_into_bspline_surface(
            points3d, size_u, size_v, degree_u, degree_v)

        return merged_surface

    def xy_limits(self, other_bspline_surface3d):
        """
        Compute x, y limits to define grid2d.

        """

        grid2d_direction = (
            self.rectangular_cut(
                0, 1, 0, 1).pair_with(
                other_bspline_surface3d.rectangular_cut(
                    0, 1, 0, 1)))[1]

        xmin, xmax, ymin, ymax = [], [], [], []
        if grid2d_direction[0][1] == '+y':
            xmin.append(0)
            xmax.append(1)
            ymin.append(0)
            ymax.append(0.99)
        elif grid2d_direction[0][1] == '+x':
            xmin.append(0)
            xmax.append(0.99)
            ymin.append(0)
            ymax.append(1)
        elif grid2d_direction[0][1] == '-x':
            xmin.append(0.01)
            xmax.append(1)
            ymin.append(0)
            ymax.append(1)
        elif grid2d_direction[0][1] == '-y':
            xmin.append(0)
            xmax.append(1)
            ymin.append(0.01)
            ymax.append(1)

        xmin.append(0)
        xmax.append(1)
        ymin.append(0)
        ymax.append(1)

        return xmin, xmax, ymin, ymax

    def derivatives(self, u, v, order):
        """
        Evaluates n-th order surface derivatives at the given (u, v) parameter pair.

        :param u: Point's u coordinate.
        :type u: float
        :param v: Point's v coordinate.
        :type v: float
        :param order: Order of the derivatives.
        :type order: int
        :return: A list SKL, where SKL[k][l] is the derivative of the surface S(u,v) with respect
        to u k times and v l times
        :rtype: List[`volmdlr.Vector3D`]
        """
        if self.surface.rational:
            # derivatives = self._rational_derivatives(self.surface.data,(u, v), order)
            derivatives = volmdlr.bspline_compiled.rational_derivatives(self.surface.data, (u, v), order)
        else:
            # derivatives = self._derivatives(self.surface.data, (u, v), order)
            derivatives = volmdlr.bspline_compiled.derivatives(self.surface.data, (u, v), order)
        for i in range(order + 1):
            for j in range(order + 1):
                derivatives[i][j] = volmdlr.Vector3D(*derivatives[i][j])
        return derivatives

    def repair_contours2d(self, outer_contour, inner_contours):
        """
        Repair contours on parametric domain.

        :param outer_contour: Outer contour 2D.
        :type inner_contours: wires.Contour2D
        :param inner_contours: List of 2D contours.
        :type inner_contours: list
        """
        new_inner_contours = []
        point1 = outer_contour.primitives[0].start
        point2 = outer_contour.primitives[-1].end

        u1, v1 = point1
        u2, v2 = point2

        for inner_contour in inner_contours:
            u3, v3 = inner_contour.primitives[0].start
            u4, v4 = inner_contour.primitives[-1].end

            if not inner_contour.is_ordered():
                if self.x_periodicity and self.y_periodicity:
                    raise NotImplementedError
                if self.x_periodicity:
                    outer_contour_param = [u1, u2]
                    inner_contour_param = [u3, u4]
                elif self.y_periodicity:
                    outer_contour_param = [v1, v2]
                    inner_contour_param = [v3, v4]
                else:
                    raise NotImplementedError

                point1 = outer_contour.primitives[0].start
                point2 = outer_contour.primitives[-1].end
                point3 = inner_contour.primitives[0].start
                point4 = inner_contour.primitives[-1].end

                outer_contour_direction = outer_contour_param[0] < outer_contour_param[1]
                inner_contour_direction = inner_contour_param[0] < inner_contour_param[1]
                if outer_contour_direction == inner_contour_direction:
                    inner_contour = inner_contour.invert()
                    point3 = inner_contour.primitives[0].start
                    point4 = inner_contour.primitives[-1].end

                closing_linesegment1 = edges.LineSegment2D(point2, point3)
                closing_linesegment2 = edges.LineSegment2D(point4, point1)
                new_outer_contour_primitives = outer_contour.primitives + [closing_linesegment1] + \
                                               inner_contour.primitives + \
                                               [closing_linesegment2]
                new_outer_contour = wires.Contour2D(primitives=new_outer_contour_primitives)
                new_outer_contour.order_contour(tol=1e-4)
            else:
                new_inner_contours.append(inner_contour)
        return new_outer_contour, new_inner_contours

    def _get_overlapping_theta(self, outer_contour_startend_theta, inner_contour_startend_theta):
        """
        Find overlapping theta domain between two contours on periodical Surfaces.
        """
        oc_xmin_index, outer_contour_xmin = min(enumerate(outer_contour_startend_theta), key=lambda x: x[1])
        oc_xmax_index, outer_contour_xman = max(enumerate(outer_contour_startend_theta), key=lambda x: x[1])
        inner_contour_xmin = min(inner_contour_startend_theta)
        inner_contour_xmax = max(inner_contour_startend_theta)

        # check if tetha3 or theta4 is in [theta1, theta2] interval
        overlap = outer_contour_xmin <= inner_contour_xmax and outer_contour_xman >= inner_contour_xmin

        if overlap:
            if inner_contour_xmin < outer_contour_xmin:
                overlapping_theta = outer_contour_startend_theta[oc_xmin_index]
                outer_contour_side = oc_xmin_index
                side = 0
                return overlapping_theta, outer_contour_side, side
            overlapping_theta = outer_contour_startend_theta[oc_xmax_index]
            outer_contour_side = oc_xmax_index
            side = 1
            return overlapping_theta, outer_contour_side, side

        # if not direct intersection -> find intersection at periodicity
        if inner_contour_xmin < outer_contour_xmin:
            overlapping_theta = outer_contour_startend_theta[oc_xmin_index] - 2 * math.pi
            outer_contour_side = oc_xmin_index
            side = 0
            return overlapping_theta, outer_contour_side, side
        overlapping_theta = outer_contour_startend_theta[oc_xmax_index] + 2 * math.pi
        outer_contour_side = oc_xmax_index
        side = 1
        return overlapping_theta, outer_contour_side, side

    def to_plane3d(self):
        """
        Converts a Bspline surface3d to a Plane3d.

        :return: A Plane
        :rtype: Plane3D
        """

        points_2d = [volmdlr.Point2D(0.1, 0.1),
                     volmdlr.Point2D(0.1, 0.8),
                     volmdlr.Point2D(0.8, 0.5)]
        points = [self.point2d_to_3d(pt) for pt in points_2d]

        surface3d = Plane3D.from_3_points(points[0],
                                          points[1],
                                          points[2])
        return surface3d


class BezierSurface3D(BSplineSurface3D):
    """
    A 3D Bezier surface.

    :param degree_u: The degree of the Bezier surface in the u-direction.
    :type degree_u: int
    :param degree_v: The degree of the Bezier surface in the v-direction.
    :type degree_v: int
    :param control_points: A list of lists of control points defining the Bezier surface.
    :type control_points: List[List[`volmdlr.Point3D`]]
    :param nb_u: The number of control points in the u-direction.
    :type nb_u: int
    :param nb_v: The number of control points in the v-direction.
    :type nb_v: int
    :param name: (Optional) name for the Bezier surface.
    :type name: str
    """

    def __init__(self, degree_u: int, degree_v: int,
                 control_points: List[List[volmdlr.Point3D]],
                 nb_u: int, nb_v: int, name=''):
        u_knots = utilities.generate_knot_vector(degree_u, nb_u)
        v_knots = utilities.generate_knot_vector(degree_v, nb_v)

        u_multiplicities = [1] * len(u_knots)
        v_multiplicities = [1] * len(v_knots)

        BSplineSurface3D.__init__(self, degree_u, degree_v,
                                  control_points, nb_u, nb_v,
                                  u_multiplicities, v_multiplicities,
                                  u_knots, v_knots, None, name)