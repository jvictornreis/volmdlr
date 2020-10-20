#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
from packaging import version
import math
import numpy as npy

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import __version__ as _mpl_version
import matplotlib.pyplot as plt
import matplotlib.patches

import dessia_common as dc
import volmdlr.core
# import volmdlr.primitives3D


class Edge(dc.DessiaObject):
    def __init__(self, start, end, name=''):
        self.start = start
        self.end = end
        dc.DessiaObject.__init__(self, name=name)

    @classmethod
    def from_step(cls, arguments, object_dict):
        if object_dict[arguments[3]].__class__ is Line3D:
            return LineSegment3D(object_dict[arguments[1]],
                                 object_dict[arguments[2]], arguments[0][1:-1])

        elif object_dict[arguments[3]].__class__ is volmdlr.primitives3D.Circle3D:
            # We supposed that STEP file is reading on trigo way
            center = object_dict[arguments[3]].center
            normal = object_dict[arguments[3]].normal
            normal.normalize()
            radius = object_dict[arguments[3]].radius
            p1 = object_dict[arguments[1]]
            p2 = object_dict[arguments[2]]
            other_vec = object_dict[arguments[3]].other_vec
            if other_vec is None:
                other_vec = p1 - center
            other_vec.normalize()
            frame = volmdlr.core.Frame3D(center, other_vec, normal.cross(other_vec), normal)
            if p1 == p2:
                angle = math.pi
            else:
                theta1, theta2 = volmdlr.core.posangle_arc(p1, p2, radius, frame)
                if theta1 > theta2:  # sens trigo
                    angle = math.pi + (theta1 + theta2) / 2
                else:
                    angle = (theta1 + theta2) / 2
            p_3 = volmdlr.core.Point3D(radius * math.cos(angle),
                                       radius * math.sin(angle),
                                       0)
            p3 = frame.OldCoordinates(p_3)
            if p1 == p3 or p2 == p3:
                p_3 = volmdlr.core.Point3D(radius * math.cos(0), radius * math.sin(0), 0)
                p3 = frame.OldCoordinates(p_3)
            arc = volmdlr.primitives3D.Arc3D(p1, p3, p2, normal, arguments[0][1:-1], other_vec)
            if math.isclose(arc.radius, 0, abs_tol=1e-9):
                if p1 == p2:
                    p_3 = volmdlr.Point3D(
                        (radius * math.cos(0), radius * math.sin(0), 0))
                    p3 = frame.OldCoordinates(p_3)
                    arc = volmdlr.primitives3D.Arc3D(p1, p3, p2, normal, arguments[0][1:-1],
                                other_vec)
            return arc

        elif object_dict[arguments[3]].__class__ is volmdlr.primitives3D.Ellipse3D:
            majorax = object_dict[arguments[3]].major_axis
            minorax = object_dict[arguments[3]].minor_axis
            center = object_dict[arguments[3]].center
            normal = object_dict[arguments[3]].normal
            normal.normalize()
            majordir = object_dict[arguments[3]].major_dir
            majordir.normalize()
            minordir = normal.cross(majordir)
            minordir.normalize()
            frame = volmdlr.Frame3D(center, majordir, minordir, normal)
            p1 = object_dict[
                arguments[1]]  # on part du principe que p1 suivant majordir
            p2 = object_dict[arguments[2]]
            if p1 == p2:
                angle = 5 * math.pi / 4
                xtra = volmdlr.Point3D((majorax * math.cos(math.pi / 2),
                                minorax * math.sin(math.pi / 2), 0))
                extra = frame.OldCoordinates(xtra)
            else:
                extra = None
                ## Positionnement des points dans leur frame
                p1_new, p2_new = frame.NewCoordinates(
                    p1), frame.NewCoordinates(p2)
                # Angle pour le p1
                u1, u2 = p1_new.vector[0] / majorax, p1_new.vector[1] / minorax
                theta1 = volmdlr.sin_cos_angle(u1, u2)
                # Angle pour le p2
                u3, u4 = p2_new.vector[0] / majorax, p2_new.vector[1] / minorax
                theta2 = volmdlr.sin_cos_angle(u3, u4)

                if theta1 > theta2:  # sens trigo
                    angle = math.pi + (theta1 + theta2) / 2
                else:
                    angle = (theta1 + theta2) / 2

            p_3 = volmdlr.Point3D(
                (majorax * math.cos(angle), minorax * math.sin(angle), 0))
            p3 = frame.OldCoordinates(p_3)

            arcellipse = ArcEllipse3D(p1, p3, p2, center, majordir, normal,
                                      arguments[0][1:-1], extra)

            return arcellipse

        elif object_dict[arguments[3]].__class__ is volmdlr.primitives3D.BSplineCurve3D:
            # print(object_dict[arguments[1]], object_dict[arguments[2]])
            # BSplineCurve3D à couper à gauche et à droite avec les points ci dessus ?
            return object_dict[arguments[3]]

        else:
            print(object_dict[arguments[3]])
            raise NotImplementedError

class Line(dc.DessiaObject):
    """
    Abstract class
    """
    def __init__(self, point1, point2, name=''):
        self.point1 = point1
        self.point2 = point2

    def unit_direction_vector(self):
        u = self.direction_vector()
        u.normalize()
        return u

    def direction_vector(self):
        return self.point2 - self.point1

    def normal_vector(self):
        return self.unit_direction_vector().normal_vector()

    def point_projection(self, point):

        u = self.point2 - self.point1
        norm_u = u.norm()
        t = (point - self.point1).dot(u) / norm_u ** 2
        projection = self.point1 + t * u

        return projection, t * norm_u

    def split(self, split_point):
        return [self.__class__(self.point1, split_point),
                self.__class__(split_point, self.point2)]

class LineSegment(Edge):
    """
    Abstract class
    """

    def unit_direction_vector(self):
        u = self.direction_vector()
        u.normalize()
        return u

    def direction_vector(self):
        return self.end - self.start

    def normal_vector(self):
        return self.unit_direction_vector().normal_vector()

    def point_projection(self, point):
        p1, p2 = self.points
        u = p2 - p1
        norm_u = u.norm()
        t = (point - p1).dot(u) / norm_u ** 2
        projection = p1 + t * u

        return projection, t * norm_u

    def split(self, split_point):
        return [self.__class__(self.start, split_point),
                self.__class__(split_point, self.end)]



class Line2D(Line):
    """
    Define an infinite line given by two points.
    """

    def __init__(self, point1, point2, *, name=''):
        Line.__init__(self, point1, point2, name=name)

    def to_3d(self, plane_origin, x1, x2):
        p3D = [p.to_3d(plane_origin, x1, x2) for p in self.points]
        return Line2D(*p3D, self.name)

    def rotation(self, center, angle, copy=True):
        if copy:
            return Line2D(
                *[p.rotation(center, angle, copy=True) for p in self.points])
        else:
            for p in self.points:
                p.rotation(center, angle, copy=False)

    def translation(self, offset, copy=True):
        if copy:
            return Line2D(
                *[p.translation(offset, copy=True) for p in self.points])
        else:
            for p in self.points:
                p.translation(offset, copy=False)

    def plot(self, ax=None, color='k', dashed=True):
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')

        p1, p2 = self.points

        if version.parse(_mpl_version) >= version.parse('3.3.2'):
            if dashed:
                ax.axline(p1.vector, p2.vector, dashes=[30, 5, 10, 5])
            else:
                ax.axline(p1.vector, p2.vector)
        else:
            u = p2 - p1
            p3 = p1 - 3 * u
            p4 = p2 + 4 * u
            if dashed:
                ax.plot([p3[0], p4[0]], [p3[1], p4[1]], color=color,
                        dashes=[30, 5, 10, 5])
            else:
                ax.plot([p3[0], p4[0]], [p3[1], p4[1]], color=color)

        return ax

    def plot_data(self, marker=None, color='black', stroke_width=1,
                  dash=False, opacity=1, arrow=False):
        p1, p2 = self.points
        u = p2 - p1
        p3 = p1 - 3 * u
        p4 = p2 + 4 * u
        return {'type': 'line',
                'data': [p3[0], p3[1],
                         p4[0], p4[1]],
                'color': color,
                'marker': marker,
                'size': stroke_width,
                'dash': dash,
                'opacity': opacity,
                'arrow': arrow
                }

    def CreateTangentCircle(self, point, other_line):
        """
        Computes the two circles that are tangent to 2 lines and intersect
        a point located on one of the two lines.
        """

        # point will be called I(x_I, y_I)
        # self will be (AB)
        # line will be (CD)

        if math.isclose(self.point_distance(point), 0, abs_tol=1e-10):
            I = volmdlr.Vector2D(point[0], point[1])
            A = volmdlr.Vector2D(self.points[0][0], self.points[0][1])
            B = volmdlr.Vector2D(self.points[1][0], self.points[1][1])
            C = volmdlr.Vector2D(other_line.points[0][0], other_line.points[0][1])
            D = volmdlr.Vector2D(other_line.points[1][0], other_line.points[1][1])

        elif math.isclose(other_line.point_distance(point), 0, abs_tol=1e-10):
            I = volmdlr.Vector2D(point[0], point[1])
            C = volmdlr.Vector2D(self.points[0][0], self.points[0][1])
            D = volmdlr.Vector2D(self.points[1][0], self.points[1][1])
            A = volmdlr.Vector2D(other_line.points[0][0], other_line.points[0][1])
            B = volmdlr.Vector2D(other_line.points[1][0], other_line.points[1][1])
        else:
            raise AttributeError("The point isn't on any of the two lines")

        # CHANGEMENT DE REPAIRE
        new_u = volmdlr.Vector2D((B - A))
        new_u.normalize()
        new_v = new_u.normalVector(unit=True)
        new_basis = volmdlr.Frame2D(I, new_u, new_v)

        new_A = new_basis.NewCoordinates(A)
        new_B = new_basis.NewCoordinates(B)
        new_C = new_basis.NewCoordinates(C)
        new_D = new_basis.NewCoordinates(D)

        if new_C[1] == 0 and new_D[1] == 0:
            # Segments are on the same line: no solution
            return None, None

        elif math.isclose(self.DirectionVector(unit=True).dot(
                other_line.normalVector(unit=True)), 0, abs_tol=1e-06):
            # Parallel segments: one solution

            segments_distance = abs(new_C[1] - new_A[1])
            r = segments_distance / 2
            new_circle_center = volmdlr.Point2D((0, npy.sign(new_C[1] - new_A[1]) * r))
            circle_center = new_basis.OldCoordinates(new_circle_center)
            circle = volmdlr.wires.Circle2D(circle_center, r)

            return circle, None

        elif math.isclose(self.DirectionVector(unit=True).dot(
                other_line.DirectionVector(unit=True)), 0, abs_tol=1e-06):
            # Perpendicular segments: 2 solution
            line_AB = Line2D(volmdlr.Point2D(new_A), volmdlr.Point2D(new_B))
            line_CD = Line2D(Pvolmdlr.oint2D(new_C), volmdlr.Point2D(new_D))
            new_pt_K = volmdlr.Point2D.LinesIntersection(line_AB, line_CD)

            r = abs(new_pt_K[0])
            new_circle_center1 = volmdlr.Point2D((0, r))
            new_circle_center2 = volmdlr.Point2D((0, -r))
            circle_center1 = new_basis.OldCoordinates(new_circle_center1)
            circle_center2 = new_basis.OldCoordinates(new_circle_center2)
            circle1 = volmdlr.wires.Circle2D(circle_center1, r)
            circle2 = volmdlr.wires.Circle2D(circle_center2, r)

            return circle1, circle2

        # =============================================================================
        # LES SEGMENTS SONT QUELCONQUES
        #   => 2 SOLUTIONS
        # =============================================================================
        else:

            line_AB = Line2D(volmdlr.Point2D(new_A), volmdlr.Point2D(new_B))
            line_CD = Line2D(volmdlr.Point2D(new_C), volmdlr.Point2D(new_D))
            new_pt_K = volmdlr.Point2D.LinesIntersection(line_AB, line_CD)
            pt_K = volmdlr.Point2D(new_basis.OldCoordinates(new_pt_K))

            if pt_K == I:
                return None, None

            # CHANGEMENT DE REPERE:
            new_u2 = volmdlr.Vector2D(pt_K - I)
            new_u2.normalize()
            new_v2 = new_u2.normalVector(unit=True)
            new_basis2 = volmdlr.Frame2D(I, new_u2, new_v2)

            new_A = new_basis2.NewCoordinates(A)
            new_B = new_basis2.NewCoordinates(B)
            new_C = new_basis2.NewCoordinates(C)
            new_D = new_basis2.NewCoordinates(D)
            new_pt_K = new_basis2.NewCoordinates(pt_K)

            teta1 = math.atan2(new_C[1], new_C[0] - new_pt_K[0])
            teta2 = math.atan2(new_D[1], new_D[0] - new_pt_K[0])

            if teta1 < 0:
                teta1 += math.pi
            if teta2 < 0:
                teta2 += math.pi

            if not math.isclose(teta1, teta2, abs_tol=1e-08):
                if math.isclose(teta1, math.pi, abs_tol=1e-08) or math.isclose(
                        teta1, 0., abs_tol=1e-08):
                    teta = teta2
                elif math.isclose(teta2, math.pi,
                                  abs_tol=1e-08) or math.isclose(teta2, 0.,
                                                                 abs_tol=1e-08):
                    teta = teta1
            else:
                teta = teta1

            r1 = new_pt_K[0] * math.sin(teta) / (1 + math.cos(teta))
            r2 = new_pt_K[0] * math.sin(teta) / (1 - math.cos(teta))

            new_circle_center1 = volmdlr.Point2D(0, -r1)
            new_circle_center2 = volmdlr.Point2D(0, r2)

            circle_center1 = new_basis2.OldCoordinates(new_circle_center1)
            circle_center2 = new_basis2.OldCoordinates(new_circle_center2)

            if new_basis.NewCoordinates(circle_center1)[1] > 0:
                circle1 = volmdlr.wires.Circle2D(circle_center1, r1)
                circle2 = volmdlr.wires.Circle2D(circle_center2, r2)
            else:
                circle1 = volmdlr.wires.Circle2D(circle_center2, r2)
                circle2 = volmdlr.wires.Circle2D(circle_center1, r1)

            return circle1, circle2


class BSplineCurve2D(Edge):
    def __init__(self, degree, control_points, knot_multiplicities, knots,
                 weights=None, periodic=False, name=''):
        self.control_points = control_points
        self.degree = degree
        knots = volmdlr.standardize_knot_vector(knots)
        self.knots = knots
        self.knot_multiplicities = knot_multiplicities
        self.weights = weights
        self.periodic = periodic
        start = self.point_at_abscissa(0)
        end = self.point_at_abscissa(self.length())

        Edge.__init__(self, start, end, name=name)

        curve = BSpline.Curve()
        curve.degree = degree
        if weights is None:
            P = [(control_points[i][0], control_points[i][1]) for i in
                 range(len(control_points))]
            curve.ctrlpts = P
        else:
            Pw = [(control_points[i][0] * weights[i],
                   control_points[i][1] * weights[i], weights[i]) for i in
                  range(len(control_points))]
            curve.ctrlptsw = Pw
        knot_vector = []
        for i, knot in enumerate(knots):
            knot_vector.extend([knot] * knot_multiplicities[i])
        curve.knotvector = knot_vector
        curve.delta = 0.1
        curve_points = curve.evalpts

        self.curve = curve
        self.points = [Point2D((p[0], p[1])) for p in curve_points]

    def length(self):
        # Approximately
        length = 0
        for k in range(0, len(self.points) - 1):
            length += (self.points[k] - self.points[k + 1]).norm()
        return length

    def point_at_abscissa(self, curvilinear_abscissa):
        # copy paste from wire3D
        length = 0.
        primitives = []
        for k in range(0, len(self.points) - 1):
            primitives.append(
                LineSegment2D(self.points[k], self.points[k + 1]))
        for primitive in primitives:
            primitive_length = primitive.length()
            if length + primitive_length >= curvilinear_abscissa:
                return primitive.point_at_abscissa(
                    curvilinear_abscissa - length)
            length += primitive_length
        # Outside of length
        raise ValueError

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        x = [p.vector[0] for p in self.points]
        y = [p.vector[1] for p in self.points]
        ax.plot(x, y, 'o-k')
        return fig, ax

    def to_3d(self, plane_origin, x1, x2):
        control_points3D = [p.to_3d(plane_origin, x1, x2) for p in
                            self.control_points]
        return BSplineCurve3D(self.degree, control_points3D,
                              self.knot_multiplicities, self.knots,
                              self.weights, self.periodic, self.name)

    def tessellation_points(self):
        return self.points


class LineSegment2D(LineSegment):
    """
    Define a line segment limited by two points
    """

    def __init__(self, start, end, *, name=''):
        Edge.__init__(self, start, end, name=name)

    def length(self):
        return self.end.point_distance(self.start)

    def point_at_abscissa(self, curvilinear_abscissa):
        return self.start + self.unit_direction_vector() * curvilinear_abscissa

    def point_distance(self, point, return_other_point=False):
        """
        Computes the distance of a point to segment of line
        """
        if self.point1 == self.point2:
            if return_other_point:
                return 0, Point2D(point)
            return 0
        distance, point = LineSegment2DPointDistance(
            [p.vector for p in self.points], point.vector)
        if return_other_point:
            return distance, Point2D(point)
        return distance

    def point_projection(self, point):
        """
        If the projection falls outside the LineSegment2D, returns None.
        """
        point, curv_abs = Line2D.point_projection(self, point)
        if curv_abs < 0 or curv_abs > self.length():
            return None, curv_abs
        return point, curv_abs

    def line_intersections(self, line):
        point = Point2D.LinesIntersection(self, line)
        if point is not None:
            point_projection1, _ = self.point_projection(point)
            if point_projection1 is None:
                return []

            if line.__class__.__name__ == 'LineSegment2D':
                point_projection2, _ = line.point_projection(point)
                if point_projection2 is None:
                    return []

            return [point_projection1]
        else:
            return []

    def plot(self, ax=None, color='k', arrow=False, width=None,
                plot_points=False):
        if ax is None:
            fig, ax = plt.subplots()
            # ax.set_aspect('equal')
        # else:
        #     fig = ax.figure

        p1, p2 = self.start, self.end
        if arrow:
            if plot_points:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color,
                        style='o-')
            else:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color)

            length = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
            if width is None:
                width = length / 1000.
                head_length = length / 20.
                head_width = head_length / 2.
            else:
                head_width = 2 * width
                head_length = head_width
            ax.arrow(p1[0], p1[1],
                     (p2[0] - p1[0]) / length * (length - head_length),
                     (p2[1] - p1[1]) / length * (length - head_length),
                     head_width=head_width, fc='b', linewidth=0,
                     head_length=head_length, width=width, alpha=0.3)
        else:
            if width is None:
                width = 1
            if plot_points:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color,
                        marker='o', linewidth=width)
            else:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color,
                        linewidth=width)
        return ax

    def to_3d(self, plane_origin, x1, x2):
        start = self.start.to_3d(plane_origin, x1, x2)
        end = self.end.to_3d(plane_origin, x1, x2)
        return LineSegment3D(start, end, self.name)

    def reverse(self):
        return LineSegment2D(self.end.copy(), self.points[0].copy())

    def to_line(self):
        return Line2D(*self.points)

    def rotation(self, center, angle, copy=True):
        if copy:
            return LineSegment2D(
                *[p.rotation(center, angle, copy=True) for p in self.points])
        else:
            for p in self.points:
                p.rotation(center, angle, copy=False)

    def translation(self, offset, copy=True):
        if copy:
            return LineSegment2D(
                *[p.translation(offset, copy=True) for p in self.points])
        else:
            for p in self.points:
                p.translation(offset, copy=False)

    def frame_mapping(self, frame, side, copy=True):
        """
        side = 'old' or 'new'
        """
        if side == 'old':
            if copy:
                return LineSegment2D(
                    *[frame.OldCoordinates(p) for p in self.points])
            else:
                self.points = [frame.OldCoordinates(p) for p in self.points]
        if side == 'new':
            if copy:
                return LineSegment2D(
                    *[frame.NewCoordinates(p) for p in self.points])
            else:
                self.points = [frame.NewCoordinates(p) for p in self.points]

    def plot_data(self, marker=None, color='black', stroke_width=1,
                  dash=False, opacity=1, arrow=False):
        return {'type': 'line',
                'data': [self.points[0].vector[0], self.points[0].vector[1],
                         self.end.vector[0], self.end.vector[1]],
                'color': color,
                'marker': marker,
                'size': stroke_width,
                'dash': dash,
                'opacity': opacity,
                'arrow': arrow
                }

    def CreateTangentCircle(self, point, other_line):
        circle1, circle2 = Line2D.CreateTangentCircle(other_line, point, self)
        if circle1 is not None:
            point_J1, curv_abs1 = Line2D.point_projection(self, circle1.center)
            if curv_abs1 < 0. or curv_abs1 > self.length():
                circle1 = None
        if circle2 is not None:
            point_J2, curv_abs2 = Line2D.point_projection(self, circle2.center)
            if curv_abs2 < 0. or curv_abs2 > self.length():
                circle2 = None
        return circle1, circle2

    def tessellation_points(self):
        return [self.start, self.end]

    def polygon_points(self, min_x_density=None, min_y_density=None):
        n = 0  # Number of points to insert between start and end
        if min_x_density:
            dx = abs(self.start[0] - self.end[0])
            n = max(n, math.floor(dx * min_x_density))
        if min_y_density:
            dy = abs(self.start[1] - self.end[1])
            n = max(n, math.floor(dy * min_y_density))

        if n:
            l = self.length()
            return [self.point_at_abscissa(i * l / (n + 1)) for i in
                    range(n + 2)]
        else:
            return [self.start, self.end]


class Arc2D(Edge):
    """
    angle: the angle measure always >= 0
    """

    def __init__(self,
                 start: volmdlr.Point2D,
                 interior: volmdlr.Point2D,
                 end: volmdlr.Point2D,
                 name:str=''):

        Edge.__init__(self, start=start, end=end, name=name)
        self.interior = interior
        xi, yi = interior.x, interior.y
        xe, ye = end.x, end.y
        xs, ys = start.x, start.y
        try:
            A = volmdlr.core.Matrix22(2 * (xs - xi), 2 * (ys - yi),
                                 2 * (xs - xe), 2 * (ys - ye))
            b = - volmdlr.Vector2D(xi ** 2 + yi ** 2 - xs ** 2 - ys ** 2,
                                   xe ** 2 + ye ** 2 - xs ** 2 - ys ** 2)
            inv_A = A.inverse()
            x = inv_A.vector_multiplication(b)
            self.center = volmdlr.Point2D(x.x, x.y)
        except ValueError:
            A = npy.array([[2 * (xs - xi), 2 * (ys - yi)],
                           [2 * (xs - xe), 2 * (ys - ye)]])
            b = - npy.array([xi ** 2 + yi ** 2 - xs ** 2 - ys ** 2,
                             xe ** 2 + ye ** 2 - xs ** 2 - ys ** 2])
            self.center = Point2D(solve(A, b))

        r1 = self.start - self.center
        r2 = self.end - self.center
        ri = self.interior - self.center

        self.radius = r1.norm()
        angle1 = math.atan2(r1.y, r1.x)
        anglei = math.atan2(ri.y, ri.x)
        angle2 = math.atan2(r2.y, r2.x)

        # Going trigo/clock wise from start to interior
        if anglei < angle1:
            trigowise_path = (anglei + volmdlr.TWO_PI) - angle1
            clockwise_path = angle1 - anglei
        else:
            trigowise_path = anglei - angle1
            clockwise_path = angle1 - anglei + volmdlr.TWO_PI

        # Going trigo wise from interior to interior
        if angle2 < anglei:
            trigowise_path += (angle2 + volmdlr.TWO_PI) - anglei
            clockwise_path += anglei - angle2
        else:
            trigowise_path += angle2 - anglei
            clockwise_path += anglei - angle2 + volmdlr.TWO_PI

        if clockwise_path > trigowise_path:
            self.is_trigo = True
            self.angle1 = angle1
            self.angle2 = angle2
            self.angle = trigowise_path
        else:
            # Clock wise
            self.is_trigo = False
            self.angle1 = angle2
            self.angle2 = angle1
            self.angle = clockwise_path

    def _get_points(self):
        return [self.start, self.interior, self.end]

    points = property(_get_points)

    def tessellation_points(self, resolution_for_circle=40):
        number_points_tesselation = math.ceil(
            resolution_for_circle * abs(self.angle) / 2 / math.pi)
        number_points_tesselation = max(number_points_tesselation, 5)
        l = self.length()
        return [self.point_at_abscissa(
            i / (number_points_tesselation - 1) * l) for i in
            range(number_points_tesselation)]

    def point_belongs(self, point):
        """
        Computes if the point belongs to the pizza slice drawn by the arc and its center
        """
        circle = volmdlr.wires.Circle2D(self.center, self.radius)
        if not circle.point_belongs(point):
            return False
        vector_start = self.start - self.center
        vector_point = point - self.center
        vector_end = self.end - self.center
        if self.is_trigo:
            vector_start, vector_end = vector_end, vector_start
        arc_angle = volmdlr.clockwise_angle(vector_start, vector_end)
        point_angle = volmdlr.clockwise_angle(vector_start, vector_point)
        if point_angle <= arc_angle:
            return True

    def point_distance(self, point):
        vector_start = self.start - self.center
        vector_point = point - self.center
        vector_end = self.end - self.center
        if self.is_trigo:
            vector_start, vector_end = vector_end, vector_start
        arc_angle = volmdlr.clockwise_angle(vector_start, vector_end)
        point_angle = volmdlr.clockwise_angle(vector_start, vector_point)
        if point_angle <= arc_angle:
            return abs(
                LineSegment2D(point, self.center).length() - self.radius)
        else:
            return min(LineSegment2D(point, self.start).length(),
                       LineSegment2D(point, self.end).length())

    def line_intersections(self, line):
        circle = volmdlr.wires.Circle2D(self.center, self.radius)
        circle_intersection_points = circle.line_intersections(line)

        if circle_intersection_points is None:
            return None

        intersection_points = []
        for pt in circle_intersection_points:
            if self.point_belongs(pt):
                intersection_points.append(pt)
        return intersection_points

    def length(self):
        return self.radius * abs(self.angle)

    def point_at_abscissa(self, curvilinear_abscissa):
        if self.is_trigo:
            return self.start.rotation(self.center,
                                       curvilinear_abscissa / self.radius)
            # return self.start.rotation(self.center, curvilinear_abscissa*self.angle)
        else:
            return self.start.rotation(self.center,
                                       -curvilinear_abscissa / self.radius)
            # return self.start.rotation(self.center, -curvilinear_abscissa*self.angle)

    def middle_point(self):
        l = self.length()
        return self.point_at_abscissa(0.5 * l)

    def area(self):
        if self.angle2 < self.angle1:
            angle = self.angle2 + volmdlr.TWO_PI - self.angle1
        else:
            angle = self.angle2 - self.angle1
        return self.radius ** 2 * angle / 2

    def center_of_mass(self):
        #        u=self.middle.vector-self.center.vector
        u = self.middle_point() - self.center
        u.normalize()
        alpha = abs(self.angle)
        return self.center + 4 / (3 * alpha) * self.radius * math.sin(
            alpha * 0.5) * u

    def plot(self, ax=None, color='k', plot_points=False):
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')

        if plot_points:
            for p in [self.center, self.start, self.interior, self.end]:
                p.plot(ax=ax)

        ax.add_patch(matplotlib.patches.Arc(self.center, 2 * self.radius, 2 * self.radius, angle=0,
                                 theta1=self.angle1 * 0.5 / math.pi * 360,
                                 theta2=self.angle2 * 0.5 / math.pi * 360,
                                 color=color))

        return ax

    def to_3d(self, plane_origin, x, y):
        ps = self.start.to_3d(plane_origin, x, y)
        pi = self.interior.to_3d(plane_origin, x, y)
        pe = self.end.to_3d(plane_origin, x, y)

        return volmdlr.edges.Arc3D(ps, pi, pe, name=self.name)

    def rotation(self, center, angle, copy=True):
        if copy:
            return Arc2D(*[p.rotation(center, angle, copy=True) for p in
                           [self.start, self.interior, self.end]])
        else:
            self.__init__(*[p.rotation(center, angle, copy=True) for p in
                            [self.start, self.interior, self.end]])

    def translation(self, offset, copy=True):
        if copy:
            return Arc2D(*[p.translation(offset, copy=True) for p in
                           [self.start, self.interior, self.end]])
        else:
            self.__init__(*[p.translation(offset, copy=True) for p in
                            [self.start, self.interior, self.end]])

    def frame_mapping(self, frame, side, copy=True):
        """
        side = 'old' or 'new'
        """
        if copy:
            return Arc2D(*[p.frame_mapping(frame, side, copy=True) for p in
                           [self.start, self.interior, self.end]])
        else:
            self.__init__(*[p.frame_mapping(frame, side, copy=True) for p in
                            [self.start, self.interior, self.end]])

    def SecondMomentArea(self, point):
        """
        Second moment area of part of disk
        """
        if self.angle2 < self.angle1:
            angle2 = self.angle2 + volmdlr.volmdlr.TWO_PI

        else:
            angle2 = self.angle2
        angle1 = self.angle1

        Ix = self.radius ** 4 / 8 * (angle2 - angle1 + 0.5 * (
                math.sin(2 * angle1) - math.sin(2 * angle2)))
        Iy = self.radius ** 4 / 8 * (angle2 - angle1 + 0.5 * (
                math.sin(2 * angle2) - math.sin(2 * angle1)))
        Ixy = self.radius ** 4 / 8 * (
                math.cos(angle1) ** 2 - math.cos(angle2) ** 2)
        Ic = npy.array([[Ix, Ixy], [Ixy, Iy]])
        return volmdlr.geometry.Huygens2D(Ic, self.Area(), self.center, point)

    def Discretise(self, num=10):
        list_node = []
        if (self.angle1 < 0) and (self.angle2 > 0):
            delta_angle = -self.angle1 + self.angle2
        elif (self.angle1 > 0) and (self.angle2 < 0):
            delta_angle = (2 * npy.pi + self.angle2) - self.angle1
        else:
            delta_angle = self.angle2 - self.angle1
        for angle in npy.arange(self.angle1, self.angle1 + delta_angle,
                                delta_angle / (num * 1.)):
            list_node.append(volmdlr.Point2D(self.center + self.radius * volmdlr.Vector2D(
                (npy.cos(angle), npy.sin(angle)))))
        list_node.append(volmdlr.Point2D(self.center + self.radius * volmdlr.Vector2D((npy.cos(
            self.angle1 + delta_angle), npy.sin(self.angle1 + delta_angle)))))
        if list_node[0] == self.start:
            return list_node
        else:
            return list_node[::-1]

    def plot_data(self, marker=None, color='black', stroke_width=1, dash=False,
                  opacity=1):
        list_node = self.Discretise()
        data = []
        for nd in list_node:
            data.append({'x': nd.vector[0], 'y': nd.vector[1]})
        return {'type': 'arc',
                'cx': self.center.vector[0],
                'cy': self.center.vector[1],
                'data': data,
                'r': self.radius,
                'color': color,
                'opacity': opacity,
                'size': stroke_width,
                'dash': None,
                'marker': marker,
                'angle1': self.angle1,
                'angle2': self.angle2, }

    def copy(self):
        return Arc2D(self.start.copy(),
                     self.interior.copy(),
                     self.end.copy())

    def split(self, split_point: volmdlr.Point2D):
        raise NotImplementedError
        return [Arc2D(self.start, self.split_point)]

    def polygon_points(self, points_per_radian=10, min_x_density=None,
                       min_y_density=None):

        number_points = math.ceil(self.angle * points_per_radian)
        densities = []
        for d in [min_x_density, min_y_density]:
            if d:
                densities.append(d)
        if densities:
            number_points = max(number_points,
                                min(densities) * self.angle * self.radius)
        l = self.length()
        return [self.point_at_abscissa(i * l / number_points) \
                for i in range(number_points + 1)]


class ArcEllipse2D(Edge):
    """

    """

    def __init__(self, start, interior, end, center, major_dir, name='',
                 extra=None):
        Edge.__init__(self, start, end, name)
        self.interior = interior
        self.center = center
        self.extra = extra
        self.major_dir = major_dir
        self.minor_dir = self.major_dir.deterministic_unit_normal_vector()

        frame = volmdlr.Frame2D(self.center, self.major_dir, self.minor_dir)
        start_new, end_new = frame.NewCoordinates(
            self.start), frame.NewCoordinates(self.end)
        interior_new, center_new = frame.NewCoordinates(
            self.interior), frame.NewCoordinates(self.center)

        #### from : https://math.stackexchange.com/questions/339126/how-to-draw-an-ellipse-if-a-center-and-3-arbitrary-points-on-it-are-given
        def theta_A_B(s, i, e,
                      c):  # theta=angle d'inclinaison ellipse par rapport à horizontal(sens horaire),A=demi grd axe, B=demi petit axe
            xs, ys, xi, yi, xe, ye = s[0] - c[0], s[1] - c[1], i[0] - c[0], i[
                1] - c[1], e[0] - c[0], e[1] - c[1]
            A = npy.array(([xs ** 2, ys ** 2, 2 * xs * ys],
                           [xi ** 2, yi ** 2, 2 * xi * yi],
                           [xe ** 2, ye ** 2, 2 * xe * ye]))
            invA = npy.linalg.inv(A)
            One = npy.array(([1],
                             [1],
                             [1]))
            C = npy.dot(invA, One)  # matrice colonne de taille 3
            theta = 0
            c1 = C[0] + C[1]
            c2 = (C[1] - C[0]) / math.cos(2 * theta)
            gdaxe = math.sqrt((2 / (c1 - c2)))
            ptax = math.sqrt((2 / (c1 + c2)))
            return theta, gdaxe, ptax

        if start == end:
            extra_new = frame.NewCoordinates(self.extra)
            theta, A, B = theta_A_B(start_new, extra_new, interior_new,
                                    center_new)
        else:
            theta, A, B = theta_A_B(start_new, interior_new, end_new,
                                    center_new)

        self.Gradius = A
        self.Sradius = B
        self.theta = theta

        # Angle pour start
        u1, u2 = start_new.vector[0] / self.Gradius, start_new.vector[
            1] / self.Sradius
        angle1 = volmdlr.core.sin_cos_angle(u1, u2)
        # Angle pour end
        u3, u4 = end_new.vector[0] / self.Gradius, end_new.vector[
            1] / self.Sradius
        angle2 = volmdlr.core.sin_cos_angle(u3, u4)
        # Angle pour interior
        u5, u6 = interior_new.vector[0] / self.Gradius, interior_new.vector[
            1] / self.Sradius
        anglei = volmdlr.core.sin_cos_angle(u5, u6)

        # Going trigo/clock wise from start to interior
        if anglei < angle1:
            trigowise_path = (anglei + volmdlr.TWO_PI) - angle1
            clockwise_path = angle1 - anglei
        else:
            trigowise_path = anglei - angle1
            clockwise_path = angle1 - anglei + volmdlr.TWO_PI

        # Going trigo wise from interior to interior
        if angle2 < anglei:
            trigowise_path += (angle2 + volmdlr.TWO_PI) - anglei
            clockwise_path += anglei - angle2
        else:
            trigowise_path += angle2 - anglei
            clockwise_path += anglei - angle2 + volmdlr.TWO_PI

        if clockwise_path > trigowise_path:
            self.is_trigo = True
            self.angle = trigowise_path
        else:
            # Clock wise
            self.is_trigo = False
            self.angle = clockwise_path

        if self.start == self.end or self.angle == 0:
            self.angle = volmdlr.volmdlr.TWO_PI

        if self.is_trigo:  # sens trigo
            self.offset_angle = angle1
        else:
            self.offset_angle = angle2


    def _get_points(self):
        return self.tessellation_points()

    points = property(_get_points)

    def tessellation_points(self, resolution_for_ellipse=40):
        number_points_tesselation = math.ceil(
            resolution_for_ellipse * abs(0.5 * self.angle / math.pi))

        frame2d = volmdlr.Frame2D(self.center, self.major_dir, self.minor_dir)

        tessellation_points_2D = [(volmdlr.Point2D((self.Gradius * math.cos(
            self.offset_angle + self.angle * i / (number_points_tesselation)),
                                            self.Sradius * math.sin(
                                                self.offset_angle + self.angle * i / (
                                                    number_points_tesselation)))))
                                  for i in
                                  range(number_points_tesselation + 1)]

        global_points = []
        for pt in tessellation_points_2D:
            global_points.append(frame2d.OldCoordinates(pt))

        return global_points

    def to_3d(self, plane_origin, x, y):
        ps = self.start.to_3d(plane_origin, x, y)
        pi = self.interior.to_3d(plane_origin, x, y)
        pe = self.end.to_3d(plane_origin, x, y)
        pc = self.center.to_3d(plane_origin, x, y)
        if self.extra is None:
            pextra = None
        else:
            pextra = self.extra.to_3d(plane_origin, x, y)
        if ps == pe:
            p3 = pextra
        else:
            p3 = pe
        plane = volmdlr.faces.Plane3D.from_3_points(ps, pi, p3)
        n = plane.normal
        major_dir = self.major_dir.to_3d(plane_origin, x, y)
        major_dir.normalize()

        return ArcEllipse3D(ps, pi, pe, pc, major_dir, normal=n,
                            name=self.name, extra=pextra)

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
        else:
            fig = ax.figure

        self.interior.plot(ax=ax, color='m')
        self.start.plot(ax=ax, color='r')
        self.end.plot(ax=ax, color='b')
        self.center.plot(ax=ax, color='y')

        x = []
        y = []
        for px, py in self.tessellation_points():
            x.append(px)
            y.append(py)

        plt.plot(x, y, 'k')
        return fig, ax



class Line3D(Line):
    _non_eq_attributes = ['name', 'basis_primitives', 'bounding_box']

    """
    Define an infinite line passing through the 2 points
    """

    def __init__(self, point1: volmdlr.Point3D, point2: volmdlr.Point3D,
                 name: str=''):
        Line.__init__(self, point1, point2, name=name)
        self.bounding_box = self._bounding_box()


    def _bounding_box(self):
        points = [self.point1, self.point2]

        xmin = min([pt[0] for pt in points])
        xmax = max([pt[0] for pt in points])
        ymin = min([pt[1] for pt in points])
        ymax = max([pt[1] for pt in points])
        zmin = min([pt[2] for pt in points])
        zmax = max([pt[2] for pt in points])

        return volmdlr.core.BoundingBox(xmin, xmax, ymin, ymax, zmin, zmax)

    def point_at_abscissa(self, curvilinear_abscissa):
        return self.point1 + (
                self.point2 - self.point1) * curvilinear_abscissa

    def plot(self, ax=None, color='k', dashed=True):
        if ax is None:
            fig = plt.figure()
            ax = Axes3D(fig)
        else:
            fig = ax.figure

        # Line segment
        x = [self.point1.x, self.point2.x]
        y = [self.point1.y, self.point2.y]
        z = [self.point1.z, self.point2.z]
        ax.plot(x, y, z, 'ok')

        # Drawing 3 times length of segment on each side
        u = self.point2 - self.points[0]
        x1, y1, z1 = (self.points[0] - 3 * u).vector
        x2, y2, z2 = (self.point2 + 3 * u).vector
        if dashed:
            ax.plot([x1, x2], [y1, y2], [z1, z2], color=color,
                    dashes=[30, 5, 10, 5])
        else:
            ax.plot([x1, x2], [y1, y2], [z1, z2], color=color)
        return ax

    def PlaneProjection2D(self, center, x, y):
        return Line2D(self.points[0].PlaneProjection2D(center, x, y),
                      self.point2.PlaneProjection2D(center, x, y))

    def MinimumDistancePoints(self, other_line):
        """
        Returns the points on this line and the other line that are the closest
        of lines
        """
        u = self.point2 - self.point1
        v = other_line.point2 - other_line.point1
        w = self.point1 - other_line.point1
        a = u.dot(u)
        b = u.dot(v)
        c = v.dot(v)
        d = u.dot(w)
        e = v.dot(w)

        s = (b * e - c * d) / (a * c - b ** 2)
        t = (a * e - b * d) / (a * c - b ** 2)
        p1 = self.point1 + s * u
        p2 = other_line.point1 + t * v
        return p1, p2

    def rotation(self, center, axis, angle, copy=True):
        if copy:
            return Line3D(*[p.rotation(center, axis, angle, copy=True) for p in
                            self.points])
        else:
            for p in self.points:
                p.rotation(center, axis, angle, copy=False)

    def translation(self, offset, copy=True):
        if copy:
            return Line3D(
                *[p.translation(offset, copy=True) for p in self.points])
        else:
            for p in self.points:
                p.translation(offset, copy=False)

    def frame_mapping(self, frame, side, copy=True):
        """
        side = 'old' or 'new'
        """
        if side == 'old':
            if copy:
                return Line3D(*[frame.OldCoordinates(p) for p in self.points])
            else:
                for p in self.points:
                    self.points = [frame.OldCoordinates(p) for p in
                                   self.points]
        if side == 'new':
            if copy:
                return Line3D(*[frame.NewCoordinates(p) for p in self.points])
            else:
                for p in self.points:
                    self.points = [frame.NewCoordinates(p) for p in
                                   self.points]

    def copy(self):
        return Line3D(*[p.copy() for p in self.points])

    @classmethod
    def from_step(cls, arguments, object_dict):
        point1 = object_dict[arguments[1]]
        direction = object_dict[arguments[2]]
        point2 = point1 + direction
        return cls(point1, point2, arguments[0][1:-1])

    def Intersection(self, line2):

        x1 = self.point1.vector[0]
        y1 = self.point1.vector[1]
        z1 = self.point1.vector[2]
        x2 = self.point2.vector[0]
        y2 = self.point2.vector[1]
        z2 = self.point2.vector[2]
        x3 = line2.points[0].vector[0]
        y3 = line2.points[0].vector[1]
        z3 = line2.points[0].vector[2]
        x4 = line2.point2.vector[0]
        y4 = line2.point2.vector[1]
        z4 = line2.point2.vector[2]

        if x3 == 0 and x4 == 0 and y4 - y3 == 0:
            x5, y5, z5 = x3, y3, z3
            x6, y6, z6 = x4, y4, z4
            x3, y3, z3 = x1, y1, z1
            x4, y4, z4 = x2, y2, z2
            x1, y1, z1 = x5, y5, z5
            x2, y2, z2 = x6, y6, z6

        elif y3 == 0 and y4 == 0 and x4 - x3 == 0:
            x5, y5, z5 = x3, y3, z3
            x6, y6, z6 = x4, y4, z4
            x3, y3, z3 = x1, y1, z1
            x4, y4, z4 = x2, y2, z2
            x1, y1, z1 = x5, y5, z5
            x2, y2, z2 = x6, y6, z6

        res, list_t1 = [], []

        # 2 unknown 3eq with t1 et t2 unknown

        if (x2 - x1 + y1 - y2) != 0 and (y4 - y3) != 0:
            t1 = (x3 - x1 + (x4 - x3) * (y1 - y3) / (y4 - y3)) / (
                        x2 - x1 + y1 - y2)
            t2 = (y1 - y3 + (y2 - y1) * t1) / (y4 - y3)
            res1 = z1 + (z2 - z1) * t1
            res2 = z3 + (z4 - z3) * t2
            list_t1.append(t1)
            res.append([res1, res2])

        if (z2 - z1 + y1 - y2) != 0 and (y4 - y3) != 0:
            t1 = (z3 - z1 + (z4 - z3) * (y1 - y3) / (y4 - y3)) / (
                        z2 - z1 + y1 - y2)
            t2 = (y1 - y3 + (y2 - y1) * t1) / (y4 - y3)
            res1 = x1 + (x2 - x1) * t1
            res2 = x3 + (x4 - x3) * t2
            list_t1.append(t1)
            res.append([res1, res2])

        if (z2 - z1 + x1 - x2) != 0 and (x4 - x3) != 0:
            t1 = (z3 - z1 + (z4 - z3) * (x1 - x3) / (x4 - x3)) / (
                        z2 - z1 + x1 - x2)
            t2 = (x1 - x3 + (x2 - x1) * t1) / (x4 - x3)
            res1 = y1 + (y2 - y1) * t1
            res2 = y3 + (y4 - y3) * t2
            list_t1.append(t1)
            res.append([res1, res2])

        if len(res) == 0:
            return None

        for pair, t1 in zip(res, list_t1):
            res1, res2 = pair[0], pair[1]
            if math.isclose(res1, res2,
                            abs_tol=1e-7):  # if there is an intersection point
                return volmdlr.Point3D([x1 + (x2 - x1) * t1, y1 + (y2 - y1) * t1,
                                z1 + (z2 - z1) * t1])

        return None

class LineSegment3D(LineSegment):
    """
    Define a line segment limited by two points
    """

    def __init__(self, start, end, name=''):
        LineSegment.__init__(self, start, end, name)
        self.bounding_box = self._bounding_box()

    def __hash__(self):
        return 2 + hash(self.start) + hash(self.end)

    def __eq__(self, other_linesegment3d):
        if other_linesegment3d.__class__ != self.__class__:
            return False
        return

    def _bounding_box(self):
        points = [self.start, self.end]

        xmin = min([pt[0] for pt in points])
        xmax = max([pt[0] for pt in points])
        ymin = min([pt[1] for pt in points])
        ymax = max([pt[1] for pt in points])
        zmin = min([pt[2] for pt in points])
        zmax = max([pt[2] for pt in points])

        return volmdlr.core.BoundingBox(xmin, xmax, ymin, ymax, zmin, zmax)


    def length(self):
        return self.end.point_distance(self.start)

    def point_at_abscissa(self, curvilinear_abscissa):
        return self.start + curvilinear_abscissa * (
                    self.end - self.start) / self.length()

    def frenet(self, curvilinear_abscissa):
        return self.unit_direction_vector(), None

    def middle_point(self):
        l = self.length()
        return self.point_at_abscissa(0.5 * l)

    def PlaneProjection2D(self, center, x, y):
        return LineSegment2D(self.start.PlaneProjection2D(center, x, y),
                             self.point2.PlaneProjection2D(center, x, y))

    def Intersection(self, segment2):
        x1 = self.start.vector[0]
        y1 = self.start.vector[1]
        z1 = self.start.vector[2]
        x2 = self.point2.vector[0]
        y2 = self.point2.vector[1]
        z2 = self.point2.vector[2]
        x3 = segment2.start.vector[0]
        y3 = segment2.start.vector[1]
        z3 = segment2.start.vector[2]
        x4 = segment2.end_point.vector[0]
        y4 = segment2.end_point.vector[1]
        z4 = segment2.end_point.vector[2]

        if x3 == 0 and x4 == 0 and y4 - y3 == 0:
            x5, y5, z5 = x3, y3, z3
            x6, y6, z6 = x4, y4, z4
            x3, y3, z3 = x1, y1, z1
            x4, y4, z4 = x2, y2, z2
            x1, y1, z1 = x5, y5, z5
            x2, y2, z2 = x6, y6, z6

        elif y3 == 0 and y4 == 0 and x4 - x3 == 0:
            x5, y5, z5 = x3, y3, z3
            x6, y6, z6 = x4, y4, z4
            x3, y3, z3 = x1, y1, z1
            x4, y4, z4 = x2, y2, z2
            x1, y1, z1 = x5, y5, z5
            x2, y2, z2 = x6, y6, z6

        res, list_t1 = [], []

        # 2 unknown 3eq with t1 et t2 unknown
        if (x2 - x1 + y1 - y2) != 0 and (y4 - y3) != 0:
            t1 = (x3 - x1 + (x4 - x3) * (y1 - y3) / (y4 - y3)) / (
                        x2 - x1 + y1 - y2)
            t2 = (y1 - y3 + (y2 - y1) * t1) / (y4 - y3)
            res1 = z1 + (z2 - z1) * t1
            res2 = z3 + (z4 - z3) * t2
            list_t1.append(t1)
            res.append([res1, res2])

        if (z2 - z1 + y1 - y2) != 0 and (y4 - y3) != 0:
            t1 = (z3 - z1 + (z4 - z3) * (y1 - y3) / (y4 - y3)) / (
                        z2 - z1 + y1 - y2)
            t2 = (y1 - y3 + (y2 - y1) * t1) / (y4 - y3)
            res1 = x1 + (x2 - x1) * t1
            res2 = x3 + (x4 - x3) * t2
            list_t1.append(t1)
            res.append([res1, res2])

        if (z2 - z1 + x1 - x2) != 0 and (x4 - x3) != 0:
            t1 = (z3 - z1 + (z4 - z3) * (x1 - x3) / (x4 - x3)) / (
                        z2 - z1 + x1 - x2)
            t2 = (x1 - x3 + (x2 - x1) * t1) / (x4 - x3)
            res1 = y1 + (y2 - y1) * t1
            res2 = y3 + (y4 - y3) * t2
            list_t1.append(t1)
            res.append([res1, res2])

        if len(res) == 0:
            return None

        for pair, t1 in zip(res, list_t1):
            res1, res2 = pair[0], pair[1]
            if math.isclose(res1, res2,
                            abs_tol=1e-7):  # if there is an intersection point
                if t1 >= 0 or t1 <= 1:
                    return volmdlr.Point3D([x1 + (x2 - x1) * t1, y1 + (y2 - y1) * t1,
                                    z1 + (z2 - z1) * t1])

        return None

    def rotation(self, center, axis, angle, copy=True):
        if copy:
            return LineSegment3D(
                *[p.rotation(center, axis, angle, copy=True) for p in
                  self.points])
        else:
            Edge.rotation(self, center, axis, angle, copy=False)
            self.bounding_box = self._bounding_box()

    def __contains__(self, point):
        point1, point2 = self.start, self.end
        axis = volmdlr.Vector3D(point2 - point1)
        test = point.rotation(point1, axis, math.pi)
        if test == point:
            return True
        else:
            return False

    def translation(self, offset, copy=True):
        if copy:
            return LineSegment3D(
                *[p.translation(offset, copy=True) for p in self.points])
        else:
            Edge.translation(self, offset, copy=False)
            self.bounding_box = self._bounding_box()

    def frame_mapping(self, frame, side, copy=True):
        """
        side = 'old' or 'new'
        """
        if side == 'old':
            if copy:
                return LineSegment3D(
                    *[frame.OldCoordinates(p) for p in self.points])
            else:
                Edge.frame_mapping(self, frame, side, copy=False)
                self.bounding_box = self._bounding_box()
        if side == 'new':
            if copy:
                return LineSegment3D(
                    *[frame.NewCoordinates(p) for p in self.points])
            else:
                Edge.frame_mapping(self, frame, side, copy=False)
                self.bounding_box = self._bounding_box()

    def copy(self):
        return LineSegment3D(self.start.copy(), self.end.copy())

    def plot(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure

        points = [self.start, self.end]
        x = [p.x for p in points]
        y = [p.y for p in points]
        z = [p.z for p in points]
        ax.plot(x, y, z, 'o-k')
        return ax

    def plot2D(self, x_3D, y_3D, ax=None, color='k', width=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure

        edge2D = self.PlaneProjection2D(volmdlr.O3D, x_3D, y_3D)
        edge2D.plot(ax=ax, color=color, width=width)
        return ax

    def plot_data(self, x_3D, y_3D, marker=None, color='black', stroke_width=1,
                  dash=False, opacity=1, arrow=False):
        edge2D = self.PlaneProjection2D(volmdlr.O3D, x_3D, y_3D)
        return edge2D.plot_data(marker, color, stroke_width,
                                dash, opacity, arrow)

    def FreeCADExport(self, name, ndigits=6):
        name = 'primitive' + str(name)
        x1, y1, z1 = round(1000 * self.start, ndigits)
        x2, y2, z2 = round(1000 * self.end, ndigits)
        return '{} = Part.LineSegment(fc.Vector({},{},{}),fc.Vector({},{},{}))\n'.format(
            name, x1, y1, z1, x2, y2, z2)

    def to_line(self):
        return Line3D(*self.points)

    def babylon_script(self, color=(1, 1, 1), name='line', type_='line',
                       parent=None):
        if type_ == 'line' or type_ == 'dashed':
            s = 'var myPoints = [];\n'
            s += 'var point1 = new BABYLON.Vector3({},{},{});\n'.format(
                *self.start)
            s += 'myPoints.push(point1);\n'
            s += 'var point2 = new BABYLON.Vector3({},{},{});\n'.format(
                *self.end)
            s += 'myPoints.push(point2);\n'
            if type_ == 'line':
                s += 'var {} = BABYLON.MeshBuilder.CreateLines("lines", {{points: myPoints}}, scene);\n'.format(
                    name)
            elif type_ == 'dashed':
                s += 'var {} = BABYLON.MeshBuilder.CreateDashedLines("lines", {{points: myPoints, dashNb:20}}, scene);'.format(
                    name)
            s += '{}.color = new BABYLON.Color3{};\n'.format(name,
                                                             tuple(color))
        elif type_ == 'tube':
            radius = 0.03 * self.start.point_distance(self.end)
            s = 'var points = [new BABYLON.Vector3({},{},{}), new BABYLON.Vector3({},{},{})];\n'.format(
                *self.start, *self.end)
            s += 'var {} = BABYLON.MeshBuilder.CreateTube("frame_U", {{path: points, radius: {}}}, {});'.format(
                name, radius, parent)
        #            s += 'line.material = red_material;\n'

        else:
            raise NotImplementedError

        if parent is not None:
            s += '{}.parent = {};\n'.format(name, parent)

        return s

    def to_2d(self, plane_origin, x1, x2):
        p2D = [p.to_2d(plane_origin, x1, x2) for p in (self.start, self.end)]
        return LineSegment2D(*p2D, name=self.name)

    def reverse(self):
        return LineSegment3D(self.end.copy(), self.start.copy())

    def MinimumDistancePoints(self, other_line):
        """
        Returns the points on this line and the other line that are the closest
        of lines
        """
        u = self.end - self.start
        v = other_line.end_point - other_line.start
        w = self.start - other_line.start
        a = u.dot(u)
        b = u.dot(v)
        c = v.dot(v)
        d = u.dot(w)
        e = v.dot(w)
        if (a * c - b ** 2) != 0:
            s = (b * e - c * d) / (a * c - b ** 2)
            t = (a * e - b * d) / (a * c - b ** 2)
            p1 = self.start + s * u
            p2 = other_line.start + t * v
            return p1, p2, s, t
        else:
            return None, None, -1, -1

    def Matrix_distance(self, other_line):
        u = self.direction_vector()
        v = other_line.direction_vector()
        w = other_line.start - self.start

        a = u.dot(u)
        b = -u.dot(v)
        d = v.dot(v)

        e = w.dot(u)
        f = -w.dot(v)

        A = npy.array([[a, b],
                       [b, d]])
        B = npy.array([e, f])

        res = scp.optimize.lsq_linear(A, B, bounds=(0, 1))
        p1 = self.point_at_abscissa(res.x[0] * self.length())
        p2 = other_line.point_at_abscissa(
            res.x[1] * other_line.length())
        return p1, p2

    def parallele_distance(self, other_linesegment):
        ptA, ptB, ptC = self.start, self.end, other_linesegment.points[0]
        u = volmdlr.Vector3D((ptA - ptB).vector)
        u.normalize()
        plane1 = volmdlr.faces.Plane3D.from_3_points(ptA, ptB, ptC)
        v = u.cross(plane1.normal)  # distance vector
        # ptA = k*u + c*v + ptC
        res = (ptA - ptC).vector
        x, y, z = res[0], res[1], res[2]
        u1, u2, u3 = u.vector[0], u.vector[1], u.vector[2]
        v1, v2, v3 = v.vector[0], v.vector[1], v.vector[2]

        if (u1 * v2 - v1 * u2) != 0 and u1 != 0:
            c = (y * u1 - x * u2) / (u1 * v2 - v1 * u2)
            k = (x - c * v1) / u1
            if math.isclose(k * u3 + c * v3, z, abs_tol=1e-7):
                return k
        elif (u1 * v3 - v1 * u3) != 0 and u1 != 0:
            c = (z * u1 - x * u3) / (u1 * v3 - v1 * u3)
            k = (x - c * v1) / u1
            if math.isclose(k * u2 + c * v2, y, abs_tol=1e-7):
                return k
        elif (v1 * u2 - v2 * u1) != 0 and u2 != 0:
            c = (u2 * x - y * u1) / (v1 * u2 - v2 * u1)
            k = (y - c * v2) / u2
            if math.isclose(k * u3 + c * v3, z, abs_tol=1e-7):
                return k
        elif (v3 * u2 - v2 * u3) != 0 and u2 != 0:
            c = (u2 * z - y * u3) / (v3 * u2 - v2 * u3)
            k = (y - c * v2) / u2
            if math.isclose(k * u1 + c * v1, x, abs_tol=1e-7):
                return k
        elif (u1 * v3 - v1 * u3) != 0 and u3 != 0:
            c = (z * u1 - x * u3) / (u1 * v3 - v1 * u3)
            k = (z - c * v3) / u3
            if math.isclose(k * u2 + c * v2, y, abs_tol=1e-7):
                return k
        elif (u2 * v3 - v2 * u3) != 0 and u3 != 0:
            c = (z * u2 - y * u3) / (u2 * v3 - v2 * u3)
            k = (z - c * v3) / u3
            if math.isclose(k * u1 + c * v1, x, abs_tol=1e-7):
                return k
        else:
            return NotImplementedError

    def minimum_distance(self, element, return_points=False):
        if element.__class__ is Arc3D or element.__class__ is volmdlr.wires.Circle3D:
            pt1, pt2 = element.minimum_distance_points_line(self)
            if return_points:
                return pt1.point_distance(pt2), pt1, pt2
            else:
                return pt1.point_distance(pt2)

        elif element.__class__ is LineSegment3D:
            p1, p2 = self.Matrix_distance(element)
            if return_points:
                return p1.point_distance(p2), p1, p2
            else:
                return p1.point_distance(p2)

        else:
            return NotImplementedError

    def extrusion(self, extrusion_vector):
        u = self.unit_direction_vector()
        v = extrusion_vector.copy()
        v.normalize()
        w = u.cross(v)
        l1 = self.length()
        l2 = extrusion_vector.norm()
        # outer_contour = Polygon2D([O2D, Point2D((l1, 0.)),
        #                            Point2D((l1, l2)), Point2D((0., l2))])
        plane = volmdlr.faces.Plane3D(volmdlr.Frame3D(self.start, u, v, w))
        return plane.rectangular_cut(0, l1, 0, l2)

    def revolution(self, axis_point, axis, angle):
        axis_line3d = Line3D(axis_point, axis_point + axis)

        p1_proj, _ = axis_line3d.point_projection(self.start)
        p2_proj, _ = axis_line3d.point_projection(self.end)
        d1 = self.start.point_distance(p1_proj)
        d2 = self.end.point_distance(p2_proj)
        if d1 != 0.:
            u = (self.start - p1_proj)  # Unit vector from p1_proj to p1
            u.normalize()
        elif d2 != 0.:
            u = (self.end - p2_proj)  # Unit vector from p1_proj to p1
            u.normalize()
        else:
            return None

        if u.is_colinear_to(self.direction_vector()):
            # Planar face
            v = axis.cross(u)
            surface = volmdlr.faces.Plane3D(volmdlr.Frame3D(p1_proj, u, v, axis))
            if angle == volmdlr.TWO_PI:
                # Only 2 circles as countours
                r, R = sorted([d1, d2])
                # v1 = axis.cross(u)
                outer_contour3d = volmdlr.wires.Circle3D(volmdlr.Frame3D(p1_proj, u, v, axis),
                                           R)
                if not math.isclose(r, 0, abs_tol=1e-9):
                    inner_contours3d = [volmdlr.wires.Circle3D(volmdlr.Frame3D(p1_proj, u, v, axis)
                                                 , r)]
                else:
                    inner_contours3d = []
            else:
                # Two arcs and lines
                arc1_i = self.end.rotation(axis=axis,
                                                 axis_point=axis_point,
                                                 angle=0.5 * angle)
                arc1_e = self.end.rotation(axis=axis,
                                           axis_point=axis_point,
                                           angle=angle)
                arc2_i = self.start.rotation(axis=axis,
                                                 axis_point=axis_point,
                                                 angle=0.5 * angle)
                arc2_s = self.start.rotation(axis=axis,
                                                 axis_point=axis_point,
                                                 angle=angle)

                arc1 = Arc3D(self.end, arc1_i, arc1_e)
                arc2 = Arc3D(arc2_s, arc1_i, self.start)
                line2 = LineSegment3D(arc1_e, arc2_s)
                outer_contour3d = volmdlr.wires.Contour3D([self, arc1, line2, arc2])
                inner_contours3d = []
            return volmdlr.faces.PlaneFace3D.from_contours3d(
                outer_contour3d=outer_contour3d,
                inner_contours3d=inner_contours3d)

        elif d1 != d2:
            # Conical
            v = axis.cross(u)
            w = axis.cross(v)
            u1 = self.direction_vector()
            semi_angle = math.asin(u1.cross(axis).norm())
            surface = volmdlr.surfaces.ConicalSurface3D(volmdlr.Frame3D(p1_proj, axis, v, w),
                                       semi_angle)
            return surface.rectangular_cut(0, self.length(), 0, angle)
        else:
            v = axis.cross(u)
            surface = volmdlr.faces.CylindricalSurface3D(volmdlr.Frame3D(p1_proj, u, v, axis), d1)
            return surface.rectangular_cut(0, angle,
                                           0, (self.end-self.start).dot(axis))


class BSplineCurve3D(volmdlr.core.Primitive3D):
    def __init__(self, degree, control_points, knot_multiplicities, knots,
                 weights=None, periodic=False, name=''):
        volmdlr.core.Primitive3D.__init__(self, basis_primitives=control_points, name=name)
        self.control_points = control_points
        self.degree = degree
        knots = volmdlr.core.standardize_knot_vector(knots)
        self.knots = knots
        self.knot_multiplicities = knot_multiplicities
        self.weights = weights
        self.periodic = periodic
        self.name = name

        curve = BSpline.Curve()
        curve.degree = degree
        if weights is None:
            P = [(control_points[i][0], control_points[i][1],
                  control_points[i][2]) for i in range(len(control_points))]
            curve.ctrlpts = P
        else:
            Pw = [(control_points[i][0] * weights[i],
                   control_points[i][1] * weights[i],
                   control_points[i][2] * weights[i], weights[i]) for i in
                  range(len(control_points))]
            curve.ctrlptsw = Pw
        knot_vector = []
        for i, knot in enumerate(knots):
            knot_vector.extend([knot] * knot_multiplicities[i])
        curve.knotvector = knot_vector
        curve.delta = 0.1
        curve_points = curve.evalpts

        self.curve = curve
        self.points = [volmdlr.Point3D((p[0], p[1], p[2])) for p in curve_points]

    def length(self):
        # Approximately
        length = 0
        for k in range(0, len(self.points) - 1):
            length += (self.points[k] - self.points[k + 1]).norm()
        return length

    def point_at_abscissa(self, curvilinear_abscissa):
        # copy paste from wire3D
        length = 0.
        primitives = []
        for k in range(0, len(self.points) - 1):
            primitives.append(
                LineSegment3D(self.points[k], self.points[k + 1]))
        for primitive in primitives:
            primitive_length = primitive.length()
            if length + primitive_length >= curvilinear_abscissa:
                return primitive.point_at_abscissa(
                    curvilinear_abscissa - length)
            length += primitive_length
        # Outside of length
        raise ValueError

    def FreeCADExport(self, ip, ndigits=3):
        name = 'primitive{}'.format(ip)
        points = '['
        for i in range(len(self.control_points)):
            point = 'fc.Vector({},{},{}),'.format(self.control_points[i][0],
                                                  self.control_points[i][1],
                                                  self.control_points[i][2])
            points += point
        points = points[:-1]
        points += ']'
        # !!! : A QUOI SERT LE DERNIER ARG DE BSplineCurve (False)?
        # LA MULTIPLICITE EN 3e ARG ET LES KNOTS EN 2e ARG ?
        return '{} = Part.BSplineCurve({},{},{},{},{},{},{})\n'.format(name,
                                                                       points,
                                                                       self.knot_multiplicities,
                                                                       self.knots,
                                                                       self.periodic,
                                                                       self.degree,
                                                                       self.weights,
                                                                       False)

    @classmethod
    def from_step(cls, arguments, object_dict):
        name = arguments[0][1:-1]
        degree = int(arguments[1])
        points = [object_dict[int(i[1:])] for i in arguments[2]]
        # curve_form = arguments[3]
        if arguments[4] == '.F.':
            closed_curve = False
        elif arguments[4] == '.T.':
            closed_curve = True
        else:
            raise ValueError
        # self_intersect = arguments[5]
        knot_multiplicities = [int(i) for i in arguments[6][1:-1].split(",")]
        knots = [float(i) for i in arguments[7][1:-1].split(",")]
        # knot_spec = arguments[8]
        knot_vector = []
        for i, knot in enumerate(knots):
            knot_vector.extend([knot] * knot_multiplicities[i])

        if 9 in range(len(arguments)):
            weight_data = [float(i) for i in arguments[9][1:-1].split(",")]
        else:
            weight_data = None

        # FORCING CLOSED_CURVE = FALSE:
        closed_curve = False
        return cls(degree, points, knot_multiplicities, knots, weight_data,
                   closed_curve, name)

    def point_distance(self, pt1):
        distances = []
        for point in self.points:
            #            vmpt = Point3D((point[1], point[2], point[3]))
            distances.append(pt1.point_distance(point))
        return min(distances)

    def rotation(self, center, axis, angle, copy=True):
        new_control_points = [p.rotation(center, axis, angle, True) for p in
                              self.control_points]
        new_BSplineCurve3D = BSplineCurve3D(self.degree, new_control_points,
                                            self.knot_multiplicities,
                                            self.knots, self.weights,
                                            self.periodic, self.name)
        if copy:
            return new_BSplineCurve3D
        else:
            self.control_points = new_control_points
            self.curve = new_BSplineCurve3D.curve
            self.points = new_BSplineCurve3D.points

    def translation(self, offset, copy=True):
        new_control_points = [p.translation(offset, True) for p in
                              self.control_points]
        new_BSplineCurve3D = BSplineCurve3D(self.degree, new_control_points,
                                            self.knot_multiplicities,
                                            self.knots, self.weights,
                                            self.periodic, self.name)
        if copy:
            return new_BSplineCurve3D
        else:
            self.control_points = new_control_points
            self.curve = new_BSplineCurve3D.curve
            self.points = new_BSplineCurve3D.points

    # Copy paste du LineSegment3D
    def plot(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure

        x = [p.vector[0] for p in self.points]
        y = [p.vector[1] for p in self.points]
        z = [p.vector[2] for p in self.points]
        ax.plot(x, y, z, 'o-k')
        return ax

    def to_2d(self, plane_origin, x1, x2):
        control_points2D = [p.to_2d(plane_origin, x1, x2) for p in
                            self.control_points]
        return BSplineCurve2D(self.degree, control_points2D,
                              self.knot_multiplicities, self.knots,
                              self.weights, self.periodic, self.name)

    def tessellation_points(self):
        return self.points


class Arc3D(Edge):
    """
    An arc is defined by a starting point, an end point and an interior point

    """

    def __init__(self, start, interior, end, name=''):
        """

        """
        self.interior = interior
        Edge.__init__(self, start=start, end=end, name=name)
        self.setup_arc(start, interior, end, name=name)

    @classmethod
    def from_angle(cls, start: volmdlr.Point3D, angle: float,
                   axis_point: volmdlr.Point3D, axis: volmdlr.Vector3D):
        start_gen = start
        int_gen = start_gen.rotation(axis_point, axis, angle / 2, copy=True)
        end_gen = start_gen.rotation(axis_point, axis, angle, copy=True)
        if angle == volmdlr.volmdlr.TWO_PI:
            line = Line3D(axis_point, axis_point + axis)
            center, _ = line.point_projection(start)
            radius = center.point_distance(start)
            u = start - center
            v = axis.cross(u)
            return volmdlr.wires.Circle3D(volmdlr.Frame3D(center, u, v, axis), radius)
        return cls(start_gen, int_gen, end_gen, axis)

    def setup_arc(self, start, interior, end, name=''):
        u1 = (self.interior - self.start)
        u2 = (self.interior - self.end)
        try:
            u1.normalize()
            u2.normalize()
        except ZeroDivisionError:
            raise ValueError(
                'Start, end and interior points of an arc must be distincts')

        self.normal = u2.cross(u1)
        self.normal.normalize()

        if u1 == u2:
            u2 = self.normal.cross(u1)
            u2.normalize()

        v1 = self.normal.cross(u1)  # v1 is normal, equal u2
        v2 = self.normal.cross(u2)  # equal -u1

        p11 = 0.5 * (start + interior)  # Mid point of segment s,m
        p12 = p11 + v1
        p21 = 0.5 * (end + interior)  # Mid point of segment s,m
        p22 = p21 + v2

        l1 = Line3D(p11, p12)
        l2 = Line3D(p21, p22)

        try:
            c1, _ = l1.MinimumDistancePoints(l2)
        except ZeroDivisionError:
            raise ValueError(
                'Start, end and interior points  of an arc must be distincts')

        self.center = c1
        self.radius = (self.center - self.start).norm()

        # Determining angle

        vec1 = (self.start - self.center)
        vec1.normalize()
        vec2 = self.normal.cross(vec1)

        r1 = self.start.to_2d(self.center, vec1, vec2)
        r2 = self.end.to_2d(self.center, vec1, vec2)
        ri = self.interior.to_2d(self.center, vec1, vec2)

        angle1 = math.atan2(r1.y, r1.x)
        anglei = math.atan2(ri.y, ri.x)
        angle2 = math.atan2(r2.y, r2.x)

        # Going trigo/clock wise from start to interior
        if anglei < angle1:
            trigowise_path = (anglei + volmdlr.TWO_PI) - angle1
            clockwise_path = angle1 - anglei
        else:
            trigowise_path = anglei - angle1
            clockwise_path = angle1 - anglei + volmdlr.TWO_PI

        # Going trigo wise from interior to interior
        if angle2 < anglei:
            trigowise_path += (angle2 + volmdlr.TWO_PI) - anglei
            clockwise_path += anglei - angle2
        else:
            trigowise_path += angle2 - anglei
            clockwise_path += anglei - angle2 + volmdlr.TWO_PI

        if clockwise_path > trigowise_path:
            self.is_trigo = True
            self.angle = trigowise_path
        else:
            # Clock wise
            self.is_trigo = False
            self.angle = clockwise_path

        if self.angle > math.pi:
            # Inverting normal to be sure to have a right defined normal for rotation
            self.normal = -self.normal

    @property
    def points(self):
        return [self.start, self.interior, self.end]

    def tessellation_points(self, resolution_for_circle=40):
        number_points_tesselation = resolution_for_circle
        l = self.length()
        tessellation_points_3D = [self.point_at_abscissa(
            l * i / (number_points_tesselation)) for i in
                                  range(number_points_tesselation + 1)]
        return tessellation_points_3D

    def length(self):
        return self.radius * abs(self.angle)

    def point_at_abscissa(self, curvilinear_abscissa):
        return self.start.rotation(self.center, self.normal,
                                   curvilinear_abscissa / self.radius,
                                   copy=True)

    def frenet(self, curvilinear_abscissa):
        theta = curvilinear_abscissa/self.radius
        t0 = self.normal.cross(self.start-self.center)
        t0.normalize()
        tangent = t0.rotation(self.center, self.normal, theta, copy=True)
        normal = -self.normal.cross(tangent)
        return tangent, normal

    def rotation(self, rot_center, axis, angle, copy=True):
        if copy:
            new_start = self.start.rotation(rot_center, axis, angle, True)
            new_interior = self.interior.rotation(rot_center, axis, angle,
                                                  True)
            new_end = self.end.rotation(rot_center, axis, angle, True)
            return Arc3D(new_start, new_interior, new_end, name=self.name)
        else:
            self.center.rotation(rot_center, axis, angle, False)
            self.start.rotation(rot_center, axis, angle, False)
            self.interior.rotation(rot_center, axis, angle, False)
            self.end.rotation(rot_center, axis, angle, False)
            [p.rotation(rot_center, axis, angle, False) for p in
             self.primitives]

    def translation(self, offset, copy=True):
        if copy:
            new_start = self.start.translation(offset, True)
            new_interior = self.interior.translation(offset, True)
            new_end = self.end.translation(offset, True)
            return Arc3D(new_start, new_interior, new_end, name=self.name)
        else:
            self.center.translation(offset, False)
            self.start.translation(offset, False)
            self.interior.translation(offset, False)
            self.end.translation(offset, False)
            [p.translation(offset, False) for p in self.primitives]

    def plot(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = Axes3D(fig)
        else:
            fig = None

        ax.plot([self.interior[0]], [self.interior[1]], [self.interior[2]],
                color='b')
        ax.plot([self.start[0]], [self.start[1]], [self.start[2]], c='r')
        ax.plot([self.end[0]], [self.end[1]], [self.end[2]], c='r')
        ax.plot([self.interior[0]], [self.interior[1]], [self.interior[2]],
                c='g')
        x = []
        y = []
        z = []
        for px, py, pz in self.tessellation_points():
            x.append(px)
            y.append(py)
            z.append(pz)

        ax.plot(x, y, z, 'k')
        return ax

    def plot2D(self, center=volmdlr.O3D,
                  x3d=volmdlr.X3D, y3D=volmdlr.Y3D,
                  ax=None, color='k'):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure

        # TODO: Enhance this plot
        l = self.length()
        x = []
        y = []
        for i in range(30):
            p = self.point_at_abscissa(i / (29.) * l)
            xi, yi = p.PlaneProjection2D(center, volmdlr.X3D, volmdlr.Y3D)
            x.append(xi)
            y.append(yi)
        ax.plot(x, y, color=color)

        return ax

    def FreeCADExport(self, name, ndigits=6):
        xs, ys, zs = round(1000 * self.start, ndigits)
        xi, yi, zi = round(1000 * self.interior, ndigits)
        xe, ye, ze = round(1000 * self.end, ndigits)
        return '{} = Part.Arc(fc.Vector({},{},{}),fc.Vector({},{},{}),fc.Vector({},{},{}))\n'\
            .format(name, xs, ys, zs, xi, yi, zi, xe, ye, ze)

    def copy(self):
        return Arc3D(self.start.copy(), self.interior.copy(), self.end.copy())

    def frame_mapping(self, frame, side, copy=True):
        """
        side = 'old' or 'new'
        """
        if side == 'old':
            new_start = frame.OldCoordinates(self.start.copy())
            new_interior = frame.OldCoordinates(self.interior.copy())
            new_end = frame.OldCoordinates(self.end.copy())
            if copy:
                return Arc3D(new_start, new_interior, new_end, normal=None,
                             name=self.name)
            else:
                self.start, self.interior, self.end = new_start, new_interior, new_end
                self.setup_arc(self.start, self.interior, self.end)

        if side == 'new':
            new_start = frame.NewCoordinates(self.start.copy())
            new_interior = frame.NewCoordinates(self.interior.copy())
            new_end = frame.NewCoordinates(self.end.copy())
            if copy:
                return Arc3D(new_start, new_interior, new_end, normal=None,
                             name=self.name)
            else:
                self.start, self.interior, self.end = new_start, new_interior, new_end
                self.setup_arc(self.start, self.interior, self.end)

    def to_2d(self, plane_origin, x, y):
        ps = self.start.to_2d(plane_origin, x, y)
        pi = self.interior.to_2d(plane_origin, x, y)
        pe = self.end.to_2d(plane_origin, x, y)
        return Arc2D(ps, pi, pe, name=self.name)

    def minimum_distance_points_arc(self, other_arc):

        u1 = self.start - self.center
        u1.normalize()
        u2 = self.normal.cross(u1)

        w = other_arc.center - self.center

        u3 = other_arc.start - other_arc.center
        u3.normalize()
        u4 = other_arc.normal.cross(u3)

        r1, r2 = self.radius, other_arc.radius

        a, b, c, d = u1.dot(u1), u1.dot(u2), u1.dot(u3), u1.dot(u4)
        e, f, g = u2.dot(u2), u2.dot(u3), u2.dot(u4)
        h, i = u3.dot(u3), u3.dot(u4)
        j = u4.dot(u4)
        k, l, m, n, o = w.dot(u1), w.dot(u2), w.dot(u3), w.dot(u4), w.dot(w)

        def distance_squared(x):
            return (a * ((math.cos(x[0])) ** 2) * r1 ** 2 + e * (
                        (math.sin(x[0])) ** 2) * r1 ** 2
                    + o + h * ((math.cos(x[1])) ** 2) * r2 ** 2 + j * (
                                (math.sin(x[1])) ** 2) * r2 ** 2
                    + b * math.sin(2 * x[0]) * r1 ** 2 - 2 * r1 * math.cos(
                        x[0]) * k
                    - 2 * r1 * r2 * math.cos(x[0]) * math.cos(x[1]) * c
                    - 2 * r1 * r2 * math.cos(x[0]) * math.sin(
                        x[1]) * d - 2 * r1 * math.sin(x[0]) * l
                    - 2 * r1 * r2 * math.sin(x[0]) * math.cos(x[1]) * f
                    - 2 * r1 * r2 * math.sin(x[0]) * math.sin(
                        x[1]) * g + 2 * r2 * math.cos(x[1]) * m
                    + 2 * r2 * math.sin(x[1]) * n + i * math.sin(
                        2 * x[1]) * r2 ** 2)

        x01 = npy.array([self.angle / 2, other_arc.angle / 2])

        res1 = scp.optimize.least_squares(distance_squared, x01,
                                          bounds=[(0, 0), (
                                              self.angle, other_arc.angle)])

        p1 = self.point_at_abscissa(res1.x[0] * r1)
        p2 = other_arc.point_at_abscissa(res1.x[1] * r2)

        return p1, p2

    def minimum_distance_points_line(self, other_line):

        u = other_line.DirectionVector()
        k = self.start - self.center
        k.normalize()
        w = self.center - other_line.points[0]
        v = self.normal.cross(k)

        r = self.radius

        a = u.dot(u)
        b = u.dot(v)
        c = u.dot(k)
        d = v.dot(v)
        e = v.dot(k)
        f = k.dot(k)
        g = w.dot(u)
        h = w.dot(v)
        i = w.dot(k)
        j = w.dot(w)

        # x = (s, theta)
        def distance_squared(x):
            return (a * x[0] ** 2 + j + d * (
                        (math.sin(x[1])) ** 2) * r ** 2 + f * (
                                (math.cos(x[1])) ** 2) * r ** 2
                    - 2 * x[0] * g - 2 * x[0] * r * math.sin(x[1]) * b - 2 * x[
                        0] * r * math.cos(x[1]) * c
                    + 2 * r * math.sin(x[1]) * h + 2 * r * math.cos(x[1]) * i
                    + math.sin(2 * x[1]) * e * r ** 2)

        x01 = npy.array([0.5, self.angle / 2])
        x02 = npy.array([0.5, 0])
        x03 = npy.array([0.5, self.angle])

        res1 = scp.optimize.least_squares(distance_squared, x01,
                                          bounds=[(0, 0), (1, self.angle)])
        res2 = scp.optimize.least_squares(distance_squared, x02,
                                          bounds=[(0, 0), (1, self.angle)])
        res3 = scp.optimize.least_squares(distance_squared, x03,
                                          bounds=[(0, 0), (1, self.angle)])

        p1 = other_line.point_at_abscissa(
            res1.x[0] * other_line.length())
        p2 = self.point_at_abscissa(res1.x[1] * r)

        res = [res2, res3]
        for couple in res:
            ptest1 = other_line.point_at_abscissa(
                couple.x[0] * other_line.length())
            ptest2 = self.point_at_abscissa(couple.x[1] * r)
            dtest = ptest1.point_distance(ptest2)
            if dtest < d:
                p1, p2 = ptest1, ptest2

        return p1, p2

    def minimum_distance(self, element, return_points=False):
        if element.__class__ is Arc3D or element.__class__ is Circle3D:
            p1, p2 = self.minimum_distance_points_arc(element)
            if return_points:
                return p1.point_distance(p2), p1, p2
            else:
                return p1.point_distance(p2)

        elif element.__class__ is LineSegment3D:
            pt1, pt2 = self.minimum_distance_points_line(element)
            if return_points:
                return pt1.point_distance(pt2), pt1, pt2
            else:
                return pt1.point_distance(pt2)
        else:
            return NotImplementedError

    def extrusion(self, extrusion_vector):
        if self.normal.is_colinear_to(extrusion_vector):
            u = self.start - self.center
            u.normalize()
            w = extrusion_vector.copy()
            w.normalize()
            v = w.cross(u)
            arc2d = self.to_2d(self.center, u, v)
            cylinder = volmdlr.faces.CylindricalSurface3D(volmdlr.Frame3D(self.center,
                                                    u,
                                                    v,
                                                    w),
                                            self.radius
                                            )
            return cylinder.rectangular_cut(arc2d.angle1,
                                            arc2d.angle2,
                                            0, extrusion_vector.norm())
        else:
            raise NotImplementedError('Elliptic faces not handled: dot={}'.format(
                self.normal.dot(extrusion_vector)
            ))


    def revolution(self, axis_point: volmdlr.Point3D, axis: volmdlr.Vector3D,
                   angle: float):
        line3d = Line3D(axis_point, axis_point + axis)
        tore_center, _ = line3d.point_projection(self.center)
        u =  self.center - tore_center
        u.normalize()
        v = axis.cross(u)
        if not math.isclose(self.normal.dot(u), 0., abs_tol=1e-9):
            raise NotImplementedError(
                'Outside of plane revolution not supported')

        R = tore_center.point_distance(self.center)
        surface = volmdlr.faces.ToroidalSurface3D(volmdlr.Frame3D(tore_center, u, v, axis), R,
                                    self.radius)
        arc2d = self.to_2d(tore_center, u, axis)
        return surface.rectangular_cut(0, angle,
                                       arc2d.angle1, arc2d.angle2)



class ArcEllipse3D(Edge):
    """
    An arc is defined by a starting point, an end point and an interior point

    """

    def __init__(self, start, interior, end, center, major_dir,
                 name=''):#, extra=None):
        # Extra is an additionnal point if start=end because you need 3 points on the arcellipse to define it
        Edge.__init__(self, start=start, end=end, name=name)
        self.interior = interior
        self.center = center
        major_dir.normalize()
        self.major_dir = major_dir  # Vector for Gradius
        # self.extra = extra

        u1 = (self.interior - self.start)
        u2 = (self.interior - self.end)
        u1.normalize()
        u2.normalize()

        if u1 == u2:
            u2 = (self.interior - self.extra)
            u2.normalize()

        # if normal is None:
        n = u2.cross(u1)
        n.normalize()
        self.normal = n
        # else:
        #     n = normal
        #     n.normalize()
        #     self.normal = normal

        self.minor_dir = self.normal.cross(self.major_dir)

        frame = volmdlr.Frame3D(self.center, self.major_dir, self.minor_dir,
                        self.normal)
        start_new, end_new = frame.NewCoordinates(
            self.start), frame.NewCoordinates(self.end)
        interior_new, center_new = frame.NewCoordinates(
            self.interior), frame.NewCoordinates(self.center)

        #### from : https://math.stackexchange.com/questions/339126/how-to-draw-an-ellipse-if-a-center-and-3-arbitrary-points-on-it-are-given
        def theta_A_B(s, i, e,
                      c):  # theta=angle d'inclinaison ellipse par rapport à horizontal(sens horaire),A=demi grd axe, B=demi petit axe
            xs, ys, xi, yi, xe, ye = s[0] - c[0], s[1] - c[1], i[0] - c[0], i[
                1] - c[1], e[0] - c[0], e[1] - c[1]
            A = npy.array(([xs ** 2, ys ** 2, 2 * xs * ys],
                           [xi ** 2, yi ** 2, 2 * xi * yi],
                           [xe ** 2, ye ** 2, 2 * xe * ye]))
            invA = npy.linalg.inv(A)
            One = npy.array(([1],
                             [1],
                             [1]))
            C = npy.dot(invA, One)  # matrice colonne de taille 3
            theta = 0.5 * math.atan(2 * C[2] / (C[1] - C[0]))
            c1 = C[0] + C[1]
            c2 = (C[1] - C[0]) / math.cos(2 * theta)
            gdaxe = math.sqrt((2 / (c1 - c2)))
            ptax = math.sqrt((2 / (c1 + c2)))
            return theta, gdaxe, ptax

        if start == end:
            extra_new = frame.NewCoordinates(self.extra)
            theta, A, B = theta_A_B(start_new, extra_new, interior_new,
                                    center_new)
        else:
            theta, A, B = theta_A_B(start_new, interior_new, end_new,
                                    center_new)

        self.Gradius = A
        self.Sradius = B
        self.theta = theta

        # Angle pour start
        u1, u2 = start_new.vector[0] / self.Gradius, start_new.vector[
            1] / self.Sradius
        angle1 = sin_cos_angle(u1, u2)
        # Angle pour end
        u3, u4 = end_new.vector[0] / self.Gradius, end_new.vector[
            1] / self.Sradius
        angle2 = sin_cos_angle(u3, u4)
        # Angle pour interior
        u5, u6 = interior_new.vector[0] / self.Gradius, interior_new.vector[
            1] / self.Sradius
        anglei = sin_cos_angle(u5, u6)

        # Going trigo/clock wise from start to interior
        if anglei < angle1:
            trigowise_path = (anglei + volmdlr.TWO_PI) - angle1
            clockwise_path = angle1 - anglei
        else:
            trigowise_path = anglei - angle1
            clockwise_path = angle1 - anglei + volmdlr.TWO_PI

        # Going trigo wise from interior to interior
        if angle2 < anglei:
            trigowise_path += (angle2 + volmdlr.TWO_PI) - anglei
            clockwise_path += anglei - angle2
        else:
            trigowise_path += angle2 - anglei
            clockwise_path += anglei - angle2 + volmdlr.TWO_PI

        if clockwise_path > trigowise_path:
            self.is_trigo = True
            self.angle = trigowise_path
        else:
            # Clock wise
            self.is_trigo = False
            self.angle = clockwise_path

        if self.start == self.end:
            self.angle = volmdlr.TWO_PI

        if self.is_trigo:
            self.offset_angle = angle1
        else:
            self.offset_angle = angle2

        Primitive3D.__init__(self, basis_primitives=self.tessellation_points(),
                             name=name)

    def _get_points(self):
        return self.tessellation_points()

    points = property(_get_points)

    def tessellation_points(self, resolution_for_ellipse=40):
        number_points_tesselation = math.ceil(
            resolution_for_ellipse * abs(0.5 * self.angle / math.pi))

        plane3d = volmdlr.faces.Plane3D(self.center, self.major_dir, self.minor_dir,
                          self.normal)
        frame3d = Frame3D(self.center, plane3d.vectors[0], plane3d.vectors[1],
                          plane3d.normal)

        tessellation_points_3D = [Point3D((self.Gradius * math.cos(
            self.offset_angle + self.angle * i / (number_points_tesselation)),
                                           self.Sradius * math.sin(
                                               self.offset_angle + self.angle * i / (
                                                   number_points_tesselation)),
                                           0)) for i in
                                  range(number_points_tesselation + 1)]

        global_points = []
        for pt in tessellation_points_3D:
            global_points.append(frame3d.OldCoordinates(pt))

        return global_points

    def to_2d(self, plane_origin, x, y):
        ps = self.start.to_2d(plane_origin, x, y)
        pi = self.interior.to_2d(plane_origin, x, y)
        pe = self.end.to_2d(plane_origin, x, y)
        center = self.center.to_2d(plane_origin, x, y)

        if self.extra is None:
            pextra = None
        else:
            pextra = self.extra.to_2d(plane_origin, x, y)

        maj_dir2d = self.major_dir.to_2d(plane_origin, x, y)
        maj_dir2d.normalize()
        return ArcEllipse2D(ps, pi, pe, center, maj_dir2d, name=self.name,
                            extra=pextra)

    def length(self):
        return self.angle * math.sqrt(
            (self.Gradius ** 2 + self.Sradius ** 2) / 2)

    def plot(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = Axes3D(fig)
        else:
            fig = None

        ax.plot([self.interior[0]], [self.interior[1]], [self.interior[2]],
                color='b')
        ax.plot([self.start[0]], [self.start[1]], [self.start[2]], c='r')
        ax.plot([self.end[0]], [self.end[1]], [self.end[2]], c='r')
        ax.plot([self.interior[0]], [self.interior[1]], [self.interior[2]],
                c='g')
        x = []
        y = []
        z = []
        for px, py, pz in self.tessellation_points():
            x.append(px)
            y.append(py)
            z.append(pz)

        ax.plot(x, y, z, 'k')
        return ax

    def plot2D(self, x3d, y3D, ax, color='k'):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure

        # TODO: Enhance this plot
        l = self.length()
        x = []
        y = []
        for i in range(30):
            p = self.point_at_abscissa(i / (29.) * l)
            xi, yi = p.PlaneProjection2D(X3D, Y3D)
            x.append(xi)
            y.append(yi)
        ax.plot(x, y, color=color)

        return ax

    def FreeCADExport(self, name, ndigits=6):
        xs, ys, zs = round(1000 * self.start, ndigits).vector
        xi, yi, zi = round(1000 * self.interior, ndigits).vector
        xe, ye, ze = round(1000 * self.end, ndigits).vector
        return '{} = Part.Arc(fc.Vector({},{},{}),fc.Vector({},{},{}),fc.Vector({},{},{}))\n'.format(
            name, xs, ys, zs, xi, yi, zi, xe, ye, ze)
