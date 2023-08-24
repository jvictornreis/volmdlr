"""
Unit tests for volmdlr.faces.BSplineSurface3D
"""
import unittest

import volmdlr.edges as vme
import volmdlr.wires as vmw
import volmdlr.faces as vmf
import volmdlr.grid
from volmdlr.models import bspline_surfaces
from volmdlr import surfaces


GEOMDL_DELTA = 0.001


class TestBSplineSurface3D(unittest.TestCase):

    def setUp(self):
        """Create a B-spline surface instance as a fixture"""
        # Set degrees
        degree_u = 3
        degree_v = 3

        ctrlpts = [
            [-25.0, -25.0, -10.0], [-25.0, -15.0, -5.0], [-25.0, -5.0, 0.0], [-25.0, 5.0, 0.0],
            [-25.0, 15.0, -5.0], [-25.0, 25.0, -10.0], [-15.0, -25.0, -8.0], [-15.0, -15.0, -4.0],
            [-15.0, -5.0, -4.0], [-15.0, 5.0, -4.0], [-15.0, 15.0, -4.0], [-15.0, 25.0, -8.0],
            [-5.0, -25.0, -5.0], [-5.0, -15.0, -3.0], [-5.0, -5.0, -8.0], [-5.0, 5.0, -8.0],
            [-5.0, 15.0, -3.0], [-5.0, 25.0, -5.0], [5.0, -25.0, -3.0], [5.0, -15.0, -2.0],
            [5.0, -5.0, -8.0], [5.0, 5.0, -8.0], [5.0, 15.0, -2.0], [5.0, 25.0, -3.0],
            [15.0, -25.0, -8.0], [15.0, -15.0, -4.0], [15.0, -5.0, -4.0], [15.0, 5.0, -4.0],
            [15.0, 15.0, -4.0], [15.0, 25.0, -8.0], [25.0, -25.0, -10.0], [25.0, -15.0, -5.0],
            [25.0, -5.0, 2.0], [25.0, 5.0, 2.0], [25.0, 15.0, -5.0], [25.0, 25.0, -10.0],
        ]

        nb_u, nb_v = 6, 6

        # Set knot vectors
        knots_u = [0.0, 0.33, 0.66, 1.0]
        u_multiplicities = [4, 1, 1, 4]
        knots_v = [0.0, 0.33, 0.66, 1.0]
        v_multiplicities = [4, 1, 1, 4]
        # knotvector_u = [0.0, 0.0, 0.0, 0.0, 0.33, 0.66, 1.0, 1.0, 1.0, 1.0]
        # knotvector_v = [0.0, 0.0, 0.0, 0.0, 0.33, 0.66, 1.0, 1.0, 1.0, 1.0]

        self.spline_surf = surfaces.BSplineSurface3D(degree_u, degree_v, ctrlpts, nb_u, nb_v,
                                                     u_multiplicities, v_multiplicities, knots_u, knots_v)
        weights = [0.5, 1.0, 0.75, 1.0, 0.25, 1.0, 0.5, 1.0, 0.75, 1.0, 0.25, 1.0,
                   0.5, 1.0, 0.75, 1.0, 0.25, 1.0, 0.5, 1.0, 0.75, 1.0, 0.25, 1.0,
                   0.5, 1.0, 0.75, 1.0, 0.25, 1.0, 0.5, 1.0, 0.75, 1.0, 0.25, 1.0]
        self.nurbs_surf = surfaces.BSplineSurface3D(degree_u, degree_v, ctrlpts, nb_u, nb_v,
                                                     u_multiplicities, v_multiplicities, knots_u, knots_v, weights)

    def test_point2d_to_3d(self):
        test_cases = [
            (volmdlr.Point2D(0.0, 0.0), (-25.0, -25.0, -10.0)),
            (volmdlr.Point2D(0.0, 0.2), (-25.0, -11.403, -3.385)),
            (volmdlr.Point2D(0.0, 1.0), (-25.0, 25.0, -10.0)),
            (volmdlr.Point2D(0.3, 0.0), (-7.006, -25.0, -5.725)),
            (volmdlr.Point2D(0.3, 0.4), (-7.006, -3.308, -6.265)),
            (volmdlr.Point2D(0.3, 1.0), (-7.006, 25.0, -5.725)),
            (volmdlr.Point2D(0.6, 0.0), (3.533, -25.0, -4.224)),
            (volmdlr.Point2D(0.6, 0.6), (3.533, 3.533, -6.801)),
            (volmdlr.Point2D(0.6, 1.0), (3.533, 25.0, -4.224)),
            (volmdlr.Point2D(1.0, 0.0), (25.0, -25.0, -10.0)),
            (volmdlr.Point2D(1.0, 0.8), (25.0, 11.636, -2.751)),
            (volmdlr.Point2D(1.0, 1.0), (25.0, 25.0, -10.0)),
        ]

        for param, res in test_cases:
            evalpt = self.spline_surf.point2d_to_3d(param)
            self.assertAlmostEqual(evalpt[0], res[0], delta=GEOMDL_DELTA)
            self.assertAlmostEqual(evalpt[1], res[1], delta=GEOMDL_DELTA)
            self.assertAlmostEqual(evalpt[2], res[2], delta=GEOMDL_DELTA)

        test_data = [
            (volmdlr.Point2D(0.0, 0.0), (-25.0, -25.0, -10.0)),
            (volmdlr.Point2D(0.0, 0.2), (-25.0, -11.563, -3.489)),
            (volmdlr.Point2D(0.0, 1.0), (-25.0, 25.0, -10.0)),
            (volmdlr.Point2D(0.3, 0.0), (-7.006, -25.0, -5.725)),
            (volmdlr.Point2D(0.3, 0.4), (-7.006, -3.052, -6.196)),
            (volmdlr.Point2D(0.3, 1.0), (-7.006, 25.0, -5.725)),
            (volmdlr.Point2D(0.6, 0.0), (3.533, -25.0, -4.224)),
            (volmdlr.Point2D(0.6, 0.6), (3.533, 2.868, -7.257)),
            (volmdlr.Point2D(0.6, 1.0), (3.533, 25.0, -4.224)),
            (volmdlr.Point2D(1.0, 0.0), (25.0, -25.0, -10.0)),
            (volmdlr.Point2D(1.0, 0.8), (25.0, 9.425, -1.175)),
            (volmdlr.Point2D(1.0, 1.0), (25.0, 25.0, -10.0)),
        ]
        for param, res in test_data:
            evalpt = self.nurbs_surf.point2d_to_3d(param)
            self.assertAlmostEqual(evalpt[0], res[0], delta=GEOMDL_DELTA)
            self.assertAlmostEqual(evalpt[1], res[1], delta=GEOMDL_DELTA)
            self.assertAlmostEqual(evalpt[2], res[2], delta=GEOMDL_DELTA)

    def test_derivatives(self):
        test_data = [
            (
                (0.0, 0.25),
                1,
                [
                    [[-25.0, -9.0771, -2.3972], [5.5511e-15, 43.6910, 17.5411]],
                    [[90.9090, 0.0, -15.0882], [-5.9750e-15, 0.0, -140.0367]],
                ],
            ),
            (
                (0.95, 0.75),
                2,
                [
                    [
                        [20.8948, 9.3097, -2.4845],
                        [-1.1347e-14, 43.7672, -15.0153],
                        [-5.0393e-30, 100.1022, -74.1165],
                    ],
                    [
                        [76.2308, -1.6965e-15, 18.0372],
                        [9.8212e-15, -5.9448e-15, -158.5462],
                        [4.3615e-30, -2.4356e-13, -284.3037],
                    ],
                    [
                        [224.5342, -5.6794e-14, 93.3843],
                        [4.9856e-14, -4.0400e-13, -542.6274],
                        [2.2140e-29, -1.88662e-12, -318.8808],
                    ],
                ],
            ),
        ]
        for param, order, res in test_data:
            deriv = self.spline_surf.derivatives(*param, order=order)
            for computed, expected in zip(deriv, res):
                for idx in range(order + 1):
                    for c, e in zip(computed[idx], expected[idx]):
                        self.assertAlmostEqual(c, e, delta=GEOMDL_DELTA)

        test_data = [
            (
                (0.0, 0.25),
                1,
                [
                    [[-25.0, -9.4859, -2.6519], [9.1135e-15, 42.3855, 16.1635]],
                    [[90.9090, 0.0, -12.5504], [-3.6454e-14, 0.0, -135.9542]]
                ],
            ),
            (
                (0.95, 0.75),
                2,
                [
                    [
                        [20.8948, 6.5923, -1.4087],
                        [0.0, 42.0924, -14.6714],
                        [-4.4688e-14, 498.0982, -230.2790]
                    ],
                    [
                        [76.2308, -3.1813e-15, 31.9560],
                        [2.6692e-14, -1.6062e-14, -134.4819],
                        [-4.6035e-14, -4.5596e-13, -1646.1763]
                    ],
                    [
                        [224.5342, -1.9181e-14, 144.3825],
                        [-8.6754e-14, -4.3593e-13, -433.4414],
                        [1.2012e-12, -5.9424e-12, -4603.0856]
                    ]
                ],
            ),
        ]
        for param, order, res in test_data:
            deriv = self.nurbs_surf.derivatives(*param, order=order)
            for computed, expected in zip(deriv, res):
                for idx in range(order + 1):
                    for c, e in zip(computed[idx], expected[idx]):
                        self.assertAlmostEqual(c, e, delta=GEOMDL_DELTA)

    def test_interpolate_surface(self):
        points = [volmdlr.Point3D(1.0, 0.0, 0.0), volmdlr.Point3D(0.70710678, 0.70710678, 0.0),
                  volmdlr.Point3D(0.0, 1.0, 0.0), volmdlr.Point3D(-0.70710678, 0.70710678, 0.0),
                  volmdlr.Point3D(-1.0, 0.0, 0.0), volmdlr.Point3D(-0.70710678, -0.70710678, 0.0),
                  volmdlr.Point3D(0.0, -1.0, 0.0), volmdlr.Point3D(0.70710678, -0.70710678, 0.0),
                  volmdlr.Point3D(1.0, 0.0, 0.0), volmdlr.Point3D(1.0, 0.0, 1.0),
                  volmdlr.Point3D(0.70710678, 0.70710678, 1.0), volmdlr.Point3D(0.0, 1.0, 1.0),
                  volmdlr.Point3D(-0.70710678, 0.70710678, 1.0), volmdlr.Point3D(-1.0, 0.0, 1.0),
                  volmdlr.Point3D(-0.70710678, -0.70710678, 1.0), volmdlr.Point3D(0.0, -1.0, 1.0),
                  volmdlr.Point3D(0.70710678, -0.70710678, 1.0), volmdlr.Point3D(1.0, 0.0, 1.0)]

        degree_u = 1
        degree_v = 2
        size_u = 2
        size_v = 9
        surface = surfaces.BSplineSurface3D.points_fitting_into_bspline_surface(points, size_u, size_v,
                                                                                degree_u, degree_v)

        expected_points = [volmdlr.Point3D(1.0, 0.0, 0.0),
                           volmdlr.Point3D(0.9580995893491125, 0.6733882798117155, 0.0),
                           volmdlr.Point3D(-0.0005819501479292128, 1.0804111054393308, 0.0),
                           volmdlr.Point3D(-0.7628715805621301, 0.7627405224267781, 0.0),
                           volmdlr.Point3D(-1.0790428064792899, 0.0, 0.0),
                           volmdlr.Point3D(-0.7628715805621301, -0.7627405224267783, 0.0),
                           volmdlr.Point3D(-0.0005819501479290552, -1.0804111054393304, 0.0),
                           volmdlr.Point3D(0.9580995893491127, -0.6733882798117156, 0.0),
                           volmdlr.Point3D(1.0, 0.0, 0.0),
                           volmdlr.Point3D(1.0, 0.0, 1.0),
                           volmdlr.Point3D(0.9580995893491125, 0.6733882798117155, 1.0),
                           volmdlr.Point3D(-0.0005819501479292128, 1.0804111054393308, 1.0),
                           volmdlr.Point3D(-0.7628715805621301, 0.7627405224267781, 1.0),
                           volmdlr.Point3D(-1.0790428064792899, 0.0, 1.0),
                           volmdlr.Point3D(-0.7628715805621301, -0.7627405224267783, 1.0),
                           volmdlr.Point3D(-0.0005819501479290552, -1.0804111054393304, 1.0),
                           volmdlr.Point3D(0.9580995893491127, -0.6733882798117156, 1.0),
                           volmdlr.Point3D(1.0, 0.0, 1.0)]
        for point, expected_point in zip(surface.control_points, expected_points):
            self.assertTrue(point.is_close(expected_point))
        point1 = surface.point2d_to_3d(volmdlr.Point2D(0.0, 0.0))
        point2 = surface.point2d_to_3d(volmdlr.Point2D(0.25, 0.25))
        point3 = surface.point2d_to_3d(volmdlr.Point2D(0.5, 0.5))
        point4 = surface.point2d_to_3d(volmdlr.Point2D(0.75, 0.75))
        point5 = surface.point2d_to_3d(volmdlr.Point2D(1.0, 1.0))

        for point in [point1, point2, point3, point4, point5]:
            self.assertAlmostEqual(point.point_distance(volmdlr.Point3D(0.0, 0.0, point.z)), 1.0)


    def test_contour2d_parametric_to_dimension(self):
        bspline_face = vmf.BSplineFace3D.from_surface_rectangular_cut(bspline_surfaces.bspline_surface_2, 0, 1, 0, 1)
        contour2d = bspline_surfaces.bspline_surface_2.contour3d_to_2d(bspline_face.outer_contour3d)
        grid2d = volmdlr.grid.Grid2D.from_properties((0, 1), (0, 1), (10, 10))
        contour2d_dim = bspline_surfaces.bspline_surface_2.contour2d_parametric_to_dimension(contour2d, grid2d)
        self.assertEqual(len(contour2d_dim.primitives), 4)
        self.assertAlmostEqual(contour2d_dim.area(), 16.657085821451233, places=2)
        self.assertAlmostEqual(contour2d_dim.length(), 16.823814079415172, places=2)

    def test_periodicity(self):
        bspline_suface = surfaces.BSplineSurface3D.load_from_file('surfaces/surface3d_8.json')
        self.assertAlmostEqual(bspline_suface.x_periodicity,  0.8888888888888888)
        self.assertFalse(bspline_suface.y_periodicity)

    def test_bbox(self):
        surface = bspline_surfaces.bspline_surface_3
        bbox = surface.bounding_box
        volume = bbox.volume()

        # Check if the bounding box volume is correct
        self.assertEqual(volume, 4.0)

    def test_arc3d_to_2d(self):
        bspline_surface = surfaces.BSplineSurface3D.load_from_file('surfaces/BSplineSurface3D_with_Arc3D.json')
        arc = vme.Arc3D.from_3_points(volmdlr.Point3D(-0.01, -0.013722146986970815, 0.026677756316261864),
                        volmdlr.Point3D(-0.01, 0.013517082603, 0.026782241839),
                        volmdlr.Point3D(-0.01, 0.029612430603, 0.004806657236))

        test = bspline_surface.arc3d_to_2d(arc3d=arc)[0]

        inv_prof = bspline_surface.linesegment2d_to_3d(test)[0]

        # Verifies the inversion operation
        self.assertIsInstance(inv_prof, vme.Arc3D)
        self.assertTrue(inv_prof.start.is_close(arc.start))
        # self.assertTrue(inv_prof.interior.is_close(arc.interior))
        self.assertTrue(inv_prof.end.is_close(arc.end))

        # Strange case from step file
        bspline_surface = surfaces.BSplineSurface3D.load_from_file(
            'surfaces/objects_bspline_test/bsplinesurface_arc3d_to_2d_surface.json')
        arc = vme.Arc3D.load_from_file("surfaces/objects_bspline_test/bsplinesurface_arc3d_to_2d_arc3d.json")
        brep = bspline_surface.arc3d_to_2d(arc)[0]
        self.assertTrue(brep.start.is_close(volmdlr.Point2D(1, 0)))

    def test_bsplinecurve3d_to_2d(self):
        bspline_surface = bspline_surfaces.bspline_surface_4
        control_points = [volmdlr.Point3D(-0.012138106431296442, 0.11769707710908962, -0.10360094389690414),
         volmdlr.Point3D(-0.012153195391844274, 0.1177764571887428, -0.10360691055433219),
         volmdlr.Point3D(-0.01216612946601426, 0.11785649353385147, -0.10361063821784446),
         volmdlr.Point3D(-0.012176888504086755, 0.11793706145749239, -0.10361212108019317)]
        weights = [1.0, 0.9994807070752826, 0.9994807070752826, 1.0]
        original_bspline = vme.BSplineCurve3D(3, control_points, [4, 4], [0, 1], weights, False)
        bspline_on_parametric_domain = bspline_surface.bsplinecurve3d_to_2d(original_bspline)[0]
        bspline_after_transfomation = bspline_surface.linesegment2d_to_3d(bspline_on_parametric_domain)[0]
        original_length = original_bspline.length()
        length_after_transformation = bspline_after_transfomation.length()
        point = original_bspline.point_at_abscissa(0.5 * original_length)
        point_test = bspline_after_transfomation.point_at_abscissa(0.5 * length_after_transformation)
        self.assertAlmostEqual(original_length, length_after_transformation, places=6)
        # self.assertTrue(point.is_close(point_test, 1e-6))

    def test_bsplinecurve2d_to_3d(self):
        surface = surfaces.BSplineSurface3D.load_from_file("surfaces/objects_bspline_test/bspline_surface_with_arcs.json")
        contour3d = vmw.Contour3D.load_from_file("surfaces/objects_bspline_test/bspline_contour_with_arcs.json")

        contour2d = surface.contour3d_to_2d(contour3d)
        bspline_1 = contour2d.primitives[0]
        arc3d = surface.bsplinecurve2d_to_3d(bspline_1)[0]
        self.assertTrue(isinstance(bspline_1, vme.BSplineCurve2D))
        self.assertTrue(isinstance(arc3d, vme.Arc3D))

    def test_arcellipse3d_to_2d(self):
        arcellipse = vme.ArcEllipse3D.load_from_file("surfaces/objects_bspline_test/arcellipse_on_bsplinesurface.json")
        bsplinesurface = surfaces.BSplineSurface3D.load_from_file(
            "surfaces/objects_bspline_test/bsplinesurface_with_arcellipse.json")
        test = bsplinesurface.arcellipse3d_to_2d(arcellipse)[0]
        self.assertTrue(isinstance(test, vme.LineSegment2D))
        self.assertTrue(test.start.is_close(volmdlr.Point2D(0.5, 0.0), 1e-4))
        self.assertTrue(test.end.is_close(volmdlr.Point2D(0.5, 1), 1e-4))

        # todo: Uncomment this block when finish debugging contour2d healing
        # surface = surfaces.BSplineSurface3D.load_from_file(
        #     "surfaces/objects_bspline_test/bspline_surface_self_intersecting_contour.json")
        # contour3d = vmw.Contour3D.load_from_file(
        #     "surfaces/objects_bspline_test/bspline_contour_self_intersecting_contour.json")
        # face = surface.face_from_contours3d([contour3d])
        # self.assertTrue(face.surface2d.outer_contour.is_ordered())

    def test_contour3d_to_2d(self):
        surface = surfaces.BSplineSurface3D.load_from_file("surfaces/objects_bspline_test/periodicalsurface.json")
        contour3d = vmw.Contour3D.load_from_file("surfaces/objects_bspline_test/periodicalsurface_contour.json")
        contour2d = surface.contour3d_to_2d(contour3d)
        self.assertTrue(contour2d.is_ordered())
        self.assertAlmostEqual(contour2d.area(), 1/6, 5)

        surface = surfaces.BSplineSurface3D.load_from_file(
            "surfaces/objects_bspline_test/contour3d_to_2d_small_primitives_surface.json")
        contour3d = vmw.Contour3D.load_from_file(
            "surfaces/objects_bspline_test/contour3d_to_2d_small_primitives_contour.json")
        contour2d = surface.contour3d_to_2d(contour3d)
        self.assertTrue(contour2d.is_ordered(1e-2)) # 1e-2 is an acceptable value, because this is parametric dimension

        surface = surfaces.BSplineSurface3D.load_from_file(
            "surfaces/objects_bspline_test/surface_with_singularity.json")
        contour3d = vmw.Contour3D.load_from_file(
            "surfaces/objects_bspline_test/surface_with_singularity_contour.json")
        contour2d = surface.contour3d_to_2d(contour3d)
        self.assertTrue(contour2d.is_ordered())

if __name__ == '__main__':
    unittest.main(verbosity=0)
