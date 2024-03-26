#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:25:22 2019

@author: ringhausen
"""

import math

import volmdlr as vm
import volmdlr.edges as vme
import volmdlr.primitives2d as p2d
import volmdlr.primitives3d as p3d
import volmdlr.wires as vmw

# %%

p1 = vm.Point2D(0, 0)
p2 = vm.Point2D(0, 2)
p3 = vm.Point2D(2, 4)
p4 = vm.Point2D(4, 4)
p5 = vm.Point2D(4, 3)
p6 = vm.Point2D(3, 2)
p7 = vm.Point2D(3, 0)

l1 = p2d.OpenedRoundedLineSegments2D([p7, p1, p2], {})
l2 = vme.Arc2D.from_3_points(p2, vm.Point2D(math.sqrt(2) / 2, 3 + math.sqrt(2) / 2), p3)
l3 = p2d.OpenedRoundedLineSegments2D([p3, p4, p5, p6], {}, adapt_radius=True)
l4 = vme.Arc2D.from_3_points(p6, vm.Point2D(4, 1), p7)
c1 = vmw.Contour2D([l1, l2, l3, l4])

p8 = vm.Point2D(1, 1)
p9 = vm.Point2D(2, 1)
p10 = vm.Point2D(2, 2)
p11 = vm.Point2D(1, 2)
c2 = p2d.ClosedRoundedLineSegments2D([p8, p9, p10, p11], {})
# c2 = vmw.Contour2D([inner])

profile = p3d.ExtrudedProfile(vm.OXYZ, c1, [], 1)
# profile.plot()

model = vm.model.VolumeModel([profile])
model.babylonjs()

# %%

p1 = vm.Point2D(0, 0)
p2 = vm.Point2D(2, 0)
p3 = vm.Point2D(2, 2)
p4 = vm.Point2D(0, 2)

p5 = vm.Point2D(0.5, 0.5)
p6 = vm.Point2D(1.5, 0.5)
p7 = vm.Point2D(1.5, 1.5)
p8 = vm.Point2D(0.5, 1.5)

l1 = p2d.ClosedRoundedLineSegments2D([p1, p2, p3, p4], {})
c1 = vm.wires.Contour2D(l1.primitives)

l2 = p2d.ClosedRoundedLineSegments2D([p5, p6, p7, p8], {})
c2 = vm.wires.Contour2D(l2.primitives)

profile = p3d.ExtrudedProfile(vm.OXYZ, c1, [c2], 1)

model = vm.model.VolumeModel([profile])
model.babylonjs()
