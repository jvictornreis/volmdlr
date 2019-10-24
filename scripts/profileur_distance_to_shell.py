#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:57:51 2019

@author: ringhausen
"""

import volmdlr as  vm
import volmdlr.primitives3D as p3d

moteur = vm.Step('/home/ringhausen/Documents/git/ClientsProjects/Renault/CMO/data/step/MOTEUR HRevoUNIFY v2.stp')
#boite = vm.Step('/home/ringhausen/Documents/git/ClientsProjects/Renault/CMO/data/step/BOITE E-TECHg2.stp')
#cmo = vm.Step('/home/ringhausen/Documents/git/ClientsProjects/Renault/CMO/data/step/CMO2.stp')


volumemodel = vm.VolumeModel([],[])

moteur.to_volume_model('moteur', volumemodel)
#boite.to_volume_model('boite', volumemodel)
#cmo.to_volume_model('cmo', volumemodel)

origin = vm.Point3D((0,0,0))
u = vm.Vector3D((0.2,0,0))
v = vm.Vector3D((0,0.2,0))
w = vm.Vector3D((0,0,0.2))
frame = vm.Frame3D(origin, u, v, w)
primitive3d = p3d.Block(frame, 'test', (1,0,1))
#volumemodel.shells.append(primitive3d)

#origin = vm.Point3D((0,0,0))
#u = vm.Vector3D((0.5,0,0))
#v = vm.Vector3D((0,0.5,0))
#w = vm.Vector3D((0,0,0.5))
#frame = vm.Frame3D(origin, u, v, w)
#primitive3d = p3d.Block(frame, 'test', (1,0,1))
#volumemodel.shells.append(primitive3d)

#origin = vm.Point3D((0,0,0))
#u = vm.Vector3D((2,1,0))
#v = vm.Vector3D((0,1,0))
#w = vm.Vector3D((0.5,0,1))
#frame = vm.Frame3D(origin, u, v, w)
#primitive3d = p3d.Block(frame, 'test') #, 'test', (1,0,1))
#volumemodel.shells.append(primitive3d)

#volumemodel = vm.VolumeModel([],[])
#position = vm.Point3D((1,2,3))
#axis = vm.Vector3D((1,0.5,0.2))
#radius = 0.5
#length = 2
#primitive3d = p3d.Cone(position, axis, radius, length)
#primitive3d = p3d.Cylinder(position, axis, radius, length)
#volumemodel.shells.append(primitive3d)
#volumemodel.shells.append(primitive3d.bounding_box)
#volumemodel.BabylonShow()

#union = volumemodel.shells[0].union(primitive3d)
#volumemodel.shells[0] = union
#volumemodel.shells.append(union.bounding_box)
#union.Translation((1,2,-0.5), False)
#volumemodel.BabylonShow()

shell0 = volumemodel.shells[0] # LE MOTEUR
#shell0.color = (1,0,0)
#shell1 = volumemodel.shells[1] # LA BOITE
#shell2 = volumemodel.shells[2] # LE BLOCK
#shell2 = volumemodel.shells[2] # LE CMO
#shell3 = volumemodel.shells[3] # UN PETIT BOUT SE TROUVANT A L'INTERIEUR DU CMO
#shell4 = volumemodel.shells[4] # UN PETIT BOUT SE TROUVANT A L'INTERIEUR DU CMO

#del volumemodel.shells[4]
#del volumemodel.shells[3]

volumemodel.BabylonShow()


#%%

origin = vm.Point3D((1,0.1,0.6))
u = vm.y3D
v = vm.z3D
w = vm.x3D
frame = vm.Frame3D(origin, u, v, w)


#print('fram_mapping copy=False')
#shell0.frame_mapping(frame, 'new', False)
#print('Success')
#print('Transaliton copy=False')
#shell1.Translation(vm.Vector3D((1,0.5,0)), False)
#print('Success')
#
#print('frame_mapping copy=True')
#volumemodel.shells[1] = shell1.frame_mapping(frame, 'new', True)
#print('Success')
#print('Translation copy=True')
#volumemodel.shells[0] = shell0.Translation(vm.Vector3D((-1,0.5,0)), True)
#print('Success')

#volumemodel.shells.append(volumemodel.shells[0].bounding_box)
#volumemodel.shells.append(volumemodel.shells[1].bounding_box)

#for face in volumemodel.shells[0].faces+volumemodel.shells[1].faces:
#    volumemodel.shells.append(face.bounding_box)

shell1.Translation(vm.Vector3D((0,0,1)), False)


boo10 = volumemodel.shells[1].is_inside_shell(volumemodel.shells[0])
boo01 = volumemodel.shells[0].is_inside_shell(volumemodel.shells[1])
print('shell1 is inside shell0', boo10)
print('shell0 is inside shell1', boo01)

mesure = volumemodel.shells[0].distance_to_shell(volumemodel.shells[1], volumemodel)
if mesure is not None:
    volumemodel.shells.append(mesure)
    print('distance =', mesure.distance)
    
internal = shell1.intersection_internal_aabb_volume(shell0)
#print('internal volume', internal)
external = shell1.intersection_external_aabb_volume(shell0)
#print('external volume', external)
volumemodel.shells.append(internal)
volumemodel.shells.append(external)


#pc = shell0.shell_intersection(shell1)
#print('intersection à', pc*100, '%')

volumemodel.BabylonShow()

#%%