#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import volmdlr as vm
import volmdlr.step
import volmdlr.cloud as vmcd

for step_file in [
                    'tore1.step',
                    'cone1.step',
                    'cone2.step',
                    'cylinder.step',
                     'STEP_test1.stp',
                    'block.step',
                     'iso4162M16x55.step',
                    'aircraft_engine.step'
                  ]:
    print('filename: ', step_file)
    step = volmdlr.step.Step(step_file)
    model = step.to_volume_model()
    model.to_step(step_file+'_reexport')
    print(model.primitives)

    model.babylonjs()
    
model2 = model.copy()

assert model == model2