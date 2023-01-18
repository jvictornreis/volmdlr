#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 3 2022

@author: s.bendjebla
"""

# %% Librairies

import volmdlr.edges

# %% BsplineCurve2D (1)

dic_bspline = {
    'name': '',
    'object_class': 'volmdlr.edges.BSplineCurve2D',
    'package_version': '0.5.1.dev193+gc6bdd70e',
    'degree': 3,
    'weights': None,
    'periodic': False,
    'start': {'object_class': 'volmdlr.Point2D',
     'x': -0.4544751134860842,
     'y': 1.5707977686871983,
     'name': ''},
    'end': {'object_class': 'volmdlr.Point2D',
     'x': -0.3124009292097013,
     'y': -3.141592653588988,
     'name': ''},
    'control_points': [{'$ref': '#/start'},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.4553429239113792,
      'y': 1.620595708870625,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.45583687948274487,
      'y': 1.7225897961445817,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.4520590939005643,
      'y': 1.8859455670744714,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.4432673038692045,
      'y': 2.0505943691631203,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.43149023009611276,
      'y': 2.196369778630224,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.41807624324211334,
      'y': 2.3265228465729813,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.40398926099996524,
      'y': 2.442576243391531,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.390014646035353,
      'y': 2.5455258258288014,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.3767380630341135,
      'y': 2.636436435451037,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.364420681738801,
      'y': 2.717006060784525,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.352993811394062,
      'y': 2.790125033607336,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.34264250918889616,
      'y': 2.856354311373755,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.3334693750607423,
      'y': 2.9167961005375314,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.3251167944404208,
      'y': 2.9758770725556776,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.31834827936261717,
      'y': 3.031890968787418,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.31362756905698363,
      'y': 3.086173401011527,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.31239657524755504,
      'y': 3.1230758830026706,
      'name': ''},
     {'$ref': '#/end'}],
    'knots': [0.0,
     0.0625,
     0.125,
     0.1875,
     0.25,
     0.3125,
     0.375,
     0.4375,
     0.5,
     0.5625,
     0.625,
     0.6875,
     0.75,
     0.8125,
     0.875,
     0.9375,
     1.0],
    'knot_multiplicities': [4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
    'points': [{'$ref': '#/start'},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.4548466635787877,
      'y': 1.5950384158111264,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.45511756697976646,
      'y': 1.6194803667255397,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.45528447308284226,
      'y': 1.6441370032839118,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.455344031281837,
      'y': 1.6690217073397149,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.4552928909705727,
      'y': 1.694147860746423,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.4551277015428712,
      'y': 1.7195288453575088,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.4548452250154656,
      'y': 1.7451734233409908,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.45444302314566176,
      'y': 1.7710575522618923,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.45391900376144056,
      'y': 1.7971429941929713,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.4532710760748605,
      'y': 1.8233914544334284,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.45249714929798024,
      'y': 1.8497646382824626,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.45159513264285833,
      'y': 1.8762242510392753,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.4505633688659587,
      'y': 1.9027300922649175,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.4494060865225913,
      'y': 1.929216089219333,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.4481317559665275,
      'y': 1.955597523420418,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.44674894119712977,
      'y': 1.9817892647466278,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.44526620621376106,
      'y': 2.0077061830764182,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.4436921150157843,
      'y': 2.0332631482882464,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.4420351915186864,
      'y': 2.058376409937326,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.44030269810607103,
      'y': 2.083005639184192,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.4385004119546198,
      'y': 2.107161627632355,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.4366340250481718,
      'y': 2.130858099201241,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.4347092293705654,
      'y': 2.1541087778102743,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.4327317169056396,
      'y': 2.17692738737888,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.43070717541087433,
      'y': 2.199327632826092,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.42864078125434185,
      'y': 2.221320920023585,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.4265367176098109,
      'y': 2.242914189751056,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.4243990535393636,
      'y': 2.264113869777636,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.42223185810508257,
      'y': 2.2849263878724546,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.4200392003690504,
      'y': 2.305358171804641,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.41782514936267906,
      'y': 2.3254156492528355,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.4155936235578411,
      'y': 2.345104803680675,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.41334804177974516,
      'y': 2.3644301443805538,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.4110917193419994,
      'y': 2.3833958752413955,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.4088279715582119,
      'y': 2.4020062001521234,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.40656011374199097,
      'y': 2.4202653230016637,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.40429146120694454,
      'y': 2.4381774476789384,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.4020252351425574,
      'y': 2.4557468105876525,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.3997641070863636,
      'y': 2.472977838006455,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.3975105496254905,
      'y': 2.4898750249405714,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.3952670350726511,
      'y': 2.506442866490023,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.3930360357405587,
      'y': 2.5226858577548303,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.3908200239419262,
      'y': 2.538608493835015,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.388621397802237,
      'y': 2.5542154145829894,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.38644175510555184,
      'y': 2.5695128214586704,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.3842822053156306,
      'y': 2.5845078687211487,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.38214385092898795,
      'y': 2.599207724223843,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.3800277944421384,
      'y': 2.613619555820171,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.3779351383515963,
      'y': 2.6277505313635516,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.3758669561441629,
      'y': 2.641607923568505,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.3738236540832329,
      'y': 2.6552014169549416,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.37180497120879513,
      'y': 2.668543107848158,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.3698106175511247,
      'y': 2.6816451974345528,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.36784030314049704,
      'y': 2.694519886900528,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.3658937380071872,
      'y': 2.7071793774324835,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.36397063762970416,
      'y': 2.719635820202718,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.36207109934234555,
      'y': 2.731897860995254,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.36019584632888635,
      'y': 2.743968400374352,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.35834565978589195,
      'y': 2.755849806354134,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.3565213209099278,
      'y': 2.767544446948719,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.354723610897559,
      'y': 2.7790546901722277,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.3529533108320427,
      'y': 2.7903829047825366,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.3512111196480645,
      'y': 2.8015319987603426,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.3494975093236369,
      'y': 2.8125063698288493,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.347812912972,
      'y': 2.8233106708194367,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.34615776370639384,
      'y': 2.8339495545634827,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.34453249464005853,
      'y': 2.8444276738923673,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.3429375388862342,
      'y': 2.8547496816374696,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.34137306276343055,
      'y': 2.864922157883073,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.33983794478187596,
      'y': 2.87496098549184,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.3383306753939227,
      'y': 2.884884850551473,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.3368497449728728,
      'y': 2.894712439720711,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.33539364389202847,
      'y': 2.9044624396582925,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.3339608625246917,
      'y': 2.9141535370229574,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.332550224012169,
      'y': 2.923802598123358,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.3311634478098794,
      'y': 2.933410645480362,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.3298037446668912,
      'y': 2.9429705437496354,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.3284743376570138,
      'y': 2.9524750901664705,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.32717844985405625,
      'y': 2.961917081966161,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.32591930433182814,
      'y': 2.9712893163839995,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.3247000866268476,
      'y': 2.9805849715556563,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.3235233278703795,
      'y': 2.989803866032311,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.322391003343133,
      'y': 2.998951458721221,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.3213050706642383,
      'y': 3.0080333877461474,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.32026748745282596,
      'y': 3.017055291230848,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.3192802113280263,
      'y': 3.0260228072990825,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.3183452133649473,
      'y': 3.034941628508501,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.3174650741446269,
      'y': 3.0438199130702976,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.3166432199811892,
      'y': 3.052669240466372,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.3158831394849492,
      'y': 3.0615014421873665,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.31518832126622176,
      'y': 3.0703283497239235,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.3145622539353218,
      'y': 3.0791617945666845,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.314008426188227,
      'y': 3.08783875316965,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.31353034813979486,
      'y': 3.052475966713578,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.3131315794021342,
      'y': 2.8281563457618653,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.3128156865577697,
      'y': 2.255734781785127,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.3125862361892262,
      'y': 1.1760661662539973,
      'name': ''},
     {'object_class': 'volmdlr.Point2D',
      'x': -0.3124467948790285,
      'y': -0.569994609360928,
      'name': ''},
     {'$ref': '#/end'}]}

bspline_curve2d_1 = volmdlr.edges.BSplineCurve2D.dict_to_object(dic_bspline)
