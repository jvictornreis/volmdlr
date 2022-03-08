<h1 align="center">
  <img src="https://partage.dessia.tech/thumbnail/7861a783126742be8fe8/1024/Logo_Dessia_transparent_web.png" style="width:300px"><br/>Volmdlr
</h1>

<h4 align="center">
  A computations-oriented python VOLume MoDeLeR with STEP support for import and export
</h4>

<div align="center">
  <a href="http://dessia.tech/"><img src="https://img.shields.io/website-up-down-green-red/http/dessia.tech.svg"></a>  
  <a href="https://GitHub.com/Dessia-tech/volmdlr/stargazers/"><img src="https://badgen.net/github/stars/Dessia-tech/volmdlr"></a>  
  <a href="https://drone-opensource.dessia.tech/Dessia-tech/volmdlr"><img src="https://drone-opensource.dessia.tech/api/badges/Dessia-tech/volmdlr/status.svg?branch=master"></a>
  <a href="https://pypi.org/project/volmdlr/"><img src="https://img.shields.io/pypi/v/volmdlr.svg"></a>
  <a href="https://github.com/Dessia-tech/volmdlr/graphs/contributors"><img src="https://img.shields.io/github/contributors/Dessia-tech/volmdlr.svg"></a>
  <a href="https://github.com/Dessia-tech/volmdlr/issues"><img src="https://img.shields.io/github/issues/Dessia-tech/volmdlr.svg"></a>
</div>

<div align="center">
  <a href="#features"><b>Features</b></a> |
  <a href="#installation"><b>Installation</b></a> |
  <a href="https://github.com/Dessia-tech/volmdlr/tree/master/scripts"><b>Usage</b></a> |
  <a href="https://documentation.dessia.tech/volmdlr/"><b>Documentation</b></a> |
  <a href="#licence"><b>Licence</b></a> |
  <a href="#contributors"><b>Contributors</b></a> |
</div>

## Description

Volmdlr is a python volume modeler used as a CAD platform.
It is simple to understand and operate.
With it, you can easily create 3D models.
Check the exemples to see what you can do with this library.

<img src="https://raw.githubusercontent.com/Dessia-tech/volmdlr/master/doc/source/images/casing.jpg" width="42%" />
<img src="https://raw.githubusercontent.com/Dessia-tech/volmdlr/master/doc/source/images/casing_contours.png" width="57%" /><br/>
<i>A casing is defined by a 2D contour formed with the primitive RoundedLineSegment2D. This contour is offset by the casing width.</i><br/>

<img src="https://raw.githubusercontent.com/Dessia-tech/volmdlr/master/doc/source/images/sweep1.jpg" width="47%" />
<img src="https://raw.githubusercontent.com/Dessia-tech/volmdlr/master/doc/source/images/sweepMPLPlot.jpg" width="52%" /><br/>
<i>A Sweep is pipes, created with Circle2D/Arc2D which is contained in a Contour2D. You have to create the neutral fiber, i.e., the pipe’s road, with the primitive RoundedLineSegment3D.</i><br/>

<img src="https://raw.githubusercontent.com/Dessia-tech/volmdlr/master/doc/source/images/polygon.jpg" width="47%" /><br/>
<i>A polygon is defined out of points. Random points are sampled and the tested whether they are inside or outside the polygon. They are plotted with the Matplotlib binding MPLPlot with custom styles:
- red if they are outside,
- blue if they are inside
</i>
## Features

- [x] Generate 2D and 3D geometries from python
- [x] Handles complexe geometries : B-spline curves and surfaces
- [x] Primitives provide computational tasks : distances, belonging, union, intersections, etc.
- [x] STEP/STL imports and exports
- [x] Geometries display in your web browser with [babylon.js](https://www.babylonjs.com/)

## Requirements

Before using Volmdlr, be sure to have a C/C++ compiler (not necessary on Linux).  
N.B : With Windows you have to download one and allow it to read Python’s code.

## Installation

```bash
pip install volmdlr
# or
pip3 install volmdlr
```

## Usage

See the [script](https://github.com/Dessia-tech/volmdlr/tree/master/scripts) folder for examples

## Documentation

https://documentation.dessia.tech/volmdlr/

## Licence

100% opensource on LGPL licence. See LICENCE for more details.

## Contributors

- [DessiA team](https://github.com/orgs/Dessia-tech/people)
