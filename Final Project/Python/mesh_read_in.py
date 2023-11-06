import os
import gmsh
import numpy as np
from mesh_generate import mesh_regen

mesh_regen()

gmsh.initialize()
gmsh.open("airfoil.msh")

gmsh.fltk.run()