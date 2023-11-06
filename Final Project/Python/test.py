import gmsh
import sys

# Initialize gmsh and add a new model
gmsh.initialize(sys.argv)
gmsh.model.add('SquareTest')

rectangle = gmsh.model.occ.addRectangle(-1,-1,0,2,2)
rectangle_loop = gmsh.model.occ.addPlaneSurface(rectangle)
gmsh.model.occ.synchronize()

# Mesh the rectangle
gmsh.model.mesh.generate(2)


# Display the mesh
gmsh.fltk.run()

# Finalize gmsh
gmsh.finalize()