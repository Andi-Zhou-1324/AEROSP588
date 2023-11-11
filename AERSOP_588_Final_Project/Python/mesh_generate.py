import os
import gmsh
import numpy as np
def mesh_regen():
    #Loading in the main airfoil coordinates from AE623
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    geometry_dir= os.path.join(project_dir, 'Geometry')
    file_path   = os.path.join(geometry_dir, 'main.txt')

    main_coord = np.loadtxt(file_path) #main airfoik coordinates
    main_coord = np.hstack((main_coord, np.zeros((np.size(main_coord,0),1))))

    #Begin Meshing
    # Initialize the Gmsh API
    gmsh.initialize()

    gmsh.model.add("circle_mesh")

    gmsh_main_coord = []
    for i, point in enumerate(main_coord):
        # gmsh.model.occ.addPoint(x, y, z, meshSize)
        # In OCC, the mesh size is more of a suggestion than a strict requirement
        gmsh_main_coord.append(gmsh.model.occ.addPoint(point[0], point[1], point[2]))

    airfoil_spline = gmsh.model.occ.addBSpline(gmsh_main_coord)
    airfoil_curve = gmsh.model.occ.addCurveLoop([airfoil_spline])

    # Create circle and its curve loop
    center_x, center_y, center_z = 0, 0, 0
    radius = 50
    circle = gmsh.model.occ.addCircle(center_x, center_y, center_z, radius)
    circle_curve = gmsh.model.occ.addCurveLoop([circle])



    # Create surfaces
    airfoil_surface = gmsh.model.occ.addPlaneSurface([airfoil_curve])
    circle_surface = gmsh.model.occ.addPlaneSurface([circle_curve])

    fluid = gmsh.model.occ.cut([(2, circle_surface)], [(2, airfoil_surface)])
    gmsh.model.occ.synchronize()




    # Mesh refinement near the airfoil points
    point_origin = [0.5, 0, 0]  # The origin point

    # Define the center point of refinement
    point_tag = gmsh.model.occ.addPoint(*point_origin)

    # Create the Distance field for the point
    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "PointsList", [point_tag])

    # Create the first Threshold field for the inner circle
    inner_threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(inner_threshold_field, "InField", distance_field)
    gmsh.model.mesh.field.setNumber(inner_threshold_field, "SizeMin", 0.01)
    gmsh.model.mesh.field.setNumber(inner_threshold_field, "SizeMax", 20)
    gmsh.model.mesh.field.setNumber(inner_threshold_field, "DistMin", 0.5)
    gmsh.model.mesh.field.setNumber(inner_threshold_field, "DistMax", 50)  # Set the inner radius


    # Set the Min field as the background field
    gmsh.model.mesh.field.setAsBackgroundMesh(inner_threshold_field)

    gmsh.model.occ.synchronize()


    gmsh.model.mesh.generate(2)



    gmsh.write("airfoil.msh")
    # Synchronize after the boolean operation and deletion
    gmsh.model.occ.synchronize()

    #gmsh.fltk.run()
    gmsh.finalize()
