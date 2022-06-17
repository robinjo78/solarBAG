import numpy as np
import pyvista as pv
import solarpy as sp
from cjio import cityjson
from rtree import index
import multiprocessing as mp
import datetime as dt
import time
import vtk

from multiprocessing.spawn import freeze_support
from helpers.shape_index import create_surface_grid

# FUNCTIONS GO HERE:
# Computes the bbox of a list of geometries (triangles of a building)
def compute_bbox(list):
    xmin = list[0][0][0][0]
    ymin = list[0][0][0][1]
    zmin = list[0][0][0][2]
    xmax = xmin
    ymax = ymin
    zmax = zmin

    for tr in list:
        for point in tr[0]:
            x = point[0]
            y = point[1]
            z = point[2]

            if x < xmin:
                xmin = x
            elif x > xmax:
                xmax = x
            if y < ymin:
                ymin = y
            elif y > ymax:
                ymax = y
            if z < zmin:
                zmin = z
            elif z > zmax:
                zmax = z

    return [xmin, ymin, zmin, xmax, ymax, zmax]

def get_lod(bdg, lod):
    for g in bdg.geometry:
        if g.lod == lod:
            return g

# Creates and returns an rtree with buildings.
def create_rtree(buildings):
    # set properties for the rtree index.
    p = index.Property()
    p.dimension = 3

    # Create rtree of citymodel
    rtree_idx = index.Index(properties=p)

    # Loop to dump all buildings in the rtree.
    for i, bdg in enumerate(buildings.values(), 0):
        # Take the LoD 2.2 geometry of the building.
        # geom = bdg.geometry[2]
        geom = get_lod(bdg, "2.2")

        # Extract surfaces from the geometry.
        # surfaces = geom.get_surfaces()
        surfaces = geom.get_surfaces()[0]

        # Compute the bounding box of a building from the surfaces.
        bbox = compute_bbox(surfaces) # Might use cityjson.get_bbox function here (see script Stelios)

        # Insert the bbox into the rtree with corresponding id 'fid'.
        rtree_idx.insert(i, bbox, bdg.id)

    print("Size of the Rtree: {}".format(rtree_idx.get_size()))

    return rtree_idx

# Function to compute the area of the graph to get Wh/m^2/day.
def compute_graph_area(G):
    res = 0
    for g in G:
        res += g

    # Divide by four because the intervals are 15 minutes and the computed values are W/h, so we need W/15min and add these 4 up to get W/h.
    res = res/4

    return res

def irradiance_on_triangles(vnorms):
    date = dt.datetime(2019, 7, 5)
    lat = 52  # Delft
    h = 0

    res_list = []
    for vnorm in vnorms:
        t = [date + dt.timedelta(minutes=i) for i in range(0, 15 * 24 * 4, 15)]
        G = [sp.irradiance_on_plane(vnorm, h, i, lat) for i in t]

        res = compute_graph_area(G)
        res_list.append(res)
    
    return res_list
    
def makePolyData_semantic_surface(input_surfaces, geom):
    # Get the real coordinate boundaries of the surfaces. These are stored as generator objects.
    surfaces = [geom.get_surface_boundaries(s) for s in input_surfaces.values()]

    # Unpack the generator object into a list of surfaces with real coordinates.
    unpacked_surfaces = []
    for g in surfaces:
        unpacked_surfaces.extend(list(g))

    vlist = []
    flist = []
    i = 0
    # Index the surfaces and store as vertex and face lists. This is done for all surfaces.
    # - Take the original way it is stored in CityJSON. Probably not possible. Is another way of indexing.
    for boundary in unpacked_surfaces:
        plane = boundary[0]
        v1 = plane[0]
        v2 = plane[2]
        v3 = plane[1]

        vlist.append(v1)
        vlist.append(v2)
        vlist.append(v3)
        flist.append([3, i, i+1, i+2])
        i+=3

    # Transform the vertex and face lists to pyvista's PolyData format.
    mesh = pv.PolyData(vlist,flist)

    # Clean the data from duplicates.
    mesh = mesh.clean()

    return mesh

def makePolyData_all_surfaces(input_surfaces):
    vlist = []
    flist = []
    i = 0
    # Index the surfaces and store as vertex and face lists. This is done for all surfaces.
    # - Take the original way it is stored in CityJSON. Probably not possible. Is another way of indexing.
    for boundary in input_surfaces:
        plane = boundary[0]
        v1 = plane[0]
        v2 = plane[2]
        v3 = plane[1]

        vlist.append(v1)
        vlist.append(v2)
        vlist.append(v3)
        flist.append([3, i, i+1, i+2])
        i+=3

    # Transform the vertex and face lists to pyvista's PolyData format.
    mesh = pv.PolyData(vlist,flist)

    # Clean the data from duplicates.
    mesh = mesh.clean()

    return mesh

# Find the neighbours corresponding to a building with a certain offset in meters
def find_neighbours(roof_mesh, rtree, buildings, offset):
    roof_mesh_center = roof_mesh.center_of_mass()
    print(roof_mesh_center)

    # Applied the pre-specified offset to each coordinate in both pos and neg directions to get two opposite corners of a bounding box.
    x1 = roof_mesh_center[0] - offset
    x2 = roof_mesh_center[0] + offset
    y1 = roof_mesh_center[1] - offset
    y2 = roof_mesh_center[1] + offset
    z1 = roof_mesh_center[2] - offset
    z2 = roof_mesh_center[2] + offset

    hits = list(rtree.intersection((x1, y1, z1, x2, y2, z2), objects='True'))
    # hits = list(rtree.intersection((x1, y1, z1, x2, y2, z2), objects='raw'))
    # hits = list(rtree.intersection((x1, y1, z1, x2, y2, z2)))

    # print(hits)

    neighbour_list = []

    for item in hits:
        # Get the id of the current item and look this up in de buildings dict of cityjson
        # Then make a polydata mesh out of it and it it to the list of meshes: the neigbhours.
        # Then the neighbours can be ray traced.
        # neighbour_building = buildings[str(item.id)]
        neighbour_building = buildings[item.object]

        neighbour_geom = neighbour_building.geometry[2]
        neighbour_surfaces = neighbour_geom.get_surfaces()[0]
        neighbour_mesh = makePolyData_all_surfaces(neighbour_surfaces)

        neighbour_list.append(neighbour_mesh)
    
    if any(hits):
        neighbours = neighbour_list[0].merge(neighbour_list[1:])
        return neighbours
    else:
        # Should be building itself? OR should not happen at all??
        neighbours = roof_mesh
        return neighbours

def compute_sun_path(point):
    date = dt.datetime(2019, 7, 5)
    lat = 52  # Delft

    t = [date + dt.timedelta(minutes=i) for i in range(0, 60 * 24, 60)]

    # Compute for each point the corresponding position of the sun at a factor x away for a certain time interval.
    vsol_ned = [point + sp.solar_vector_ned(date, lat) * -500 for date in t]

    sun_path = pv.PolyData(vsol_ned)
    return sun_path

def process_building(bdg, roof_mesh, neighbours):
    geom = get_lod(bdg, "2.2")

    # Extract all surfaces from the geometry.
    surfaces = geom.get_surfaces()[0]
    mesh = makePolyData_all_surfaces(surfaces)

    # Extract only roof surfaces from the geometry.
    # roofs = geom.get_surfaces(type='roofsurface')
    # roof_mesh = makePolyData_semantic_surface(roofs, geom)

    # Compute the normals of the roof surface triangles.
    roof_mesh = roof_mesh.compute_normals()
    vnorms = roof_mesh['Normals']

    # Compute solar irradiation per triangle.
    sol_irr = irradiance_on_triangles(vnorms)
    
    # density = 0.001 # for whole buildings
    # density = 0.75     # for roofs only
    density = 3

    # Sample the triangles into a grid of points. 
    # The lower the density value, the less space will be between the points, increasing the sampling density.
    grid_points = create_surface_grid(roof_mesh, density)

    solar_roof_grid = []
    # sun_paths_mesh = []
    # ray_lists_mesh = []
    intersection_list_mesh = []
    # Process each gridded triangle of the roof of the current building
    for tuple in grid_points:
        points = tuple[0]
        index = tuple[1]

        # Get the solar value belonging to the current triangle.
        sol_val = sol_irr[index]

        gridded_triangle = []
        # sun_paths_triangle = []
        # ray_lists_triangle = []
        intersection_list_triangle = []
        # Process each point in the current triangle:
        # - create a sun path
        # - perform ray tracing to find a possible intersection between the current point and a point from the sun path
        for point in points:
            sun_path = compute_sun_path(point)
            # sun_paths_triangle.append(sun_path)

        #     # NOTE: ray tracing with its own building always gives back an intersection point? --> NO
        #     # It did this sometimes as the sampled points did sometimes lie below the triangular surface, meaning an intersection was always found.
            
        #     # ray_list = []
            intersection_point_list = []
            for sun_point in sun_path.points:
                # ray = pv.Line(sun_point, point)
        #         # ray_list.append(ray)

        #         # Find a possible intersection point between the current position of the sun and the current point of a triangle.
                intersection_point, _ = neighbours.ray_trace(sun_point, point, first_point=True) # The ray_trace function CAUSES the pickling error.
                # intersection_point = np.array(sun_point)
                if any(intersection_point):
                    intersection_point_list.append(intersection_point)

            point = pv.PolyData(point)
            point.add_field_data(len(intersection_point_list), "intersection count")

        #     # TODO: make sol_val different based on the intersection count
            point.add_field_data(sol_val, "solar irradiation")
            
            gridded_triangle.append(point)

        #     # ray_list = pv.MultiBlock(ray_list)    
        #     # ray_lists_triangle.append(ray_list)
            intersection_list_triangle.append(pv.PolyData(intersection_point_list))

        solar_roof_grid.extend(gridded_triangle)
        # sun_paths_mesh.extend(sun_paths_triangle)  
        # ray_lists_mesh.extend(ray_lists_triangle)  
        intersection_list_mesh.extend(intersection_list_triangle)

    # meshes = [mesh, solar_roof_grid, neighbours]

    # vtkstrings = []

    # w = vtk.vtkPolyDataWriter()
    # w.WriteToOutputStringOn()
    # for mesh in meshes:
    #     w.SetInputData(mesh)
    #     w.Update()
    #     w.WriteToOutputStringOn() 
    #     vtkstrings.append(w.GetOutputString())

    # print(vtkstrings[0])
    # return (mesh, solar_roof_grid, sun_paths_mesh, ray_lists_mesh, intersection_list_mesh, neighbours)
    return (mesh, solar_roof_grid, intersection_list_mesh)
    # return (mesh, solar_roof_grid, neighbours)
    # return (mesh, solar_roof_grid)

def test_one_building(buildings, rtree, start_time):
    # Take out one building.
    # fid = "25774"
    # bdg = buildings[fid]
    id = "NL.IMBAG.Pand.0503100000005509-0"
    # id = "NL.IMBAG.Pand.0503100000000018-0"
    # id2 = "NL.IMBAG.Pand.0503100000031377-0"
    # id3 = "NL.IMBAG.Pand.0503100000031378-0"

    # bdg_list = [id2, id3]

    # blocks = []
    # for id in bdg_list:
    #     mesh, grid, intersection_list, neighbours = process_building(buildings[id], rtree, buildings, tr_obj)
    #     mesh_block = pv.MultiBlock((mesh, pv.MultiBlock(grid), pv.MultiBlock(intersection_list), neighbours))
    #     # mesh_block.save("mesh_test_{}.vtm".format(id))
    #     blocks.append(mesh_block)

    # block = pv.MultiBlock((blocks))

    # block.save("two_meshes_test.vtm")


    bdg = buildings[id]

    # Process one building. Compute the necessary attributes for the surfaces and store in mesh.
    # mesh, grid, sun_path, ray_list, intersection_list, neighbours = process_building(bdg, rtree, buildings
    # mesh, grid, intersection_list, neighbours = process_building(bdg, rtree, buildings)
    mesh, grid, neighbours = process_building(bdg, rtree, buildings)
    print("Time to run this script so far: {} seconds".format(time.time() - start_time))

    # mesh_block = pv.MultiBlock((mesh, pv.MultiBlock(grid), pv.MultiBlock(intersection_list), neighbours))
    # mesh_block = pv.MultiBlock((mesh, pv.MultiBlock(grid), neighbours))
    mesh_block = pv.MultiBlock((mesh, pv.MultiBlock(grid)))
    mesh_block.save("vtm_objects/mesh_intersections_test_{}.vtm".format(id))


    # mesh_block = pv.MultiBlock((mesh, pv.MultiBlock(grid), pv.MultiBlock(intersection_list), neighbours, pv.MultiBlock(ray_list), pv.MultiBlock(sun_path)))


    # Save the mesh to vtk format.
    # mesh_block.save("vtm_objects/mesh_intersections_test_neighbours_3dbag_update_2.vtm")

    # Print the time the script took.
    print("Time to run this script to the end: {} seconds".format(time.time() - start_time))


def functionWithPickableInput(inputstring0):
    r0 = vtk.vtkPolyDataReader()
    r0.ReadFromInputStringOn()
    r0.SetInputString(inputstring0 )
    r0.Update()
    polydata0 = r0.GetOutput()
    return polydata0
    #compute the strings to use as input (they are the content of the correspondent vtk file)

# Now, it is approximately 4x faster with multiprocessing because I do not transform the whole dataset to real coordinates anymore within the loop.
def test_multiple_buildings(buildings, rtree, start_time):
    bdg_list = list(buildings.keys())[:2]

    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    from tqdm import tqdm

    futures = []

    with ProcessPoolExecutor(max_workers= mp.cpu_count()-2) as pool:
        print(mp.cpu_count())
        # The library tqdm is used to display a progress bar in the terminal.
        with tqdm(total=len(bdg_list)) as progress:
            # futures = []

            for id in bdg_list:
                print(id)
                geom = get_lod(buildings[id], "2.2")

                # TODO: look into getting access to verts right away.

                # Extract only roof surfaces from the geometry.
                roofs = geom.get_surfaces(type='roofsurface')
                roof_mesh = makePolyData_semantic_surface(roofs, geom)

                # Find the neighbours of the current mesh according to a certain offset value.
                neighbours = find_neighbours(roof_mesh, rtree, buildings, 150)
                # print(neighbours)
                
                future = pool.submit(process_building, buildings[id], roof_mesh, neighbours)
                future.add_done_callback(lambda p: progress.update())
                
                # roof, wall, floor, grid = future.result()
                # mesh_block = pv.MultiBlock((roof, floor, wall, grid))
                mesh, grid, intersections = future.result() # CHECK WHAT ERROR HAPPENS HERE! --> pickling intersections gives the error.
                
                # mesh, grid = future.result()
                # mesh, grid, neighbours = future.result()        # Pickling mesh, grid and neighbours is possible.
                                                                # BUT neighbours list is empty when using ProcessPoolExecutor, so should fix this.
                                                                # DEBUG for test_one_building function and this function when finding the neigbhours.
                # futures.append(future)

                # mesh, grid = future.result()
                mesh_block = pv.MultiBlock((mesh, pv.MultiBlock(grid), pv.MultiBlock(intersections), neighbours))
                # mesh_block = pv.MultiBlock((mesh, pv.MultiBlock(grid), neighbours))
                # mesh_block = pv.MultiBlock((mesh, pv.MultiBlock(grid))) # Don't export neighbours to make it more efficient.
                                                                        # Might choose to create polydata of whole buildings list at once.
                # mesh, grid, sun_path = future.result()
                # mesh_block = pv.MultiBlock((mesh, pv.MultiBlock(grid), pv.MultiBlock(sun_path)))
                
                futures.append(mesh_block)

    # print(futures)

    # results = []

    # for future in futures:
    #     # mesh, grid, neighbours = future.result()
    #     mesh, grid = future.result()
    #     # mesh_block = pv.MultiBlock((mesh, pv.MultiBlock(grid), neighbours))
    #     mesh_block = pv.MultiBlock((mesh, pv.MultiBlock(grid)))
    #     results.append(mesh_block)

    # buildings_mesh = makePolyData_all_surfaces(.get_surfaces()[0])
    # print(buildings_mesh)

    block = pv.MultiBlock(futures)
    # block = pv.MultiBlock(results)
    block.save("vtm_objects/citymodel_sol_grid_3dbag_update_test_2.vtm")

    # result = [process_building(count, buildings[fid], tr_obj) for count, fid in enumerate(bdg_list, 1)]
    
    print("Time to run this script: {} seconds".format(time.time() - start_time))

# THE MAIN WORKFLOW BELOW:
def main():
    # start_time to keep track of the running time.
    start_time = time.time()

    # Load the CityJSON file from a path.
    # path = "/mnt/c/Users/hurkm/repos/solarBAG/data/3dbag_v21031_7425c21b_3007_v11.city.json" # a linux path
    path = "/mnt/c/Users/hurkm/repos/solarBAG/data/3dbag_v210908_fd2cee53_3007_new_v11_triangulated.city.json" # a linux path

    cm = cityjson.load(path)

    # Get the buildings from the city model as dict (there are only buildings).
    buildings = cm.get_cityobjects(type='BuildingPart')

    # Create rtree for further processing
    rtree_idx = create_rtree(buildings)
    # rtree_idx = index.Index()

    # Call functions that manipulate the geometries
    # test_one_building(buildings, rtree_idx, start_time)
    test_multiple_buildings(buildings, rtree_idx, start_time)

if __name__ == "__main__":
    freeze_support()
    main()