from math import floor
from multiprocessing.spawn import freeze_support
from pickle import NONE
from matplotlib.colors import ListedColormap
from pyvista import set_plot_theme
import rtree
set_plot_theme('document')

import numpy as np
import pyvista as pv
from cjio import cityjson
from rtree import index
import datetime as dt
import solarpy as sp
import time
import multiprocessing as mp

from shape_index import create_surface_grid

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

# Creates and returns an rtree with buildings.
def create_rtree(buildings, tr_obj):
    # set properties for the rtree index.
    p = index.Property()
    p.dimension = 3

    # Create rtree of citymodel
    rtree_idx = index.Index(properties=p)

    # Loop to dump all buildings in the rtree.
    for bdg in buildings.values():
        # Take the LoD 2.2 geometry of the building.
        geom = bdg.geometry[2]
        
        # Transform the geometry to real coordinates.
        geom_tr = geom.transform(tr_obj)

        # Extract surfaces from the geometry.
        surfaces = geom_tr.get_surfaces()[0]

        # Compute the bounding box of a building from the surfaces.
        bbox = compute_bbox(surfaces)

        # Insert the bbox into the rtree with corresponding id 'fid'.
        rtree_idx.insert(bdg.attributes['fid'], bbox)

    print("Size of the Rtree: {}".format(rtree_idx.get_size()))

    return rtree_idx

# Performs ray tracing to find an intersection between the sun and a building.
def find_intersection():
    # Use the pyvista ray_trace function. (Using the position of the sun at characteristic declination)
    # Look at rendering shadows in the scene: https://docs.pyvista.org/examples/04-lights/shadows.html
    return None

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
    lng = 4.36 # Delft
    h = 0

    # Look at finding position of the sun to do ray tracing.
    dec = sp.declination(date)
    # print("Declination: {} ".format(dec))

    res_list = []
    for vnorm in vnorms:
        t = [date + dt.timedelta(minutes=i) for i in range(0, 15 * 24 * 4, 15)]
        G = [sp.irradiance_on_plane(vnorm, h, i, lat) for i in t]

        # Compute the solar vector in a local geodetic horizon reference frame (NED - North, East, Down).
        # vsol_ned = [sp.solar_vector_ned(date, lat) for date in t]
        # print(vsol_ned[10:25])

        # Convert the solar vector from NED to geocentric coordinates (ECEF)
        # vsol_ecef = [sp.ned2ecef(v, lat, lng) for v in vsol_ned]
        # print(vsol_ecef[10:250])

        res = compute_graph_area(G)
        res_list.append(res)

        # CHECK WHAT INTERVAL TO USE (DON'T just take something (2))
        # area = integrate(G,2)
        # print("area: {}".format(area))
    return res_list

# This function is not used now.
def create_colormap(sol_irr):
    # Create color map for solar irradiation values.
    black = np.array([11/256, 11/256, 11/256, 1])
    green = np.array([0, 255/256, 0, 1])
    yellow = np.array([255/256, 245/256, 0, 1])
    orange = np.array([255/256, 165/256, 0, 1])
    red = np.array([1, 0, 0, 1])

    mapping = np.linspace(np.min(sol_irr), np.max(sol_irr), 256)
    newcolors = np.empty((256, 4))
    newcolors[mapping >= np.max(sol_irr)*0.8] = red
    newcolors[mapping < np.max(sol_irr)*0.8] = orange
    newcolors[mapping < np.max(sol_irr)*0.6] = yellow
    newcolors[mapping < np.max(sol_irr)*0.4] = green
    newcolors[mapping < np.max(sol_irr)*0.05] = black

    color_map = ListedColormap(newcolors)

    return color_map
    
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

def save_sun_path(point):
    date = dt.datetime(2019, 7, 5)
    lat = 52  # Delft

    t = [date + dt.timedelta(minutes=i) for i in range(0, 60 * 24, 60)]

    # Compute for each point the corresponding position of the sun at a factor x away for a certain time interval.
    vsol_ned = [point + sp.solar_vector_ned(date, lat) * -200 for date in t]

    sun_path = pv.PolyData(vsol_ned)
    return sun_path


def process_building(bdg, transformation_object):
    # print("building:", bdg)
    
    geom = bdg.geometry[2]
    # print(geom.surfaces)

    # Transform from indices to the real coordinates/values.
    # NOTE: this can be done at once when loading the city model.
    # geom_tr = geom.transform(transformation_object)

    # Extract all surfaces from the geometry.
    surfaces = geom.get_surfaces()[0]
    mesh = makePolyData_all_surfaces(surfaces)

    # Extract only roof surfaces from the geometry.
    roofs = geom.get_surfaces(type='roofsurface')

    roof_mesh = makePolyData_semantic_surface(roofs, geom)

    # Compute the normals of the roof surface triangles.
    roof_mesh = roof_mesh.compute_normals()

    vnorms = roof_mesh['Normals']

    # Compute solar irradiation per triangle.
    sol_irr = irradiance_on_triangles(vnorms)
    
    # density = 0.001 # for whole buildings
    density = 0.75     # for roofs only

    # Sample the triangles into a grid of points. 
    # The lower the density value, the less space will be between the points, increasing the sampling density.
    grid_points = create_surface_grid(roof_mesh, density)

    # Specify the neigbhouring buildings to check while ray tracing.
    # Add buildings satisfying the criteria in this set.
    # Now a placeholder.
    # neighbours = pv.Sphere()
    neighbours = mesh

    solar_roof_grid = []
    sun_paths_mesh = []
    ray_lists_mesh = []
    intersection_list_mesh = []
    # Process each gridded triangle of the roof of the current building
    for tuple in grid_points:
        points = tuple[0]
        index = tuple[1]

        sol_val = sol_irr[index]

        gridded_triangle = []
        sun_paths_triangle = []
        ray_lists_triangle = []
        intersection_list_triangle = []
        # TODO call the ray tracing function here for each point in the current tuple (triangle)
        for point in points:
            sun_path = save_sun_path(point)
            sun_paths_triangle.append(sun_path)
            # print(sun_path)

            # NOTE: ray tracing with its own building always gives back an intersection point? --> NO
            # It did this sometimes as the sampled points did sometimes lie below the triangular surface, meaning an intersection was always found.
            
            ray_list = []
            intersection_point_list = []
            for p in sun_path.points:
                ray = pv.Line(p, point)
                ray_list.append(ray)

                intersection_point, intersection_cell = neighbours.ray_trace(p, point, first_point=True)

                # print(intersection_point, p)
                if any(intersection_point):
                    intersection_point_list.append(intersection_point)

            point = pv.PolyData(point)
            point.add_field_data(len(intersection_point_list), "intersection count")

            # TODO: make sol_val different based on the intersection count
            point.add_field_data(sol_val, "solar irradiation")
            
            gridded_triangle.append(point)
            # print(intersection_count)
            # print(len(intersection_point_list))

            ray_list = pv.MultiBlock(ray_list)    
            ray_lists_triangle.append(ray_list)
            intersection_list_triangle.append(pv.PolyData(intersection_point_list))
            # print("intersection list: ", intersection_list)

        # gridded_triangle = pv.PolyData(processed_points)
        # sol_val = sol_irr[index]

        # gridded_triangle.add_field_data(sol_val, "solar irradiation")

        # solar_roof_grid.append(pv.MultiBlock(gridded_triangle))
        solar_roof_grid.extend(gridded_triangle)
        sun_paths_mesh.extend(sun_paths_triangle)  
        ray_lists_mesh.extend(ray_lists_triangle)  
        intersection_list_mesh.extend(intersection_list_triangle)
        # print(len(solar_roof_grid))
        # print(len(sun_paths_mesh))
        # print(len(ray_lists_mesh))
        # print(len(intersection_list_mesh))
    
    # print(solar_roof_grid)

    # Add the solar irradiance values as attribute to the surface triangles. This was the previous solution.
    # roof_mesh["Sol value"] = sol_irr

    # print(len(sun_paths_mesh))
    # print(len(ray_lists_mesh))
    # print(len(intersection_list_mesh))

    # return (roof_mesh, floor_mesh, wall_mesh, grid)
    # return (mesh, solar_roof_grid)
    return (mesh, solar_roof_grid, sun_paths_mesh, ray_lists_mesh, intersection_list_mesh)
    # b_i = 81
    # return (mesh, solar_roof_grid, sun_paths_mesh[b_i], ray_lists_mesh[b_i], intersection_list_mesh[b_i])

def test_one_building(buildings, tr_obj, start_time):
    # Take out one building.
    # fid = "254" # This is just a block/cube
    # fid = "18918"
    fid = "25774"
    # fid = "138790"
    bdg = buildings[fid]

    # Process one building. Compute the necessary attributes for the surfaces and store in mesh.
    # roof, wall, floor, grid = process_building(bdg, tr_obj)
    # mesh_block = pv.MultiBlock((roof, floor, wall, grid))
    mesh, grid, sun_path, ray_list, intersection_list = process_building(bdg, tr_obj)
    # mesh_block = pv.MultiBlock((mesh, pv.MultiBlock(grid), pv.MultiBlock(sun_path), pv.MultiBlock(ray_list)))
    mesh_block = pv.MultiBlock((mesh, pv.MultiBlock(grid), pv.MultiBlock(intersection_list)))
    # print(intersection_list.points)
    # print(grid[81].points)

    # Save the mesh to vtk format.
    # mesh_block.save("mesh_sol_grid_sun_path_intersections.vtm")
    mesh_block.save("mesh_intersections_test.vtm")

    # Print the time the script took.
    print("Time to run this script: {} seconds".format(time.time() - start_time))

# Now, it is approximately 4x faster with multiprocessing because I do not transform the whole dataset to real coordinates anymore within the loop.
def test_multiple_buildings(buildings, tr_obj, start_time):
    bdg_list = list(buildings.keys())[:50]

    # Process all buildings in a list of buildings by using list comprehension with multiprocessing.
    # Start parallelisation:
    # pool = mp.Pool(mp.cpu_count()-2)
    # result = pool.map(process_building, [(count, buildings[fid], tr_obj) for count, fid in enumerate(bdg_list, 1)])
    # pool.close()

    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm

    with ProcessPoolExecutor(max_workers= mp.cpu_count()-2) as pool:
        with tqdm(total=len(bdg_list)) as progress:
            futures = []

            for count, fid in enumerate(bdg_list, 1):
                future = pool.submit(process_building, buildings[fid], tr_obj)
                future.add_done_callback(lambda p: progress.update())
                
                # roof, wall, floor, grid = future.result()
                # mesh_block = pv.MultiBlock((roof, floor, wall, grid))
                mesh, grid = future.result()
                mesh_block = pv.MultiBlock((mesh, pv.MultiBlock(grid)))
                # mesh, grid, sun_path = future.result()
                # mesh_block = pv.MultiBlock((mesh, pv.MultiBlock(grid), pv.MultiBlock(sun_path)))
                
                futures.append(mesh_block)

            # print(futures)

            block = pv.MultiBlock(futures)
            block.save("citymodel_sol_grid.vtm")

            # for future in futures:
            #     print(future)

    # result = [process_building(count, buildings[fid], tr_obj) for count, fid in enumerate(bdg_list, 1)]
    
    print("Time to run this script: {} seconds".format(time.time() - start_time))

# THE MAIN WORKFLOW BELOW:
def main():
    # start_time to keep track of the running time.
    start_time = time.time()

    # Load the CityJSON file from a path.
    # path = "C:\\Users\\hurkm\\Documents\\TU\\Geomatics\\Jaar 2\\GEO2020 - Msc Thesis\\Data\\CityJSON_tiles\\3dbag_v21031_7425c21b_3007.json"
    # linux path:
    path = "/mnt/c/Users/hurkm/repos/solarBAG/data/3dbag_v21031_7425c21b_3007.json"
    cm = cityjson.load(path)

    # Transform from indices to the real coordinates/values.
    # NOTE: this can be done at once when loading the city model. Check whether I want this or keep the indices? https://cjio.readthedocs.io/en/stable/api_tutorial_basics.html#Load-the-city-model
    transformation_object = cm.transform

    # Get the buildings from the city model as dict (there are only buildings).
    buildings = cm.get_cityobjects(type='building')

    # Create rtree for further processing
    rtree_idx = create_rtree(buildings, transformation_object)
    print(type(rtree_idx))

    # HOW TO find suitable neighbours:
    # Use a query window to find the neighbours I want in the rtree
    # list(rtree_idx.intersection((x1, y1, z1, x2, y2, z2)))
    # Take x, y, z values as the x, y, z values of the point + the offset (100 m?) I am processing

    # Call functions that manipulate the geometries
    test_one_building(buildings, transformation_object, start_time)
    # test_multiple_buildings(buildings, transformation_object, start_time)

if __name__ == "__main__":
    freeze_support()
    main()