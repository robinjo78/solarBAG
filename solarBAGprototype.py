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

# Not functional yet. Look into script of Stelios.
# Or skip sampling and make use of computing and projecting shadows right away.
# Samples a triangle into a regular grid of points.
def sample_triangle(A, B, C):
    y_A = A[1]
    x_B = B[0]

    x_diff = np.abs(A[0] - B[0])
    # print(x_diff)

    offset_pt = [x_B, y_A, A[2]]
    return offset_pt

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
    h = 0

    # Look at finding position of the sun to do ray tracing.
    dec = sp.declination(date)
    # print("Declination: {} ".format(dec))

    res_list = []
    for vnorm in vnorms:
        t = [date + dt.timedelta(minutes=i) for i in range(0, 15 * 24 * 4, 15)]
        G = [sp.irradiance_on_plane(vnorm, h, i, lat) for i in t]

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
    
def makePolyData(input_surfaces, geom):
    # Get the real coordinate boundaries of the surfaces.
    surfaces = [geom.get_surface_boundaries(s) for s in input_surfaces.values()]
    # print(surfaces)

    # Unpack the generator object into a list of surfaces.
    unpacked_surfaces = []
    for g in surfaces:
        unpacked_surfaces.extend(list(g))

    vlist = []
    flist = []
    i = 0
    # This is done for all surfaces, not only roof surfaces.
    # This way is not efficient. Find a way without duplicate vertices.
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

    # print(vlist, flist)

    # Transform the vertex and face lists to pyvista's PolyData format.
    mesh = pv.PolyData(vlist,flist)
    # print("Number of points in mesh:", mesh.n_points)
    # print("Points in mesh:", mesh.points)
    # print("Indices:", mesh.surface_indices)

    # Clean the data from duplicates.
    mesh = mesh.clean()

    return mesh

def process_building(count, bdg, transformation_object):
    print("building:", bdg)
    
    geom = bdg.geometry[2]
    # print(geom.surfaces)

    # Transform from indices to the real coordinates/values.
    # NOTE: this can be done at once when loading the city model.
    geom_tr = geom.transform(transformation_object)

    # Extract all surfaces from the geometry.
    # surfaces = geom_tr.get_surfaces()[0]

    # Extract only roof surfaces from the geometry.
    roofs = geom.get_surfaces(type='roofsurface')
    floors = geom.get_surfaces(type='groundsurface')
    walls = geom.get_surfaces(type='wallsurface')

    roof_mesh = makePolyData(roofs, geom)
    floor_mesh = makePolyData(floors, geom)
    wall_mesh = makePolyData(walls, geom)

    # print("Mesh: ", mesh)
    # print("Number of points in roof mesh after cleaning:", roof_mesh.n_points)
    
    # density = 0.001 # for whole buildings
    density = 0.75     # for roofs only

    # Sample the triangles into a grid of points. 
    # The lower the density value, the less space will be between the points, increasing the sampling density.
    grid = create_surface_grid(roof_mesh, density)
    # print("Length of grid:", len(grid))
    # print("Grid:", grid)

    grid_points = []
    grid_indices = []
    for g in grid:
        grid_points.extend(g[0])
        grid_indices.append(g[1])

    # print(grid_points)
    # print(grid_indices)

    # FIND OUT HOW TO STORE THE LINK TO THE TRIANGLE IN EACH GRID POINT.
    grid = pv.PolyData(grid_points)

    roof_mesh = roof_mesh.merge(grid)

    # Compute the normals of the surface triangles.
    roof_mesh = roof_mesh.compute_normals()
    # print(mesh["Normals"])
    # FIND OUT HOW TO ACCESS AND UPDATE THE ATTRIBUTES LIKE THE NORMALS. 

    # TODO put the values of the normals to each point in the corresponding triangle.


    vnorms = roof_mesh['Normals']

    # Compute solar irradiation per triangle.
    sol_irr = irradiance_on_triangles(vnorms)
    # print(sol_irr, count)

    # Add the solar irradiance values as attribute to the surface triangles
    # This is used at visualisation.
    roof_mesh["Sol value"] = sol_irr

    # mesh = floor_mesh.merge((wall_mesh, roof_mesh))
    # mesh = roof_mesh.boolean_union(floor_mesh)

    mesh_block = pv.MultiBlock((roof_mesh, floor_mesh, wall_mesh))

    print("Processed building: {}, fid: {}".format(count, bdg.id))

    return mesh_block
    # return mesh

def test_one_building(buildings, tr_obj, start_time):
    # Take out one building.
    # fid = "254" # This is just a block/cube
    # fid = "18918"
    fid = "25774"
    # fid = "138790"
    bdg = buildings[fid]

    # Process one building. Compute the necessary attributes for the surfaces and store in mesh.
    mesh_block = process_building(1, bdg, tr_obj)

    # Save the mesh to vtk format.
    mesh_block.save("mesh.vtm")

    # Print the time the script took.
    print("Time to run this script: {} seconds".format(time.time() - start_time))

# Now, it is approximately 4x faster with multiprocessing because I do not transform the whole dataset to real coordinates anymore within the loop.
def test_multiple_buildings(buildings, tr_obj, start_time):
    bdg_list = list(buildings.keys())[:20]

    # Process all buildings in a list of buildings by using list comprehension with multiprocessing.
    # Start parallelisation:
    pool = mp.Pool(mp.cpu_count()-2)
    result = pool.starmap(process_building, [(count, buildings[fid], tr_obj) for count, fid in enumerate(bdg_list, 1)])
    pool.close()

    block = pv.MultiBlock(result)
    block.save("cm.vtm")

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
    # rtree_idx = create_rtree(buildings, transformation_object)

    # Call functions that manipulate the geometries
    test_one_building(buildings, transformation_object, start_time)
    # test_multiple_buildings(buildings, transformation_object, start_time)

if __name__ == "__main__":
    freeze_support()
    main()