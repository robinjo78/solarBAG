import numpy as np
import pyvista as pv
import solarpy as sp
from cjio import cityjson
from rtree import index
from pyproj import Proj
import multiprocessing as mp
import datetime as dt
import time
import sys
import math

from multiprocessing.spawn import freeze_support
from helpers.shape_index import create_surface_grid

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# FUNCTIONS GO HERE:

def compute_bbox(list):
    """
    Computes the 3D bounding box of a list of geometries (triangles of a building).
    Returns the maximum and minimum values of the x,y,z coordinates.
    """
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
    """
    Gets the geometry with a specific lod.
    Returns the geometry.
    """
    for g in bdg.geometry:
        if g.lod == lod or g.lod == float(lod):
            return g

def create_rtree(buildings, lod):
    """
    Creates and returns an rtree with buildings.
    """
    # Set properties for the rtree index.
    p = index.Property()
    p.dimension = 3

    # Create empty rtree for citymodel
    rtree_idx = index.Index(properties=p)

    # Loop to dump all buildings in the rtree.
    for i, bdg in enumerate(buildings.values(), 0):
        # Take the LoD 2.2 geometry of the building.
        # print(bdg)
        geom = get_lod(bdg, lod)
        # print(geom)

        # Extract surfaces from the building geometry.
        surfaces = geom.get_surfaces()[0]

        # Compute the bounding box of a building from the surfaces.
        bbox = compute_bbox(surfaces) # Might use cityjson.get_bbox function here (see script Stelios)

        # Insert the bbox into the rtree with the building's id as object (obj).
        rtree_idx.insert(i, bbox, obj=bdg.id)

    print("Size of the Rtree: {}".format(rtree_idx.get_size()))

    return rtree_idx

def compute_graph_area(G):
    """
    Function to compute the area of the graph to get Wh/m^2/day.
    """
    res = 0
    for g in G:
        res += g

    # Divide by four because the intervals are 15 minutes and the computed values are W/h, so we need W/15min and add these 4 up to get W/h.
    # res = res/4

    return res

def irradiance_on_triangles(vnorms, roof_center, proj, date, skip_timestamp_indices=[]):
    """
    Computes the solar irradiance value for a triangle.
    """
    # date = dt.datetime(2019, 7, 5) # TODO make this variable (within a loop)
    h = 0 # TODO make this variable.

    latlon = proj(roof_center[0], roof_center[1], inverse=True)
    lat = latlon[1]

    # TODO: incorporate the skip_timestamp_indices list in the computation below.

    # print(skip_timestamp_indices)

    res_list = []
    for vnorm in vnorms:
        # t = [date + dt.timedelta(minutes=i) for i in range(0, 15 * 24 * 4, 15)]
        t = [date + dt.timedelta(minutes=i) for i in range(0, 60 * 24, 60)]
        G = [sp.irradiance_on_plane(vnorm, h, i, lat) for i in t]
        # If skip_timestamp_indices is not empty, intersections occur for which solar radiation need not be computed.
        if len(skip_timestamp_indices) > 0:
            # print(G, compute_graph_area(G))
            for ind in skip_timestamp_indices:
                G[ind] = 0
            # print(G, compute_graph_area(G))

        res = compute_graph_area(G)
        res_list.append(res)
    
    return res_list

def get_semantic_surfaces(geom, sem_type):
    """
    Extract the semantic surfaces of a certain semantic type (roof, wall or ground) from the geometry.
    """
    surfaces = geom.get_surfaces(type=sem_type)

    # Get the real coordinate boundaries of the surfaces. These are stored as generator objects.
    surface_boundaries = [geom.get_surface_boundaries(s) for s in surfaces.values()]
    
    # Unpack the generator object into a list of surfaces with real coordinates.
    unpacked_surfaces = []
    for g in surface_boundaries:
        unpacked_surfaces.extend(list(g))

    return unpacked_surfaces

def makePolyData_surfaces(input_surfaces):
    """
    Convert surfaces to a Pyvista's PolyData object/mesh.
    """
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

def compute_azimuth(pt1, pt2):
    """
    Computes the azimuth between two points.
    Returns a value in degrees within range [-180,180]
    """
    return (180/math.pi) * math.atan2(pt2[0] - pt1[0], pt2[1] - pt1[1])

def skip_on_azimuth(roof_center, neighbour):
    # roof_bounds = roof.bounds
    # neighbour_bounds = neighbour.bounds

    neighbour_center = neighbour.center_of_mass()

    # Make the north range variable per latitude.
    north_range = [-48, 48]

    azimuth = compute_azimuth(roof_center, neighbour_center)

    if azimuth > north_range[0] and azimuth < north_range[1]:
        return True
    else:
        return False


def find_neighbours(id, roof_mesh, rtree, buildings, lod, offset):
    """
    Find the neighbours corresponding to a building with a certain offset in meters.
    """
    roof_mesh_center = roof_mesh.center_of_mass()

    # Applied the pre-specified offset to each coordinate in both pos and neg directions to get two opposite corners of a bounding box.
    x1 = roof_mesh_center[0] - offset
    x2 = roof_mesh_center[0] + offset
    y1 = roof_mesh_center[1] - offset
    y2 = roof_mesh_center[1] + offset
    z1 = roof_mesh_center[2] - offset
    z2 = roof_mesh_center[2] + offset

    hits = list(rtree.intersection((x1, y1, z1, x2, y2, z2), objects='True'))

    neighbour_list = []

    for item in hits:
        # Get the id of the current item and look this up in de buildings dict of cityjson.
        # Then make a polydata mesh out of it and it it to the list of meshes: the neigbhours.
        # Then the neighbours can be ray traced.
        neighbour_building = buildings[item.object]

        neighbour_geom = get_lod(neighbour_building, lod)
        neighbour_surfaces = neighbour_geom.get_surfaces()[0]
        neighbour_mesh = makePolyData_surfaces(neighbour_surfaces)

        if id == item.object:
            neighbour_list.append(neighbour_mesh)
            continue

        # Skip neighbouring buildings if their z_max is lower than the z_min of the current roof. These neighbours cannot block the current building.
        if roof_mesh.bounds[4] > neighbour_mesh.bounds[5]:
            # print("ID: {}, zmin: {}, zmax: {}".format(item.object, roof_mesh.bounds[4], neighbour_mesh.bounds[5]))
            continue

        if skip_on_azimuth(roof_mesh_center, neighbour_mesh):
            continue

        neighbour_list.append(neighbour_mesh)

    return neighbour_list

def compute_sun_path(point, proj, date):
    """
    Compute the sun path of grid point on the triangle.
    TODO for potential speed-up: compute sun path once and use that for all points? Should take a sun point far enough.
    """
    latlon = proj(point[0], point[1], inverse=True)
    lat = latlon[1]

    t = [date + dt.timedelta(minutes=i) for i in range(0, 60 * 24, 60)]

    # Compute for each point the corresponding position of the sun at a factor x away for a certain time interval.
    vsol_ned = [point + sp.solar_vector_ned(date, lat) * -500 for date in t]

    sun_path = pv.PolyData(vsol_ned)
    return sun_path

def process_building(geom, roof_mesh, neighbours, proj, density, datelist):
    """
    Process a building to enrich it with a solar irradiation value.
    """
    # Extract all surfaces from the building geometry and make it a PolyData object.
    surfaces = geom.get_surfaces()[0]
    mesh = makePolyData_surfaces(surfaces)

    # print(roof_mesh.cell_data)

    # Compute the normals of the roof surface triangles.
    roof_mesh = roof_mesh.compute_normals() # Check whether only triangles are used to compute normals, not points or such.
    vnorms = roof_mesh['Normals']

    # color_array = np.zeros(roof_mesh.n_cells)
    # print(len(color_array))
    # print(roof_mesh.n_cells)
    # color_array.fill(9)
    # roof_mesh['Color'] = color_array

    # print(roof_mesh.cell_data)

    # Sample the triangles into a grid of points. 
    # The lower the density value, the less space will be between the points, increasing the sampling density.
    grid_points = create_surface_grid(roof_mesh, density)
    # NOTE vnorms has the same length as grid_points.

    point_list = []
    # A tuple is a triangle containing its points and the index of the triangle.
    for tuple in grid_points:
        points = tuple[0]
        index = tuple[1]
        print(points)
        print(index)

        vnorm = vnorms[index]

        for point in points:
            point = pv.PolyData(point)
            point.add_field_data(vnorm, "vnorm_tr")

            # TODO add all the field data to point (vnorm, solval, parent etc.).

            # roof_mesh.add_field_data(["Red"], "Color")
            # roof_mesh.add_field_data(85000, "solval")
            # roof_mesh.add_field_data(roof_mesh.center_of_mass(), "Center")
            
            # print(roof_mesh["Color"])
            # print(roof_mesh["solval"])
            # print(roof_mesh["Center"])

            point_list.append(point)

    # print("grid points: ", len(grid_points))
    # print("point list: ", len(point_list))

    # TODO start time loop here. Replace everything before this that does not need to happen in the loop (for example grid_points).
    for date in datelist[:1]:
        # Compute solar irradiation per triangle.
        sol_irr = irradiance_on_triangles(vnorms, roof_mesh.center_of_mass(), proj, date)
        # print(len(sol_irr))

        # TODO use compute sun_path here to compute the sun_path for a day based on the center position of a roof mesh.
        # This should give a speed-up as the sun path is only computed once for each mesh instead of for each sampled point.
        # Important is to take the position of the sun quite far, otherwise values get incorrect.

        solar_roof_grid = []
        intersection_list_mesh = []
        # Process each gridded triangle of the roof of the current building
        for tuple in grid_points:
            points = tuple[0]
            index = tuple[1]

            # Get the solar value belonging to the current triangle.
            sol_val = sol_irr[index]
            vnorm = vnorms[index]

            gridded_triangle = []
            intersection_list_triangle = []
            # Process each point in the current triangle:
            # - create a sun path
            # - perform ray tracing to find a possible intersection between the current point and a point from the sun path
            for point in points:
                sun_path = compute_sun_path(point, proj, date)
                # print(len(sun_path.points))

                # NOTE: ray tracing with its own building always gives back an intersection point? --> NO
                # It did this sometimes as the sampled points did sometimes lie below the triangular surface, meaning an intersection was always found.
                
                intersection_point_list = []
                intersection_index_list = []
                for i, sun_point in enumerate(sun_path.points):
                    # Find a possible intersection point between the current position of the sun and the current point of a triangle.
                    for neighbour in neighbours:
                        intersection_point, _ = neighbour.ray_trace(sun_point, point, first_point=True)
                        if any(intersection_point):
                            intersection_index_list.append(i)
                            intersection_point_list.append(intersection_point)
                            break
                
                # print(len(intersection_index_list), intersection_index_list)
                # Make sol_val different based on the intersections.
                if len(intersection_index_list) > 0:
                    sol_val = irradiance_on_triangles([vnorm], roof_mesh.center_of_mass(), proj, date, skip_timestamp_indices=intersection_index_list)[0]
                
                # TODO to make the date loop work is to keep track of the point in order to add the values of each day in a month into a monthly and daily value.
                point = pv.PolyData(point)
                point.add_field_data(len(intersection_point_list), "intersection count")
                point.add_field_data(sol_val, "solar irradiation")
                
                gridded_triangle.append(point)
                intersection_list_triangle.append(pv.PolyData(intersection_point_list))

            solar_roof_grid.extend(gridded_triangle)
            intersection_list_mesh.extend(intersection_list_triangle)
        # print(len(solar_roof_grid))
        # print(len(intersection_list_mesh))

    return (mesh, solar_roof_grid, intersection_list_mesh)

def vtm_writer_cm(buildings, lod, path):
    """
    This is an extra function that can be used when the user wants to convert the buildings of the citymodel to a mesh beforehand.
    """
    mesh_list = []
    for id in list(buildings.keys()):
        geom = get_lod(buildings[id], lod)
        surfaces = geom.get_surfaces()[0]
        mesh = makePolyData_surfaces(surfaces)
        mesh_list.append(mesh)
    
    mesh_block = pv.MultiBlock(mesh_list)
    mesh_block.save("vtm_objects/{}/meshed_citymodel.vtm".format(path))

def vtm_writer(results, path, write_mesh=False, write_grid=False, write_intersections=False, write_neighbours=False):
    """
    Write the resulting Pyvista objects to several vtm files.
    """
    multiblocks = []
    grid_points = []
    intersection_points = []
    neighbours = []
    for result in results:
        mesh, grid, intersections = result
        # mesh, grid, intersections = result[0]
        
        multiblocks.append(mesh)
        grid_points.extend(grid)
        intersection_points.extend(intersections)
        # neighbours.extend(result[1])

    if write_mesh:
        block = pv.MultiBlock(multiblocks)
        # block.save("vtm_objects/{}/meshed_citymodel.vtm".format(path))
        block.save("vtm_objects/{}/solar_testing/mesh_2.vtm".format(path))
    
    if write_grid:
        grids = pv.MultiBlock(grid_points)
        # grids = grid_points[3]
        # grids.save("vtm_objects/{}/grid_points.vtm".format(path))
        # grids.save("vtm_objects/{}/solar_testing/grid_points_2.vtm".format(path))
        grids.save("vtm_objects/{}/solar_testing/grid_points_3_incorporate_intersections.vtm".format(path))

    if write_intersections:
        intersections_block = pv.MultiBlock(intersection_points)
        intersections_block.save("vtm_objects/{}/intersections.vtm".format(path))

    if write_neighbours:
        neighbours_block = pv.MultiBlock(neighbours)
        neighbours_block.save("vtm_objects/{}/testing/neighbours_single_distance_filter.vtm".format(path))

def process_multiple_buildings(buildings, rtree, cores, proj, lod, offset, density, datelist):
    """
    Start the whole computation pipeline for multiple buildings within multiple processes.
    """
    with ProcessPoolExecutor(max_workers=cores) as pool:
        # The library tqdm is used to display a progress bar in the terminal.
        with tqdm(total=len(buildings)) as progress:
            futures = []

            for id in list(buildings.keys())[:1]:
                geom = get_lod(buildings[id], lod)
                print(id)

                # TODO: look into getting access to verts right away.

                # Extract only roof surfaces from the building geometry.
                roof_mesh = makePolyData_surfaces(get_semantic_surfaces(geom, 'roofsurface'))

                # Find the neighbours of the current mesh according to a certain offset value.
                neighbours = find_neighbours(id, roof_mesh, rtree, buildings, lod, offset)
                # print(id, len(neighbours))
                
                future = pool.submit(process_building, geom, roof_mesh, neighbours, proj, density, datelist)
                future.add_done_callback(lambda p: progress.update())

                futures.append(future)
                # futures.append([future, neighbours])

            results = []

            for future in futures:
                results.append(future.result())
                # results.append([future[0].result(), future[1]])

    return results

# THE MAIN WORKFLOW BELOW:
def main():
    """
    Main.
    """
    # Start_time to keep track of the running time.
    start_time = time.time()

    # Load the CityJSON file from a path which is an argument passed to the python script.
    path = sys.argv[1]
    cm = cityjson.load(path)

    # Adjust the path string for convenient and organised file writing purposes.
    split_list = path.split("/")
    path_string = split_list[len(split_list)-1]

    print("Citymodel is triangulated? ", cm.is_triangulated())

    if cm.is_triangulated():
        # Get the buildings from the city model as dict (there are only buildings).
        buildings = cm.get_cityobjects(type='BuildingPart')
    else:
        print("Hi")
        cm.triangulate()
        buildings = cm.get_cityobjects(type='BuildingPart')
        print("buildings")

    # Extract epsg from the input file and get the projection object.
    epsg = cm.get_epsg()
    proj = Proj(epsg)

    # id = "NL.IMBAG.Pand.0503100000005509-0"

    # Program settings:
    cores = mp.cpu_count()-2
    lod = "2.2"
    neighbour_offset = 150
    sampling_density = 3
    #temporal resolution? Hourly? 10min?

    # Specify list of dates here and give as parameter to process_multiple_buildings:
    date_list = []

    year = 2021
    c_day = 21      # characteristic day of the month
    months = 12     # the time resolution: monthly
    for month in range(months):
        date = dt.datetime(2021, month+1, c_day)
        date_list.append(date)

    # Create rtree for further processing.
    rtree_idx = create_rtree(buildings, lod)

    # Process all buildings in the buildings dictionary.
    results = process_multiple_buildings(buildings, rtree_idx, cores, proj, lod, neighbour_offset, sampling_density, date_list)

    print("Time to run the computations: {} seconds".format(time.time() - start_time))

    vtm_writer(results, path_string, write_mesh=False, write_grid=False)
    
    print("Total time to run this script with writing to file(s): {} seconds".format(time.time() - start_time))

if __name__ == "__main__":
    freeze_support()
    main()