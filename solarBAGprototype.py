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
from helpers.geometry import surface_normal

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

def compute_daily_irradiance(G):
    """
    Function to compute the area of the graph to get Wh/m^2/day.
    """
    res = sum(G)

    # Divide by four because the intervals are 15 minutes and the computed values are W/h, so we need W/15min and add these 4 up to get W/h.
    # res = res/4

    return res

def irradiance_on_triangle(vnorm, h, date, lat, skip_timestamp_indices=[]):
    """
    Computes the solar irradiance value for a triangle.
    """
    # t = [date + dt.timedelta(minutes=i) for i in range(0, 15 * 24 * 4, 15)]
    t = [date + dt.timedelta(minutes=i) for i in range(0, 60 * 24, 60)]
    G = [sp.irradiance_on_plane(vnorm, h, i, lat) for i in t]
    # If skip_timestamp_indices is not empty, intersections occur for which solar radiation need not be computed.
    if len(skip_timestamp_indices) > 0:
        # print(G, compute_graph_area(G))
        for ind in skip_timestamp_indices:
            G[ind] = 0

    # print(G)
    res = compute_daily_irradiance(G)
    # print("res: ", res)
    
    return res

def compute_perimeter(triangle):
    "Computes the perimeter of a triangle"
    p1 = triangle[0]
    p2 = triangle[1]
    p3 = triangle[2]

    perimeter = math.dist(p1, p2) + math.dist(p2, p3) + math.dist(p3, p1)
    return perimeter

def remove_slivers(surfaces):
    """
    Removes sliver triangle from the surface list
    Based on: https://math.stackexchange.com/questions/1336265/explanation-of-the-thinness-ratio-formula
    """
    surfaces_to_keep = []
    for surface in surfaces:
        pd_surface = makePolyData_surfaces([surface])

        # compute area and perimeter of triangle.
        area = pd_surface.area # can also compute myself?
        perimeter = compute_perimeter(surface[0])

        # compute area and perimeter of circle with same perimeter as triangle.
        r_circle = perimeter/(2*math.pi)
        area_circle = math.pi * r_circle
        
        # compute the thinness ratio.
        thinness = area/area_circle

        # Remove triangles below a certain threshold.
        if thinness > 0.001:
            surfaces_to_keep.append(surface)

    return surfaces_to_keep


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
        # Get the id of the current item and look this up in the buildings dict of cityjson.
        # Then make a polydata mesh out of it and add it to the list of meshes: the neigbhours.
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

def get_lat(proj, point):
    latlon = proj(point[0], point[1], inverse=True)
    lat = latlon[1]

    return lat

def compute_sun_path(point, proj, date):
    """
    Compute the sun path of grid point on the triangle.
    TODO for potential speed-up: compute sun path once and use that for all points? Should take a sun point far enough.
    """
    # latlon = proj(point[0], point[1], inverse=True)
    # lat = latlon[1]
    
    lat = get_lat(proj, point)

    t = [date + dt.timedelta(minutes=i) for i in range(0, 60 * 24, 60)]

    # Compute for each point the corresponding position of the sun at a factor x away for a certain time interval.
    vsol_ned = [point + sp.solar_vector_ned(date, lat) * -500 for date in t]

    sun_path = pv.PolyData(vsol_ned)
    return sun_path

def compute_yearly_solar_irradiance_point(point, datelist):
    # print(point["sol_val_list_day_in_month"])

    sol_val_list_whole_month = []
    for sol_val, date in zip(point["sol_val_list_day_in_month"], datelist):
        sol_val_month = sol_val * date[1]
        sol_val_list_whole_month.append(sol_val_month)
    # print(sol_val_list_whole_month)

    sol_val_year = sum(sol_val_list_whole_month)
    # print(sol_val_year)

    point.add_field_data(sol_val_list_whole_month, "sol_val_list_whole_month")
    point.add_field_data(sol_val_year, "sol_val_year")


def process_building(geom, roof_mesh, neighbours, proj, density, datelist):
    """
    Process a building to enrich it with a solar irradiation value.
    """
    surfaces = geom.get_surfaces()[0]

    # Get the roofsurfaces and remove slivers. 
    # DOES NOT WORK YET --> LOADING IN PARAVIEW GIVES ERROR
    # roof_surfaces = get_semantic_surfaces(geom, 'roofsurface')
    # roof_surfaces_clean = remove_slivers(roof_surfaces)
    # roof_mesh_clean = makePolyData_surfaces(roof_surfaces_clean)

    mesh = makePolyData_surfaces(surfaces)                          # The mesh in PolyData

    # roof_mesh = roof_mesh.compute_normals(point_normals=True, flip_normals=False, auto_orient_normals=False)
    roof_mesh = roof_mesh.compute_normals()
    vnorms = roof_mesh['Normals']                                   # The normal vectors of all triangles in a roof
    # vnorms = roof_mesh.cell_data["Normals"]
    # print(vnorms)

    # roof_mesh_clean = roof_mesh_clean.compute_normals()
    # vnorms = roof_mesh_clean['Normals']                                   # The normal vectors of all triangles in a roof
    # print(vnorms)

    # vnorms = roof_mesh_clean.point_data["Normals"]
    # print(roof_mesh_clean.point_data["Normals"])
    # print(roof_mesh_clean.cell_data["Normals"])

    # COMPUTE NORMAL VECTOR ALTERNATIVELY
    # roof_surfaces = get_semantic_surfaces(geom, 'roofsurface')
    # vnorms_alt = []
    # for boundary in roof_surfaces:
    #     plane = boundary[0]
    #     vnorm_alt = surface_normal(plane)
    #     vnorms_alt.append(vnorm_alt)

    # print(list(zip(vnorms, vnorms_alt)))

    lat = get_lat(proj, roof_mesh.center_of_mass())
    # lat = get_lat(proj, roof_mesh_clean.center_of_mass())

    grid_points = create_surface_grid(roof_mesh, density)           # Each triangle enriched with grid points
    # grid_points = create_surface_grid(roof_mesh_clean, density)           # Each triangle enriched with grid points

    solar_roof_grid = []
    # Process each gridded triangle of the roof of the current building
    for tup in grid_points:
        points = tup[0]
        ind = tup[1]

        # the resulting vnorm is in ENU frame, point inward.
        vnorm = vnorms[ind]

        # To get the vnorm in NED frame:
        # - let the vnorm point outward (so flip signs of x y z)
        # - then swap x and y and negate z (so z is negated twice, meaning leave it as is)
        vnorm = [-vnorm[1], -vnorm[0], vnorm[2]]

        h_avg = np.average(points, 0)[2]

        pd_points = []
        for point in points:
            point = pv.PolyData(point)
            point.add_field_data(vnorm, "vnorm_tr")
            point.add_field_data([], "sol_val_list_day_in_month")
            # point.add_field_data([], "intersection_count_list")
            pd_points.append(point)

        # print(len(points),len(pd_points))

        for date in datelist:
            # print(date)
            sol_val = irradiance_on_triangle(vnorm, h_avg, date[0], lat)
            
            for pd_point in pd_points:
                sun_path = compute_sun_path(pd_point.points[0], proj, date[0])

                intersection_index_list = []
                for i, sun_point in enumerate(sun_path.points):
                    for neighbour in neighbours:
                        intersection_point, _ = neighbour.ray_trace(sun_point, pd_point.points[0], first_point=True)
                        if any(intersection_point):
                            intersection_index_list.append(i)
                            break
                if len(intersection_index_list) > 0:
                    sol_val = irradiance_on_triangle(vnorm, h_avg, date[0], lat, skip_timestamp_indices=intersection_index_list)
                
                sol_val_list = np.append(pd_point["sol_val_list_day_in_month"], sol_val)
                pd_point.add_field_data(sol_val_list, "sol_val_list_day_in_month")
                # pd_point.add_field_data(sol_val_list[0], "sol_val")

                # intersection_count_list = np.append(pd_point["intersection_count_list"], len(intersection_index_list))
                # pd_point.add_field_data(intersection_count_list, "intersection_count_list")
                # pd_point.add_field_data(intersection_count_list[0], "intersection count")

                print(sol_val)
                # print(pd_point["sol_val_list"])

        # SO pd_points contains all the point coordinates including the sol values.
        # print("Final list:", pd_points[0]["sol_val_list"])
        # print(pd_points)
        for pd_point in pd_points:
            # Write function to aggregate the daily values to monthly and yearly solar irradiation values.
            compute_yearly_solar_irradiance_point(pd_point, datelist)
            
            solar_roof_grid.append(pd_point)

    # TODO compute area of triangle and keep track of which points belong to it.
    # roof_area = roof_mesh.area      #The area of all triangles on a roof combined --> the whole building's roof probably, but I want seperate surfaces of the roof
    # print(roof_area)

    return (mesh, solar_roof_grid, [])    


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

def vtm_writer(results, path, write_mesh=False, write_grid=False, write_vector=False, write_intersections=False, write_neighbours=False):
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
        # block.save("vtm_objects/{}/solar_testing/mesh.vtm".format(path))
        # block.save("vtm_objects/{}/building_debug/NL.IMBAG.Pand.0503100000008153-0_mesh.vtm".format(path))
        # block.save("vtm_objects/{}/building_debug/NL.IMBAG.Pand.0503100000009010-0_mesh.vtm".format(path))
        # block.save("vtm_objects/{}/building_debug_2/NL.IMBAG.Pand.0503100000009010-0_mesh.vtm".format(path))
        block.save("vtm_objects/{}/building_debug_2/NL.IMBAG.Pand.0503100000008153-0_mesh.vtm".format(path))
    
    if write_grid:
        grids = pv.MultiBlock(grid_points)
        # grids = grid_points[3]
        # grids.save("vtm_objects/{}/grid_points.vtm".format(path))
        # grids.save("vtm_objects/{}/solar_testing/grid_points_2.vtm".format(path))
        # grids.save("vtm_objects/{}/solar_testing/grid_points_3_incorporate_intersections.vtm".format(path))
        # grids.save("vtm_objects/{}/solar_testing/grid_points_5_incorporate_intersections.vtm".format(path))
        # grids.save("vtm_objects/{}/solar/grid_points_solar_yearly.vtm".format(path))
        # grids.save("vtm_objects/{}/building_debug/NL.IMBAG.Pand.0503100000008153-0_grid.vtm".format(path))
        # grids.save("vtm_objects/{}/building_debug/NL.IMBAG.Pand.0503100000008153-0_grid_june.vtm".format(path))
        # grids.save("vtm_objects/{}/building_debug/NL.IMBAG.Pand.0503100000008153-0_grid_june_no_shadow.vtm".format(path))
        # grids.save("vtm_objects/{}/building_debug/NL.IMBAG.Pand.0503100000008153-0_grid_june_high_density.vtm".format(path))
        # grids.save("vtm_objects/{}/building_debug/NL.IMBAG.Pand.0503100000009010-0_grid_june_high_density.vtm".format(path))
        # grids.save("vtm_objects/{}/building_debug/NL.IMBAG.Pand.0503100000009010-0_grid_june_high_density_no_slivers.vtm".format(path))
        # grids.save("vtm_objects/{}/building_debug/NL.IMBAG.Pand.0503100000009010-0_grid_march_high_density.vtm".format(path))
        # grids.save("vtm_objects/{}/building_debug_2/NL.IMBAG.Pand.0503100000009010-0_grid_june.vtm".format(path))
        grids.save("vtm_objects/{}/building_debug_2/NL.IMBAG.Pand.0503100000008153-0_grid_january.vtm".format(path))

    if write_vector:
        vectors = []
        for point in grid:
            # print(point["vnorm_tr"])
            # print(point.points)
            vn = pv.Line(point.points[0], point.points[0] + point["vnorm_tr"])
            vectors.append(vn)
        vectors_block = pv.MultiBlock(vectors)
        # vectors_block.save("vtm_objects/{}/building_debug/NL.IMBAG.Pand.0503100000008153-0_vnorms.vtm".format(path))
        # vectors_block.save("vtm_objects/{}/building_debug/NL.IMBAG.Pand.0503100000008153-0_vnorms_high_density.vtm".format(path))
        # vectors_block.save("vtm_objects/{}/building_debug/NL.IMBAG.Pand.0503100000009010-0_vnorms_high_density.vtm".format(path))
        # vectors_block.save("vtm_objects/{}/building_debug/NL.IMBAG.Pand.0503100000009010-0_vnorms_high_density_no_slivers.vtm".format(path))
        # vectors_block.save("vtm_objects/{}/building_debug/NL.IMBAG.Pand.0503100000009010-0_vnorms_high_density_cell.vtm".format(path))
        # vectors_block.save("vtm_objects/{}/building_debug_2/NL.IMBAG.Pand.0503100000009010-0_vnorms.vtm".format(path))
        vectors_block.save("vtm_objects/{}/building_debug_2/NL.IMBAG.Pand.0503100000008153-0_vnorms.vtm".format(path))

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

            for id in list(buildings.keys()): # Taking a subset of this list is not correct as not all neighbours can be considered.
                geom = get_lod(buildings[id], lod)
                # print(id)

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

            # b_id = "NL.IMBAG.Pand.0503100000008153-0"
            # # b_id = "NL.IMBAG.Pand.0503100000009010-0"
            # geom = get_lod(buildings[b_id], lod)
            # # print(id)

            # building = buildings[b_id]
            
            # print(type(building))
            # print(building)
            # print(building.id)

            # # TODO: look into getting access to verts right away.

            # # Extract only roof surfaces from the building geometry.
            # roof_mesh = makePolyData_surfaces(get_semantic_surfaces(geom, 'roofsurface'))

            # # Find the neighbours of the current mesh according to a certain offset value.
            # neighbours = find_neighbours(b_id, roof_mesh, rtree, buildings, lod, offset)
            # # print(id, len(neighbours))
            
            # future = pool.submit(process_building, geom, roof_mesh, neighbours, proj, density, datelist)
            # future.add_done_callback(lambda p: progress.update())

            # futures.append(future)

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
        cm.triangulate()
        buildings = cm.get_cityobjects(type='BuildingPart')

    # Extract epsg from the input file and get the projection object.
    epsg = cm.get_epsg()
    proj = Proj(epsg)

    # id = "NL.IMBAG.Pand.0503100000005509-0"

    # Program settings:
    cores = mp.cpu_count()-2
    lod = "2.2"
    neighbour_offset = 150 #meters
    sampling_density = 1.5
    #temporal resolution? Hourly? 10min?

    # Specify list of dates here and give as parameter to process_multiple_buildings:
    month_list = []

    year = 2021
    c_day = 21      # characteristic day of the month
    months = 12     # the time resolution: monthly
    for month in range(months):
        date = dt.datetime(year, month+1, c_day)
        month_list.append(date)
    
    # Link the month with its corresponding number of days.
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    date_list = list(zip(month_list, days_in_month))

    # date_list_june = list([date_list[5]])
    # date_list_march = list([date_list[2]])
    date_list_january = list([date_list[0]])

    # Create rtree for further processing.
    rtree_idx = create_rtree(buildings, lod)

    # Process all buildings in the buildings dictionary.
    # results = process_multiple_buildings(buildings, rtree_idx, cores, proj, lod, neighbour_offset, sampling_density, date_list)
    # results = process_multiple_buildings(buildings, rtree_idx, cores, proj, lod, neighbour_offset, sampling_density, date_list_june)
    # results = process_multiple_buildings(buildings, rtree_idx, cores, proj, lod, neighbour_offset, sampling_density, date_list_march)
    results = process_multiple_buildings(buildings, rtree_idx, cores, proj, lod, neighbour_offset, sampling_density, date_list_january)


    print("Time to run the computations: {} seconds".format(time.time() - start_time))

    vtm_writer(results, path_string, write_mesh=False, write_grid=True, write_vector=False)
    
    print("Total time to run this script with writing to file(s): {} seconds".format(time.time() - start_time))

if __name__ == "__main__":
    freeze_support()
    main()