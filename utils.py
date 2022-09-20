""" 
Module with utility and helper functions for geometrical computations, conversions etc.
These functions are not considered as the core workflow.
"""

import datetime as dt
import json
import math
import numpy as np
import pyvista as pv
import solarpy as sp

from cjio import cityjson
from rtree import index


def find_bbox_all_tiles(cj_files, path_folder):
    """
    Searches for the metadata tag in all tiles in the folder.
    Then, it extracts the geographical extent from the metadata.
    It returns a list containing the 2D bboxes of the tile.
    """
    tile_bboxes = []
    for file in cj_files:
        path_file = path_folder + file
        f = open(path_file, "r")
        s = f.read()
        #-- find "metadata"
        posm = s.find("metadata")
        pos_start = s.find("{", posm)
        pos_end = 0
        cur = pos_start
        count = 1
        while True:
            a = s.find("{", cur+1) 
            b = s.find("}", cur+1) 
            if a < b: 
                count += 1
                cur = a
            else: 
                count -= 1
                cur = b
            if count == 0:
                pos_end = b
                break
        m = s[pos_start:pos_end+1]
        jm = json.loads(m)

        c1 = jm['geographicalExtent'][0:2]
        c2 = jm['geographicalExtent'][3:5]
        bbox_2D = c1 + c2

        tile_bboxes.append((path_file, bbox_2D))

    return tile_bboxes

def create_rtree_tile_bbox(tile_bboxes):
    # Set properties for the rtree index.
    p = index.Property()
    p.dimension = 2     # 2D is sufficient.

    # Create empty rtree for the bounding boxes of a tile
    rtree_idx_tile = index.Index(properties=p)

    # Loop to dump all tile's bboxes in the rtree.
    for i, tile in enumerate(tile_bboxes, 0):
        path = tile[0]
        bbox = tile[1]
        # Insert the bbox into the rtree with the tile's file path as object (obj).
        rtree_idx_tile.insert(i, bbox, obj=path)

    print("Size of the Rtree for tile bboxes: {}".format(rtree_idx_tile.get_size()))
    
    return rtree_idx_tile

# This function gradually builds the dictionary with all potential neighbours for a tile.
# This includes buildings from neighbouring tiles.
def build_nb_dict(nb_buildings, cm_nb, buffer_box):
    # Create an r-tree of cm_nb'
    buildings = cm_nb.get_cityobjects(type='BuildingPart')

    lod = "2.2"     # In solarBAG this is already initialised.
    rtree_idx_nb = create_rtree(buildings, lod)
    # print("Nb rtree:", rtree_idx_nb)

    # Adjust 2D buffer box to 3D to be compatible with 3D rtree.
    buffer_box_3d = [buffer_box[0], buffer_box[1], -100, buffer_box[2], buffer_box[3], 100]

    # Find the buildings in each neighbouring tile that is situated within the buffer box limits.
    hits = list(rtree_idx_nb.intersection(buffer_box_3d, objects="True"))
    print("Number of buildings within buffer:", len(hits))

    # Put all the buildings in hits in the nb_buildings dict
    for item in hits:
        bdg_id = item.object
        bdg_obj = buildings[bdg_id]

        # Add the building from a neighbouring tile to the nb_buildings dict
        # Use its id as key and the actual building object as value.
        nb_buildings[bdg_id] = bdg_obj
    
    return nb_buildings

def create_date_list(year, c_day=21, months=12):
    """
    Creates a date list with tuples where each tuple has:
    - a datetime object with a year, month and day
    - the number of days in the corresponding month

    Leaps years are not take into account.
    Parameters:
    - year
    - c_day: the characteristic day of the month, default is 21
    - months: the number of months in the year, default 12

    Returns the date list.
    """
    month_list = []

    for month in range(months):
        date = dt.datetime(year, month+1, c_day)
        month_list.append(date)
    
    # Link the month with its corresponding number of days.
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    date_list = list(zip(month_list, days_in_month))

    return date_list

def get_lod(bdg, lod):
    """
    Gets the geometry with a specific lod.
    Returns the geometry.
    """
    for g in bdg.geometry:
        if g.lod == lod or g.lod == float(lod):
            return g

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

def create_rtree(buildings, lod):
    """
    Creates and returns an RTree with buildings.
    """
    # Set properties for the rtree index.
    p = index.Property()
    p.dimension = 3

    # Create empty rtree for citymodel
    rtree_idx = index.Index(properties=p)

    # Loop to dump all buildings in the rtree.
    for i, bdg in enumerate(buildings.values(), 0):
        # Take the LoD 2.2 geometry of the building.
        geom = get_lod(bdg, lod)

        # Extract surfaces from the building geometry.
        surfaces = geom.get_surfaces()[0]

        # Compute the bounding box of a building from the surfaces.
        bbox = compute_bbox(surfaces) # Might use cityjson.get_bbox function here (see script Stelios)

        # Insert the bbox into the rtree with the building's id as object (obj).
        rtree_idx.insert(i, bbox, obj=bdg.id)

    print("Size of the Rtree: {}".format(rtree_idx.get_size()))

    return rtree_idx

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
    # - Might take the original way it is stored in CityJSON. Probably not possible. Is another way of indexing.
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

def skip_on_azimuth(roof_center, neighbour, north_range=None):
    """
    Filters potential neighbours on its azimuth.
    Neighbours in a certain northern range depending on the time of the year, can be ignored.

    Returns boolean.
    """
    neighbour_center = neighbour.center_of_mass()

    # TODO: Make the north range variable per latitude.
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
        # Then make a polydata mesh out of it and add it to the list of meshes: the neighbours.
        # Then the neighbours can be ray traced.
        neighbour_building = buildings[item.object]

        neighbour_geom = get_lod(neighbour_building, lod)
        neighbour_surfaces = neighbour_geom.get_surfaces()[0]
        neighbour_mesh = makePolyData_surfaces(neighbour_surfaces)

        # Also, add the building itself as a 'neighbour'.
        if id == item.object:
            neighbour_list.append(neighbour_mesh)
            continue

        # Skip neighbouring buildings if their z_max is lower than the z_min of the current roof. These neighbours cannot block the current building.
        if roof_mesh.bounds[4] > neighbour_mesh.bounds[5]:
            continue
        
        # TODO: add date list as parameter to skip_on_azimuth.
        if skip_on_azimuth(roof_mesh_center, neighbour_mesh):
            continue

        neighbour_list.append(neighbour_mesh)

    return neighbour_list

def is_sliver(surface):
    """
    Checks whether a surface is a sliver.
    Returns boolean.
    """
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
        return False
    else:
        return True

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
        if not(is_sliver(surface)):
            surfaces_to_keep.append(surface)

    return surfaces_to_keep

def get_lat(proj, point):
    """
    Returns the latitude of the point.
    """
    latlon = proj(point[0], point[1], inverse=True)
    lat = latlon[1]

    return lat

def compute_daily_irradiance(G):
    """
    Function to compute the area of the graph to get Wh/m^2/day.
    """
    res = sum(G)

    # TODO: make this functions possible for all time intervals.
    # Divide by four because the intervals are 15 minutes and the computed values are W/h, so we need W/15min and add these 4 up to get W/h.
    # res = res/4

    return res

def irradiance_on_triangle(vnorm, h, date, lat, skip_timestamp_indices=[]):
    """
    Computes the solar irradiance value for a triangle.
    """

    # irradiance on triangle function understanding test:
    # extraterrestrial_rad = sp.gon(date)
    # prel = sp.pressure(h)/sp.pressure(0)
    # theta_zenith = sp.theta_z(date, lat)
    # m = sp.air_mass_kastenyoung1989(np.rad2deg(theta_zenith), h)
    # alpha_int = 0.32

    # G = extraterrestrial_rad * np.exp(-prel * m * alpha_int)


    # print("Extraterrestrial radiation: ", extraterrestrial_rad)
    # print("Pressure relation: ", prel)
    # print("Pressure relation (neg): ", -prel)
    # print("Air mass: ", m)
    # print("Exp(-prel, m, 0.32): ", np.exp(-prel * m * alpha_int))
    # print("G:", G)



    # t = [date + dt.timedelta(minutes=i) for i in range(0, 15 * 24 * 4, 15)]
    t = [date + dt.timedelta(minutes=i) for i in range(0, 60 * 24, 60)]
    G = [sp.irradiance_on_plane(vnorm, h, i, lat) for i in t]
    # If skip_timestamp_indices is not empty, intersections occur for which solar radiation need not be computed.
    if len(skip_timestamp_indices) > 0:
        for ind in skip_timestamp_indices:
            G[ind] = 0

    res = compute_daily_irradiance(G)
    return res

def compute_sun_path(point, proj, date):
    """
    Compute the sun path of grid point on the triangle.
    TODO for potential speed-up: compute sun path once and use that for all points? Should take a sun point far enough. Do this for benchmarking.
    """
    lat = get_lat(proj, point)

    t = [date + dt.timedelta(minutes=i) for i in range(0, 60 * 24, 60)]

    # Compute for each point the corresponding position of the sun at a factor x away for a certain time interval.
    vsol_ned = [point + sp.solar_vector_ned(date, lat) * -500 for date in t]
    sun_path = pv.PolyData(vsol_ned)

    return sun_path

def compute_yearly_solar_irradiance_point(point, datelist):
    """
    Computes yearly solar irradiance for a point.
    """
    sol_val_list_whole_month = []
    for sol_val, date in zip(point["sol_val_list_day_in_month"], datelist):
        sol_val_month = sol_val * date[1]
        sol_val_list_whole_month.append(sol_val_month)
    
    sol_val_year = sum(sol_val_list_whole_month)

    point.add_field_data(sol_val_list_whole_month, "sol_val_list_whole_month")
    point.add_field_data(sol_val_year, "sol_val_year")

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