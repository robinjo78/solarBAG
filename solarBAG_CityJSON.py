"""
This module contains the main workflow for the computation of the solar radiation for buildings in 3D BAG.
It reads a CityJSON file, enriches its city objects with solar radiation values and writes back to CityJSON.

Complementary functions are found in utils.py
"""

import datetime as dt
import math
import multiprocessing as mp
import numpy as np
import pyvista as pv
import solarpy as sp
import sys
import time

from cjio import cityjson
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
from helpers.shape_index import create_surface_grid
from multiprocessing import freeze_support
from pyproj import Proj
from rtree import index
from tqdm import tqdm

import utils

def process_building(co_id, co, neighbours, proj, density, datelist):
    """
    Process a building to enrich it with a solar irradiation value.
    """
    new_geoms = []

    for geom in co.geometry:
        if float(geom.lod) >= 2.0:
            max_id = max(geom.surfaces.keys())

            # for srf in geom.surfaces.values():
            #     if srf['surface_idx'] is None:
            #         srf['surface_idx'] = [[]]
            # if geom.surfaces[3]['surface_idx'] is None:
            #     max_id -= 1
            # print(max_id)
            old_ids = []

            # print(geom.get_surfaces()[0])

            for r_id, rsrf in geom.get_surfaces('roofsurface').items():
                # print(r_id, rsrf)

                old_ids.append(r_id)
                # del geom.surfaces[r_id]     # deletes the semantics of the surfaces

                # Create an empty roofsurface semantic without attributes to add sliver triangles.
                geom.surfaces[r_id] = {
                    'type': rsrf['type'], 
                    'surface_idx': [[]]
                }     # THIS SEMANTIC SURFACE IS ONLY WRITTEN TO FILE WHEN THERE ARE SLIVERS, WHY???

                if 'attributes' in rsrf.keys():
                    geom.surfaces[r_id]['attributes'] = rsrf['attributes']
                
                boundaries = geom.get_surface_boundaries(rsrf)
                # Potential TODO after P4: do the triangulation here instead of for the whole file at once. This keeps track of what triangles belong to what surfaces.

                # Per surface (boundary_geometry) in boundaries (in this case triangle) I want:
                # - normal vector -- Done
                # - sampled grid points -- Done
                # - to check whether it is sliver, if true skip and just return geom -- Half done
                # - polydata object of it -- Done
                for j, boundary_geometry in enumerate(boundaries):
                    # boundary_geometry is a triangle (and should be a triangle)
                    # In case the triangle is a sliver, it should be ignored in further processing, but stored as is in the geom.
                    if utils.is_sliver(boundary_geometry):
                        surface_index = rsrf['surface_idx'][j]
                        geom.surfaces[r_id]['surface_idx'].append(surface_index)
                        
                        # print("Sliver:", surface_index)
                        continue
                    
                    pd_triangle = utils.makePolyData_surfaces([boundary_geometry])      # make it a polydata object
                    pd_triangle = pd_triangle.compute_normals()                         # compute the normal
                    # print(pd_triangle)

                    vnorm = pd_triangle['Normals'][0]      # The normal faces inward and is in ENU frame

                    # Convert normal to point outward and in NED frame (swap and negate x and y)
                    vnorm = [-vnorm[1], -vnorm[0], vnorm[2]]

                    # Get the latitude value for the triangle.
                    lat = utils.get_lat(proj, pd_triangle.center_of_mass())
                    # print("Latitude:", lat)

                    # Sample a grid of points on the triangle.
                    grid_points = create_surface_grid(pd_triangle, density)[0][0]     # The index it returns can be removed.
                    # print("Grid points:", grid_points)

                    h_avg = np.average(grid_points, 0)[2]
                    # print("Average height:", h_avg)

                    pd_points_list = []
                    for point in grid_points:
                        point = pv.PolyData(point)
                        point.add_field_data(vnorm, "vnorm_tr")
                        point.add_field_data([], "sol_val_list_day_in_month")
                        pd_points_list.append(point)

                    for date in datelist:
                        sol_val = utils.irradiance_on_triangle(vnorm, h_avg, date[0], lat)

                        for pd_point in pd_points_list:
                            sun_path = utils.compute_sun_path(pd_point.points[0], proj, date[0])

                            intersection_index_list = []
                            for i, sun_point in enumerate(sun_path.points):
                                for neighbour in neighbours:
                                    intersection_point, _ = neighbour.ray_trace(sun_point, pd_point.points[0], first_point=True)
                                    if any(intersection_point):
                                        intersection_index_list.append(i)
                                        break
                            if len(intersection_index_list) > 0:
                                sol_val = utils.irradiance_on_triangle(vnorm, h_avg, date[0], lat, skip_timestamp_indices=intersection_index_list)
                        
                            sol_val_list = np.append(pd_point["sol_val_list_day_in_month"], sol_val)
                            pd_point.add_field_data(sol_val_list, "sol_val_list_day_in_month")

                            # print(sol_val)

                    for pd_point in pd_points_list:
                        # Write function to aggregate the daily values to monthly and yearly solar irradiation values.
                        utils.compute_yearly_solar_irradiance_point(pd_point, datelist)

                        # GET THE SOL VALUES HERE.
                    
                    sol_val_avg = np.mean([pd_point['sol_val_year'] for pd_point in pd_points_list])
                    # print("Average sol val year:", sol_val_avg)
                    # TODO add more types of sol values.


                    # CREATION OF NEW SEMANTIC SURFACES:
                    surface_index = rsrf['surface_idx'][j]
                    # print(surface_index)

                    new_srf = {
                        'type': rsrf['type'],
                        'surface_idx': [surface_index]
                    }

                    # CREATION OF NEW ATTRIBUTES FOR THE SEMANTIC SURFACES
                    new_srf['attributes'] = {}
                    new_srf['attributes']['solar-potential'] = sol_val_avg
                    new_srf['attributes']['solar-potential_unit'] = "Wh/m^2/year"

                    # Copy the existing attributes of the surface if applicable.
                    if 'attributes' in rsrf.keys():
                        for key, value in rsrf['attributes'].items():
                            new_srf['attributes'][key] = value

                    # print(new_srf['attributes'])
                    
                    max_id += 1
                    # print(max_id)
                    geom.surfaces[max_id] = new_srf
            for srf in geom.surfaces.values():
                if srf['surface_idx'] is None:
                    srf['surface_idx'] = [[]]
            new_geoms.append(geom)
        else:
            new_geoms.append(geom)
    # print(geom.surfaces)
    co.geometry = new_geoms

    return co_id, co


def process_multiple_buildings(cm_copy, buildings, rtree, cores, proj, lod, offset, density, datelist):
    """
    Start the whole computation pipeline for multiple buildings within multiple processes.
    """
    with ProcessPoolExecutor(max_workers=cores) as pool:
        # The library tqdm is used to display a progress bar in the terminal.
        with tqdm(total=len(buildings)) as progress:
            futures = []
            results = [] 

            # First test this for a smaller CityJSON file.
            # TODO: create this file (cjio.subset) --> have it for one building now ('write_test_new_semantics')
            for co_id, co in cm_copy.cityobjects.items():       # going over both 'Building' and 'BuildingPart'
                # print(co_id, co)

                if cm_copy.cityobjects[co_id].type == 'Building':
                    results.append([co_id, co])
                    continue
                
                geom = utils.get_lod(cm_copy.cityobjects[co_id], lod)
                # print(cm_copy.cityobjects[co_id])
                # print(geom)

                # Extract only roof surfaces from the building geometry.
                roof_mesh = utils.makePolyData_surfaces(utils.get_semantic_surfaces(geom, 'roofsurface'))

                # Find the neighbours of the current mesh according to a certain offset value.
                # Type polydata.
                neighbours = utils.find_neighbours(id, roof_mesh, rtree, buildings, lod, offset)
                
                future = pool.submit(process_building, co_id, co, neighbours, proj, density, datelist)
                future.add_done_callback(lambda p: progress.update())

                # future contains a co (cityobject).
                futures.append(future)

            for future in futures:
                results.append(future.result())
    
    return results
                
def write_cityjson(cm_copy, results):
    new_cos = {}
    
    for res in results:
        co_id, co = res
        new_cos[co_id] = co
            
    cm_copy.cityobjects = new_cos

    path_out = 'data/write/solarBAG_write_smaller_tile_faster_2.json'
    cityjson.save(cm_copy, path_out)


def main():
    """
    Main workflow.
    """
    # Start_time to keep track of the running time.
    start_time = time.time()

    # Load the CityJSON file from a path which is an argument passed to the python script.
    path = sys.argv[1]
    cm = cityjson.load(path)

    print("Citymodel is triangulated? ", cm.is_triangulated())

    # TODO also get Buildings in order to write back to file.
    if cm.is_triangulated():
        # Get the buildings from the city model as dict (there are only buildings).
        buildings = cm.get_cityobjects(type='BuildingPart')
    else:
        cm.triangulate()
        buildings = cm.get_cityobjects(type='BuildingPart')

    # Extract epsg from the input file and get the projection object.
    epsg = cm.get_epsg()
    proj = Proj(epsg)

    # Program settings:
    cores = mp.cpu_count()-2    # do not use all cores
    lod = "2.2"                 # highest LoD available
    neighbour_offset = 150      # in meters
    sampling_density = 1.5      # the lower the denser
                                # temporal resolution? Hourly? 10min?

    # Specify list of dates here and give as parameter to process_multiple_buildings:
    date_list = utils.create_date_list(2021)
    # date_list_june = list([date_list[5]])

    # Create rtree for quickly finding building objects.
    rtree_idx = utils.create_rtree(buildings, lod)

    # Create a copy of the citymodel
    cm_copy = deepcopy(cm)

    # Process all buildings in the buildings dictionary.
    # results = process_multiple_buildings(cm_copy, buildings, rtree_idx, cores, proj, lod, neighbour_offset, sampling_density, date_list)
    # results = process_multiple_buildings(buildings, rtree_idx, cores, proj, lod, neighbour_offset, sampling_density, date_list_june)
    # process_multiple_buildings(cm_copy, buildings, rtree_idx, cores, proj, lod, neighbour_offset, sampling_density, date_list)

    # TODO: run this to check if it works
    results = process_multiple_buildings(cm_copy, buildings, rtree_idx, cores, proj, lod, neighbour_offset, sampling_density, date_list)

    print("Time to run the computations: {} seconds".format(time.time() - start_time))

    # Adjust the path string for convenient and organised file writing purposes.
    split_list = path.split("/")
    path_string = split_list[len(split_list)-1]
    
    write_cityjson(cm_copy, results)
    # utils.vtm_writer(results, path_string, write_mesh=False, write_grid=True, write_vector=False)
    
    print("Total time to run this script with writing to file(s): {} seconds".format(time.time() - start_time))

if __name__ == "__main__":
    freeze_support()
    main()