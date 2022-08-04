"""
This module contains the main workflow for the computation of the solar radiation for buildings in 3D BAG.
It reads a CityJSON file, enriches its city objects with solar radiation values and writes back to CityJSON.

Complementary functions are found in utils.py
"""

import datetime as dt
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
from os import listdir
from os.path import isfile, join
from pyproj import Proj
from rtree import index
from tqdm import tqdm

import utils

def process_building(co_id, co, neighbours, proj, density, datelist):
    """
    Process a building to enrich it with a solar irradiation value.
    """
    new_geoms = []
    # sol_vals_building = []      # Store all sol vals of the points on all surfaces of the building to compute statistics.
    
    for geom in co.geometry:
        if float(geom.lod) >= 2.0:
            max_id = max(geom.surfaces.keys())
            old_ids = []

            # print(geom.get_surfaces()[0])
            for r_id, rsrf in geom.get_surfaces('roofsurface').items():

                # old_ids.append(r_id)
                # del geom.surfaces[r_id]     # deletes the semantics of the surfaces

                # Create an empty roofsurface semantic without attributes to add sliver triangles.
                geom.surfaces[r_id] = {
                    'type': rsrf['type'], 
                    'surface_idx': [[]]
                }

                if 'attributes' in rsrf.keys():
                    geom.surfaces[r_id]['attributes'] = rsrf['attributes']
                
                boundaries = geom.get_surface_boundaries(rsrf)
                # Potential TODO after P4: do the triangulation here instead of for the whole file at once. This keeps track of what triangles belong to what surfaces.


                # STELIOS' FIX BELOW
                # Create a PolyData mesh of the roof surface boundaries.
                mesh = utils.makePolyData_surfaces(boundaries)

                # Create a sampled grid of the roof surfaces of the mesh.
                grid = create_surface_grid(mesh, density)
                
                points_list = []
                index_list = []
                # Create one list of all the points in the grid of all roof surfaces.
                # Keep track of its index by filling a separate list of the same length.
                for i, points in enumerate(grid):
                    for p in points[0]:
                        points_list.append(list(p))
                        index_list.append(i)

                # Make the point list a PolyData object to be able to assign point data to the points.
                gp_mesh = pv.PolyData(points_list)

                # For every point on the mesh the index of the surface triangle that it belongs to is stored.
                gp_mesh.point_data["surface_index"] = index_list
                
                # Loop over each triangle of the mesh.
                for j in range(mesh.n_cells):
                    # TODO: integrate sliver check. Use utils.is_sliver()
                    #     if utils.is_sliver(boundary_geometry):
                    #         surface_index = rsrf['surface_idx'][j]
                    #         geom.surfaces[r_id]['surface_idx'].append(surface_index)
                    #         continue

                    # Compute the normal vector of the mesh cell (triangle)
                    vnorm = mesh.cell_normals[j]

                    # Convert normal to point outward and in NED frame (swap and negate x and y)
                    vnorm = [-vnorm[1], -vnorm[0], vnorm[2]]

                    # Get the latitude value for the triangle.
                    lat = utils.get_lat(proj, mesh.cell_centers().points[j])

                    # Only return the grid points that belong to the current surface triangle j.
                    grid_points = gp_mesh.extract_points(np.array(index_list) == j) # Returns pyVista unstructuredGrid, so a PolyData Object.
                    
                    # Compute the average z-value (height) of the grid points.
                    h_avg = np.average(grid_points.points, 0)[2]
                    if h_avg < 0:
                        h_avg = 0

                    # No primary TODO
                    # TODO: Make newly created grid_points list compatible with code below.
                    # TODO: incorporate Stelios' feedback.
                    pd_points_list = []
                    for point in grid_points.points:
                        point = pv.PolyData(point)
                        point.add_field_data(vnorm, "vnorm_tr")
                        point.add_field_data([], "sol_val_list_day_in_month")
                        pd_points_list.append(point)
                
                    # Loop to compute the solar potential value for the surface j for each month in the year.
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

                    for pd_point in pd_points_list:
                        # Write function to aggregate the daily values to monthly and yearly solar irradiation values.
                        utils.compute_yearly_solar_irradiance_point(pd_point, datelist)
                    
                    # Create a list with the solar potential values of each point in the grid.
                    sol_vals = [pd_point['sol_val_year'] for pd_point in pd_points_list]

                    # Compute statistics for the surface/triangle.
                    stats = {
                        'solar-number_of_samples': len(grid_points.points),
                        'solar-potential_avg': np.mean(sol_vals),
                        'solar-potential_min': np.min(sol_vals),
                        'solar-potential_max': np.max(sol_vals),
                        'solar-potential_std': np.std(sol_vals),
                        'solar-potential_p50': np.percentile(sol_vals, 50),
                        'solar-potential_p70': np.percentile(sol_vals, 70),
                        'solar-potential_unit': "Wh/m^2/year"
                    }

                    # CREATION OF NEW SEMANTIC SURFACES:
                    surface_index = rsrf['surface_idx'][j]

                    new_srf = {
                        'type': rsrf['type'],
                        'surface_idx': [surface_index]
                    }

                    # CREATION OF NEW ATTRIBUTES FOR THE SEMANTIC SURFACES
                    new_srf['attributes'] = stats

                    # Copy the existing attributes of the surface if applicable.
                    if 'attributes' in rsrf.keys():
                        for key, value in rsrf['attributes'].items():
                            new_srf['attributes'][key] = value

                    max_id += 1
                    geom.surfaces[max_id] = new_srf
            for srf in geom.surfaces.values():
                if srf['surface_idx'] is None:
                    srf['surface_idx'] = [[]]
            new_geoms.append(geom)
        else:
            new_geoms.append(geom)
    co.geometry = new_geoms

    return co_id, co


def process_multiple_buildings(cm, buildings_all_tiles, rtree, cores, proj, lod, offset, density, datelist):
    """
    Start the whole computation pipeline for multiple buildings within multiple processes.
    """
    with ProcessPoolExecutor(max_workers=cores) as pool:
        # The library tqdm is used to display a progress bar in the terminal.
        with tqdm(total=len(cm.cityobjects.items())) as progress:
            futures = []
            results = [] 

            # Loop over each city object in the city model.
            for co_id, co in cm.cityobjects.items():       # going over both 'Building' and 'BuildingPart'
                if cm.cityobjects[co_id].type == 'Building':
                    results.append([co_id, co])
                    progress.update()
                    continue
                
                # Get the geometry of the city object.
                geom = utils.get_lod(cm.cityobjects[co_id], lod)

                # Extract only roof surfaces from the building geometry.
                # TODO: Potential speed-up: might not extract roof_mesh as it is ONLY used as center point to find neighbours
                roof_mesh = utils.makePolyData_surfaces(utils.get_semantic_surfaces(geom, 'roofsurface'))

                # Find the neighbours of the current mesh according to a certain offset value.
                # Type polydata.
                neighbours = utils.find_neighbours(id, roof_mesh, rtree, buildings_all_tiles, lod, offset)
                
                # Submit each building to a multiprocessing pool to process the building further.
                future = pool.submit(process_building, co_id, co, neighbours, proj, density, datelist)
                future.add_done_callback(lambda p: progress.update())

                # future contains a co_id and co (cityobject).
                futures.append(future)

            for future in futures:
                results.append(future.result())
    
    return results

def write_cityjson(path, cm, results):
    new_cos = {}

    # Assign each newly created city object (co) to its co_id in the co ditionary.
    for res in results:
        co_id, co = res
        new_cos[co_id] = co
    
    # Create a deep copy of the city model for safety purposes.
    cm_copy = deepcopy(cm)
    cm_copy.cityobjects = new_cos

    # Save the new city model to a new file.
    path_out = path.split('.')[0] + '_solar.city.json'
    cityjson.save(cm_copy, path_out)

def main():
    """
    Main workflow.
    Test for multiple tiles (in a folder).
    """
    # Start_time to keep track of the running time.
    start_time = time.time()

    # Load the CityJSON files from a path referring to a folder which is an argument passed to the python script.
    path_folder = sys.argv[1]
    cj_files = [file for file in listdir(path_folder) if isfile(join(path_folder, file))]

    # TODO: don't load all tiles all at once, only needed tiles.
    # TODO: so change this to processing the tiles in the folder sequentially.
    # NOTE: when testing, it is done for only one tile now, so the for loop below is just performed once now.
    buildings_all_tiles = {}                # dict for all buildings in all tiles in the folder to make one rtree
    buildings_in_tile_list = []             # list to add the path of a tile and its buildings
    for file in cj_files:
        # Construct the path file to load and the output path file to write back to.
        path_file = path_folder + file
        path_file_out = path_folder + "output/" + file
        
        # Load the city model (cm) from the path to the file
        cm = cityjson.load(path_file)

        print("Citymodel is triangulated? ", cm.is_triangulated())

        # TODO also get Buildings in order to write back to file. Check whether this is already done.
        if cm.is_triangulated():
            # Get the buildings as BuildingPart (these contain LoD above 0) from the city model as dict (there are only buildings).
            buildings = cm.get_cityobjects(type='BuildingPart')
        else:
            cm.triangulate()
            buildings = cm.get_cityobjects(type='BuildingPart')

        # TODO: write code to find the 8 neighbouring tiles and extract the buildings within a buffer from the tile edges. Add these buildings to the dictionary.
        
        buildings_all_tiles.update(buildings)
        buildings_in_tile_list.append([path_file_out, cm, buildings])

    # Program settings:
    cores = mp.cpu_count()-2    # do not use all cores
    lod = "2.2"                 # highest LoD available
    neighbour_offset = 150      # in meters
    sampling_density = 3        # the lower the denser
                                # temporal resolution? Hourly? 10min?

    # Specify list of dates here and give as parameter to process_multiple_buildings:
    date_list = utils.create_date_list(2021)

    # Create rtree for quickly finding building objects.
    rtree_idx = utils.create_rtree(buildings_all_tiles, lod)
    
    for path, cm, buildings in buildings_in_tile_list:
        # Extract epsg from the input file and get the projection object.
        epsg = cm.get_epsg()
        proj = Proj(epsg)

        # Process all buildings in the buildings dictionary.
        results = process_multiple_buildings(cm, buildings_all_tiles, rtree_idx, cores, proj, lod, neighbour_offset, sampling_density, date_list)
        
        print("Time to run the computations: {} seconds".format(time.time() - start_time))

        # Write the enriched city model back to a cityJSON file.
        write_cityjson(path, cm, results)
        # utils.vtm_writer(results, path_string, write_mesh=False, write_grid=True, write_vector=False)     # Can be used for writing vtm/vtk. Function is not applicable.
        
    print("Total time to run this script with writing to file(s): {} seconds".format(time.time() - start_time))
    

if __name__ == "__main__":
    freeze_support()
    main()