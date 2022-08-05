import json
import math
import numpy as np
import pandas as pd
import pyvista as pv
import rasterio
import sys

from cjio import cityjson
from rasterio.profiles import DefaultGTiffProfile
from scipy.interpolate import griddata

# This script rasterises meshes (buildings) in 3D BAG.
# Take the extent of 1 tile and use a resolution of 0,5m x 0,5m per pixel.

def makePolyData_surfaces(input_surfaces):
    """
    Convert surfaces (triangles) to a Pyvista's PolyData object/mesh.
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

def get_lod(bdg, lod):
    """
    Gets the geometry with a specific lod.
    Returns the geometry.
    """
    for g in bdg.geometry:
        if g.lod == lod or g.lod == float(lod):
            return g

def createRasterGrid(bbox, rRes):
    # bbox is stored as: [min_x, min_y, min_z, max_x, max_y, max_z]
    min_x = bbox[0]
    max_x = bbox[3]
    min_y = bbox[1]
    max_y = bbox[4]

    xRange = np.arange(min_x, max_x+math.ceil(rRes), rRes)
    yRange = np.arange(min_y, max_y+math.ceil(rRes), rRes)

    grid = np.array(np.meshgrid(xRange, yRange))

    return grid

def writeRasterToFile(filename, grid, crs, transform):
    raster = rasterio.open(filename,
                                'w', 
                                **DefaultGTiffProfile(
                                    driver='GTiff',
                                    height=grid.shape[0],
                                    width=grid.shape[1],
                                    count=1,
                                    nodata=-9999,
                                    dtype=grid.dtype,
                                    crs=crs,
                                    transform=transform))                             
    
    raster.write(grid, 1)
    raster.close()

def main_elevation():
    # Steps to rasterise 3D BAG
    # - Create a raster covering the extent of the 3D BAG tile --> Done
    # - Load the 3D BAG meshes (triangulated) from CityJSON into PyVista's PolyData data structure --> Done, hopefully
    # - For each raster pixel, cast a downward vector (from max z-value)
    # - Check whether this vector intersects with a buliding mesh
    # - If it does store its resulting elevation value as this pixel's value

    # Load the CityJSON file from a path which is an argument passed to the python script.
    path = sys.argv[1]
    cm = cityjson.load(path)

    bbox = cm.get_bbox()
    rRes = 0.5      # raster resolution

    grid = createRasterGrid(bbox, rRes)
    print(grid.dtype)
    gridX, gridY = grid
    print(len(gridX), len(gridX[0]))

    # Extract the buildings from the city model.
    buildings = cm.get_cityobjects(type='BuildingPart')
    lod = "2.2"

    # print(cm.cityobjects.keys()["vertices"])
    # verts = cm["vertices"]

    mesh_list = []
    for bdg_id in list(buildings.keys()):
        # print(bdg_id)
        geom = get_lod(buildings[bdg_id], lod)
        mesh = makePolyData_surfaces(geom.get_surfaces()[0])
        mesh_list.append(mesh)

    meshes = mesh_list[0].merge(mesh_list[1:len(mesh_list)-1])
    # print(meshes)

    # TODO loop over each raster pixel and ray trace the meshes
    # link: https://hatarilabs.com/ih-en/how-to-create-a-geospatial-raster-from-xy-data-with-python-pandas-and-rasterio-tutorial
    elevation_grid = np.full((len(gridX), len(gridX[0])), float(-9999))
    z = 500
    for i in range(len(gridX)):
        for j in range(len(gridX[i])):
            x = gridX[i][j]
            y = gridY[i][j]
            grid_point = (x, y, z)
            end_point = (x, y, z-600) 
            # print(grid_point)
            # print(end_point)
            intersections, cell = meshes.ray_trace(grid_point, end_point, first_point=True)
            if len(intersections) > 0:
                elevation_grid[i][j] = intersections[2]
                # print(intersections)

    # Extract epsg from the input file and create transform.
    epsg = cm.get_epsg()
    rasterCrs = rasterio.crs.CRS.from_epsg(epsg)
    transform = rasterio.Affine.translation(gridX[0][0]-rRes/2, gridY[0][0]-rRes/2)*rasterio.Affine.scale(rRes,rRes)

    # Write grid to a raster '.tif' file.
    writeRasterToFile("raster/3DBAG_elevation_raster_3_nodata.tif", elevation_grid, rasterCrs, transform)

def main_solar():
    # Steps to rasterise 3D BAG
    # - Create a raster covering the extent of the 3D BAG tile --> Done
    # - Load the 3D BAG meshes (triangulated) from CityJSON into PyVista's PolyData data structure --> Done, hopefully
    # - For each raster pixel, cast a downward vector (from max z-value)
    # - Check whether this vector intersects with a buliding mesh
    # - If it does, store its solar value as this pixel's value

    # Load the CityJSON file from a path which is an argument passed to the python script.
    path = sys.argv[1]
    cm = cityjson.load(path)

    bbox = cm.get_bbox()
    rRes = 0.5      # raster resolution

    grid = createRasterGrid(bbox, rRes)
    # print(grid.dtype)
    gridX, gridY = grid
    # print(len(gridX), len(gridX[0]))

    # Extract the buildings from the city model.
    buildings = cm.get_cityobjects(type='BuildingPart')
    lod = "2.2"

    # print(cm.cityobjects.keys()["vertices"])
    # verts = cm["vertices"]

    # Workflow:
    # Attach per mesh the solar value of each cell as cell_data to the polydata object
    # Extract only the surfaces with attributes and make polydata of this.
    mesh_list = []
    for bdg_id in list(buildings.keys()):
        # print(bdg_id)
        geom = get_lod(buildings[bdg_id], lod)
        sol_val_list = []
        for r_id, rsrf in geom.get_surfaces('roofsurface').items():
            # print(rsrf['surface_idx'])
            if 'attributes' in rsrf.keys():
                # print(geom.surfaces[r_id]) # has index to surface
                boundaries = geom.get_surface_boundaries(rsrf)

                roof_mesh = makePolyData_surfaces(boundaries)

                sol_val_avg = (geom.surfaces[r_id]['attributes']['solar-potential_avg'])
                # sol_val_list.append(sol_val_avg)
                roof_mesh.cell_data["solar-potential_avg"] = sol_val_avg 
                # print(roof_mesh, roof_mesh["solar-potential_avg"])

                mesh_list.append(roof_mesh)
                # print(sol_val_avg)
                # TODO: attach this value to the polydata thing.
            # else:
                # print(geom.surfaces[r_id])
        print("sol val: ", len(sol_val_list))
        print("items: ", len(geom.get_surfaces('roofsurface').items()))

        # Make polydata of the surfaces created/extracted above.
        # mesh = makePolyData_surfaces(geom.get_surfaces()[0])
        # print(geom.get_surfaces()[0])

        # mesh_list.append(mesh)

    meshes = mesh_list[0].merge(mesh_list[1:len(mesh_list)-1])
    # print(meshes)

    # TODO loop over each raster pixel and ray trace the meshes
    # link: https://hatarilabs.com/ih-en/how-to-create-a-geospatial-raster-from-xy-data-with-python-pandas-and-rasterio-tutorial
    elevation_grid = np.full((len(gridX), len(gridX[0])), float(-9999))
    z = 500
    for i in range(len(gridX)):
        for j in range(len(gridX[i])):
            x = gridX[i][j]
            y = gridY[i][j]
            grid_point = (x, y, z)
            end_point = (x, y, z-600) 
            # print(grid_point)
            # print(end_point)
            intersections, cell = meshes.ray_trace(grid_point, end_point, first_point=True)
            if len(intersections) > 0:
                # elevation_grid[i][j] = intersections[2]
                elevation_grid[i][j] = meshes["solar-potential_avg"][cell[0]]
                # print(meshes["solar-potential_avg"][cell[0]])
                # print(cell[0])
                # TODO: to get solar value access the field_data of the mesh for the cell index.
                #       when loading the cityjson file the solar value attribute should be copied over as field data attribute for the mesh.
                #       get the attribute out of roof surfaces in the geom.

    # Extract epsg from the input file and create transform.
    epsg = cm.get_epsg()
    rasterCrs = rasterio.crs.CRS.from_epsg(epsg)
    transform = rasterio.Affine.translation(gridX[0][0]-rRes/2, gridY[0][0]-rRes/2)*rasterio.Affine.scale(rRes,rRes)

    # Write grid to a raster '.tif' file.
    # writeRasterToFile("raster/3DBAG_elevation_raster_3_nodata.tif", elevation_grid, rasterCrs, transform)
    writeRasterToFile("raster/solar/3DBAG_solar_raster.tif", elevation_grid, rasterCrs, transform)

# def dummyMain():
#     # Create points
#     p1 = [1, 2]
#     p2 = [1, 3]
#     p3 = [2, 2]
#     p4 = [2, 4]
#     p5 = [8, 3.5]
#     p6 = [2.5, 4.5]
#     p7 = [1.5, 1]

#     points = [p1, p2, p3, p4, p5, p6, p7]
#     elevations = [10, 13, 24, 12, 11, 18, 31]

#     # raster resolution of 0.5m
#     rRes = 0.5

#     x = list(zip(*points))[0]
#     y = list(zip(*points))[1]

#     xRange = np.arange(min(x), max(x)+math.ceil(rRes), rRes)
#     yRange = np.arange(min(y), max(y)+math.ceil(rRes), rRes)
#     gridX, gridY = np.meshgrid(xRange, yRange)

#     gridElevation = griddata(points, elevations, (gridX, gridY), method='linear')

#     transform = rasterio.Affine.translation(gridX[0][0]-rRes/2, gridY[0][0]-rRes/2)*rasterio.Affine.scale(rRes,rRes)
#     rasterCrs = rasterio.crs.CRS.from_epsg(28992)

#     filename = 'raster/testInterpRaster2.tif'

#     print(type(gridElevation))
#     print(gridElevation)
#     print(gridElevation.shape)

#     writeRasterToFile(filename, gridElevation, rasterCrs, transform)

if __name__ == "__main__":
    # dummyMain()
    # main_elevation()
    main_solar()