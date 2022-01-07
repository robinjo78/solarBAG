from cjio import cityjson

from solarBAGprototype import makePolyData_all_surfaces, makePolyData_semantic_surface

def get_lod(bdg, lod):
    for g in bdg.geometry:
        if g.lod == lod:
            return g

if __name__ == "__main__":
    # Load the CityJSON file from a path.
    path = "/mnt/c/Users/hurkm/repos/solarBAG/data/3dbag_v210908_fd2cee53_3007_new_v11_triangulated.city.json" # a linux path

    cm = cityjson.load(path)

    buildings = cm.get_cityobjects(type='BuildingPart')
    # print(buildings)

    id = "NL.IMBAG.Pand.0503100000000018-0"
    bdg = buildings[id]

    print(bdg)

    geom = get_lod(bdg, "2.2")

    print(geom.lod)
    print(geom)

    surfaces = geom.get_surfaces()[0]
    mesh = makePolyData_all_surfaces(surfaces)

    roofs = geom.get_surfaces(type='roofsurface')
    roof_mesh = makePolyData_semantic_surface(roofs, geom)

    roof_mesh.save("NL.IMBAG.Pand.0503100000000018-0-roof.vtk")
    mesh.save("NL.IMBAG.Pand.0503100000000018-0.vtk")

# geom_2 = bdg.geometry[2] # LoD 2.2

# triangle_1 = geom_2.boundaries[0][0][0]
# vertex_1 = triangle_1[0]
# v = vertex_1

# print("Vertex 1 of building 1:", v)

# s = cm.transform["scale"]
# t = cm.transform["translate"]

# print("Scale factor:", s)
# print("Translate factor:", t)

# # Performing this computation proves that the boundaries directly extracted from the citymodel are already transformed.
# # Therefore, manual transformation is not needed. This computation computes the compressed version.
# compressed_v = [(v[0] - t[0]) / s[0], (v[1] - t[1]) / s[1], (v[2] - t[2]) / s[2]]
# print("Compressed vertex:", compressed_v)