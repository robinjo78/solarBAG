from cjio import cityjson

# Load the CityJSON file from a path.
path = "/mnt/c/Users/hurkm/repos/solarBAG/data/3dbag_v21031_7425c21b_3007_v11.city.json" # a linux path

cm = cityjson.load(path)

buildings = cm.get_cityobjects(type='building')

id = "254"
bdg = buildings[id]

geom_2 = bdg.geometry[2] # LoD 2.2

triangle_1 = geom_2.boundaries[0][0][0]
vertex_1 = triangle_1[0]
v = vertex_1

print("Vertex 1 of building 1:", v)

s = cm.transform["scale"]
t = cm.transform["translate"]

print("Scale factor:", s)
print("Translate factor:", t)

compressed_v = [(v[0] - t[0]) / s[0], (v[1] - t[1]) / s[1], (v[2] - t[2]) / s[2]]
print("Compressed vertex:", compressed_v)

# verts = [[v[0] * s[0] + t[0], v[1] * s[1] + t[1], v[2] * s[2] + t[2]]
#                 for v in cm["vertices"]]