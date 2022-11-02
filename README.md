# solarBAG
MSc thesis Geomatics on computing solar potential for large CityJSON datasets.

## Usage

First run the following:
```
virtualenv venv
. venv/bin/activate
pip install -r requirements.txt
```
This will create a new virtual environment, activate it and install all required packages to run the script.

Download one or multiple CityJSON file(s) from [3D BAG](https://3dbag.nl/en/viewer).
Then upgrade the CityJSON file(s) to CityJSON version 1.1:
```
cjio [file.json] upgrade save [new_file.city.json]
```

Triangulate the CityJSON tile(s):
```
cjio [file.json] triangulate save [new_file.city.json]
```

Store all the triangulated files in a folder and run the script by using WSL or Linux terminal:
```
python3 solarBAG_CityJSON.py [folder_with_cityjson_files]
```
The script now only works for already triangulated CityJSON files. 

The output of the program is an enriched CityJSON file. It can be visualised in [Ninja](https://ninja.cityjson.org/#).
<!---
Visualise the output *.vtm* or *.vtk* file in ParaView. (Note: to get output, the write flags in function vtm_writer() need to be True)


![alt text](https://github.com/robinjo78/solarBAG/blob/main/images/Screenshot_mesh_solar_grid.png?raw=true)
![alt text](https://github.com/robinjo78/solarBAG/blob/main/images/Screenshot_mesh_grid_intersections.png?raw=true)
-->
