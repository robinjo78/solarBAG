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

Run the script by using WSL or Linux terminal:
```
python3 solarBAGprototype.py [your_file.json]
```
The script now only works for already triangulated CityJSON files. 

Visualise the output *.vtm* or *.vtk* file in ParaView. (Note: to get output, the write flags in function vtm_writer() need to be True)

![alt text](https://github.com/robinjo78/solarBAG/blob/main/images/Screenshot_mesh_solar_grid.png?raw=true)
![alt text](https://github.com/robinjo78/solarBAG/blob/main/images/Screenshot_mesh_grid_intersections.png?raw=true)
