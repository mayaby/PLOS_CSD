import numpy as np
import xarray as xr


# GLOBAL VARIABLES
# Region
lat_max = 61
lat_min = 46
lon_max = 360-20
lon_min = 360-55
lon_max_W = -20
lon_min_W = -55

# the three latitude/longitude keys
lat_keys = ['lat', 'latitude', 'nav_lat']
lon_keys = ['lon', 'longitude', 'nav_lon']


## FUNCTIONS TO SPECIFY/EDIT THE GLOBAL VARIABLES
# set parameters
def set_boundaries(
    longitude_maximum = 360-20,
    longitude_minimum = 360-55,
    latitude_maximum = 61,
    latitude_minimum = 46):

    global lon_max, lon_min, lat_max, lat_min, lon_max_W, lon_min_W
    lon_max = longitude_maximum
    lon_min = longitude_minimum 
    lat_max = latitude_maximum
    lat_min = latitude_minimum
    lon_max_W = lon_max-360
    lon_min_W = lon_min-360

def set_lon_lat_keys(
    longitudes=['lon', 'longitude', 'nav_lon'],
    latitudes =['lat', 'latitude', 'nav_lat']):

    global lon_keys, lat_keys
    lon_keys = longitudes
    lat_keys = latitudes

def get_values():
    print("lon_min: "+str(lon_min))
    print("lon_max: "+str(lon_max))
    print("lat_min: "+str(lat_min))
    print("lat_max: "+str(lat_max))
    print("lon_keys: "+str(lon_keys))
    print("lat_keys: "+str(lat_keys))

# function to extract the assti from a dataset, given the longitude and latitude keys
def assti(data, str_lon, str_lat, print_info=False):
    mean_dims = [d for d in data.dims if d != "time"]
    if print_info:
        print("   The mean is taken over the dimensions ", mean_dims)
    if np.sum(data[str_lon].values<0) > 0:
        lon_max_tmp = lon_max_W
        lon_min_tmp = lon_min_W
    else:
        lon_max_tmp = lon_max
        lon_min_tmp = lon_min
    spg = (
        data
        .where(data[str_lon] >= lon_min_tmp)
        .where(data[str_lon] <= lon_max_tmp)
        .where(data[str_lat] >= lat_min)
        .where(data[str_lat] <= lat_max)
    )
    spg = spg.mean(dim = mean_dims)
    gl = data.mean(dim = mean_dims)
    assti = spg - gl
    return assti

# function to get the longitude / latitude keys from the model
def get_lon_lat_keys(data, location_keys):
    str_loc = [key for key in data.coords.keys() if key in location_keys]
    if len(str_loc) != 1:
        print('CAUTION: There exist the keys: ' + str(str_loc) + '. The first one is used!')
    return str_loc[0]


if __name__ == '__main__':
    # Replace 'tos_CMIP6_file.nc' with the path to your CMIP6 ocean surface temperature file.
    # The pre-computed ASSTI used in the paper is already stored in CMIP6_amoc2.nc.
    data = xr.open_dataset("tos_CMIP6_file.nc").tos
    str_lon = get_lon_lat_keys(data, lon_keys)
    str_lat = get_lon_lat_keys(data, lat_keys)

    assti_index = assti(data, str_lon, str_lat)