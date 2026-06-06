import matplotlib.pyplot as plt
import xarray as xr
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import os


# Path to the shapefile
shapefile_path = r"/home/hpcs_rnd/Yatendra_IITKgp/bias_correction/India__State_Boundary_2021_/India_Boundary.shp"

# Load the NetCDF file
ds = xr.open_dataset(r"/home/hpcs_rnd/Yatendra_IITKgp/bias_correction/Subodh_Data/GPCP_1_1.nc", decode_times=False)

# Load the shapefile using geopandas
india_shapefile = gpd.read_file(shapefile_path)

lons = ds['longitude'].values
lats = ds['latitude'].values

shape = india_shapefile.geometry.unary_union

mask = np.ones((32,32), dtype=np.float64)
for i, lon in enumerate(lons):
    for j, lat in enumerate(lats):
        point = Point(lon, lat)
        if not shape.contains(point):
            mask[j, i] = np.nan
            
            
np.save("mask32x32.npy",mask)