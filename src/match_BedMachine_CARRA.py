#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Adrien Wehrlé, University of Zurich, Switzerland

and Jason Box

"""

import xarray as xr
import dateutil.relativedelta
import numpy as np
import datetime
from osgeo import gdal
import os
import rasterio
import sys

if os.getlogin() == 'jason':
    base_path='/Users/jason/Dropbox/DEM_4D/'
    sys.path.append("/Users/jason/Dropbox/geomatcher")
    CARRA_path = "/Users/jason/Dropbox/CARRA/CARRA_rain/"

if os.getlogin() == 'adrien':
    base_path = "/home/adrien/EO-IO/DEM_4D"
    sys.path.append("/home/adrien/EO-IO/geomatcher")
    CARRA_path = "/home/adrien/EO-IO/CARRA_rain"


os.chdir(base_path)

import geomatcher.geomatcher as gm


def lon360_to_lon180(lon360):

    # reduce the angle
    lon180 = lon360 % 360

    # force it to be the positive remainder, so that 0 <= angle < 360
    lon180 = (lon180 + 360) % 360

    # force into the minimum absolute value residue class, so that -180 < angle <= 180
    lon180[lon180 > 180] -= 360

    return lon180


# %% read and preprocess BedMachine

with rasterio.open(
    f"{base_path}/raw/BedMachineGreenland-2017-09-20_surface_500m.tiff"
) as src:
    bedmachine = src.read(1)
    height = bedmachine.shape[0]
    width = bedmachine.shape[1]
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(src.transform, rows, cols)
    bedmachine_xs = np.array(xs)
    bedmachine_ys = np.array(ys)

# %% read and preprocess CARRA

# CARRA West grid dims
ni = 1269
nj = 1069

lat = np.fromfile(
    f"{CARRA_path}/ancil/2.5km_CARRA_west_lat_1269x1069.npy", dtype=np.float32
)
lat_mat = lat.reshape(ni, nj)[::-1]
lat_CARRA = lat.reshape(ni, nj)

lon = np.fromfile(
    f"{CARRA_path}/ancil/2.5km_CARRA_west_lon_1269x1069.npy", dtype=np.float32
)
lon_mat = lon.reshape(ni, nj)[::-1]

lon_pn = lon360_to_lon180(lon)
lon_CARRA = lon_pn.reshape(ni, nj)

elev = np.fromfile(
    f"{CARRA_path}/ancil/2.5km_CARRA_west_elev_1269x1069.npy", dtype=np.float32
)
elev_CARRA = elev.reshape(ni, nj)

rf = np.fromfile(
    f"{base_path}/raw/rf_2012_1269x1069_float32.npy", dtype=np.float32
)
rf = rf.reshape(ni, nj)

# %% prepare data and apply geomatcher

# reproject to meter space
CARRA_grid = np.dstack([lon_CARRA, lat_CARRA, elev_CARRA])
CARRA_grid_m = gm.convert_grid_coordinates(CARRA_grid, "4326", "3413")

bedmachine_grid_m = np.dstack([bedmachine_xs, bedmachine_ys, bedmachine])

# match
indexes = gm.match_m2m_old(CARRA_grid_m, bedmachine_grid_m, only_indexes=True)

bedmachine_on_CARRA = bedmachine_grid_m[:, :, 2].flatten()[indexes]

#%%
import matplotlib.pyplot as plt

result=np.flipud(bedmachine_on_CARRA)
plt.imshow(result)

#%%

profile = rasterio.open(f"{base_path}/raw/BedMachineGreenland-2017-09-20_surface_500m.tiff").profile

wo=1

if wo:
    with rasterio.open(f"{base_path}./output/bedmachine_on_CARRA.tif", 'w', **profile) as dst:
        dst.write(result, 1)

    with rasterio.open(f"{base_path}./output/CARRA_elev.tif", 'w', **profile) as dst:
        dst.write(np.flipud(elev_CARRA), 1)

    with rasterio.open(f"{base_path}./output/CARRA_rf_2012.tif", 'w', **profile) as dst:
        dst.write(np.flipud(rf), 1)

print("done")