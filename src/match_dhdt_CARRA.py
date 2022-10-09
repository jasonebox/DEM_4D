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



# %% read and preprocess dhdt

fn = f"{base_path}/raw/C3S_GrIS_RA_SEC_25km_vers3_2022-08-20.nc"
ds = xr.open_dataset(fn)

niE = np.shape(ds.lat.values)[1]
njE = np.shape(ds.lon.values)[0]
print(np.shape(ds.lon))

lat_dhdt = ds.lat.values
lon_dhdt = ds.lon.values

n_months = np.shape(ds.variables["dhdt"])[2]

mask = np.flipud(np.array(ds.variables["land_mask"]))

start = datetime.date(1990, 1, 1)  # This is the "days since" part

nj = 123
ni = 65
dhdt_sum = np.zeros((nj, ni))

for i in range(n_months):

    delta = dateutil.relativedelta.relativedelta(
        months=+i
    )  # Create a time delta object from the number of days

    offset = start + delta  # Add the specified number of days to 1990

    datestring = offset.strftime("%Y-%m-%d")

    print(datestring)  # >>>  2015-12-01

    dhdt = np.flipud(np.array(ds.variables["dhdt"][:, :, i]))
    dhdt[mask == 0] = np.nan
    dhdt_sum += dhdt

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

# %% prepare data and apply geomatcher

# reproject to meter space
CARRA_grid = np.dstack([lon_CARRA, lat_CARRA, elev_CARRA])
dhdt_grid = np.dstack([lon_dhdt, lat_dhdt, dhdt_sum])
CARRA_grid_m = gm.convert_grid_coordinates(CARRA_grid, "4326", "3413")
dhdt_grid_m = gm.convert_grid_coordinates(dhdt_grid, "4326", "3413")

# match
indexes = gm.match_m2m_old(CARRA_grid_m, dhdt_grid_m, only_indexes=True)

dhdt_on_CARRA = dhdt_grid_m[:, :, 2].flatten()[indexes]

#%%
import matplotlib.pyplot as plt

plt.imshow(dhdt_on_CARRA)