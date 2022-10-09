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
from scipy.ndimage import gaussian_filter

if os.getlogin() == "jason":
    base_path = "/Users/jason/Dropbox/DEM_4D/"
    sys.path.append("/Users/jason/Dropbox/geomatcher")
    CARRA_path = "/Users/jason/Dropbox/CARRA/CARRA_rain/"

if os.getlogin() == "adrien":
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

#%%
import matplotlib.pyplot as plt

plt.imshow(dhdt_sum, cmap="bwr")
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

# %% prepare data and apply geomatcher

# reproject to meter space
dhdt_grid = np.dstack([lon_dhdt, lat_dhdt, dhdt_sum])
dhdt_grid_m = gm.convert_grid_coordinates(dhdt_grid, "4326", "3413")
dhdt_grid_m[:, :, 1] = dhdt_grid_m[:, :, 1][::-1]

bedmachine_grid_m = np.dstack([bedmachine_xs, bedmachine_ys, bedmachine])

# match
indexes = gm.match_m2m_old(bedmachine_grid_m, dhdt_grid_m, only_indexes=True)

dhdt_on_bedmachine = dhdt_grid_m[:, :, 2].flatten()[indexes]

#%%
import matplotlib.pyplot as plt

result = np.flipud(dhdt_on_bedmachine)

result_smoothed = gaussian_filter(result, 20)

bedmachine_grid_m[:, :, 2][
    bedmachine_grid_m[:, :, 2] == np.nanmin(bedmachine_grid_m[:, :, 2])
] = np.nan

dz = result + bedmachine_grid_m[:, :, 2]

dz_filled = dz.copy()

dz_filled[~np.isfinite(dz_filled)] = bedmachine_grid_m[:, :, 2][~np.isfinite(dz_filled)]

plt.figure()
plt.imshow(
    dz,
)

dz_smoothed = gaussian_filter(dz, 10, mode="reflect")

# plt.figure()
# ax1 = plt.subplot(121)
# ax1.imshow(dz)
# ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)
# ax1.imshow(dz_smoothed)

#%%

profile = rasterio.open(
    f"{base_path}/raw/BedMachineGreenland-2017-09-20_surface_500m.tiff"
).profile

wo = 1

if wo:
    with rasterio.open(
        f"{base_path}/output/dz_on_bedmachine.tif", "w", **profile
    ) as dst:
        dst.write(dz, 1)

    with rasterio.open(
        f"{base_path}/output/dz_10sig_on_bedmachine.tif", "w", **profile
    ) as dst:
        dst.write(dz, 1)


print("done")
