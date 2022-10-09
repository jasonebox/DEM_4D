#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 15:39:19 2022

@author: jason
"""


import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
from glob import glob
from netCDF4 import Dataset

# from mpl_toolkits.basemap import Basemap
import pandas as pd
from datetime import datetime
import netCDF4

# CARRA grid info
# Lambert_Conformal()
#     grid_mapping_name: lambert_conformal_conic
#     standard_parallel: 72.0
#     longitude_of_central_meridian: -36.0
#     latitude_of_projection_origin: 72.0
#     earth_radius: 6367470.0
#     false_easting: 1334211.3405653758
#     false_northing: 1584010.8994621644
#     longitudeOfFirstGridPointInDegrees: 302.903
#     latitudeOfFirstGridPointInDegrees: 55.81

AW = 0

if not AW:
    path = "/Users/jason/Dropbox/Surface_Elevation_Change/"
    os.chdir(path)

# --------------------------------------------

ly = "x"

# global plot settings
th = 1
font_size = 16
plt.rcParams["axes.facecolor"] = "k"
plt.rcParams["axes.edgecolor"] = "k"
plt.rcParams["font.size"] = font_size

# read data
fn = "./C3S_GrIS_RA_SEC_25km_vers3_2022-08-20.nc"
nc2 = Dataset(fn, "r")
print(nc2.variables)
time = nc2.variables["time"]

from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

# time_convert = netCDF4.num2date(time[:], time.units, time.calendar)

n_months = np.shape(nc2.variables["dhdt"])[2]

#   is more precise (e.g. days = int(9465.0))

start = date(1990, 1, 1)  # This is the "days since" part

# print(type(offset))         # >>>  <class 'datetime.date'>

nj = 123
ni = 65
#%%

dhdt_sum = np.zeros((nj, ni))

mask = np.flipud(np.array(nc2.variables["land_mask"][:, :]))
# for i in range(n_months):
for i in range(n_months):

    delta = relativedelta(
        months=+i
    )  # Create a time delta object from the number of days

    offset = start + delta  # Add the specified number of days to 1990

    datestring = offset.strftime("%Y-%m-%d")

    print(datestring)  # >>>  2015-12-01

    dhdt = np.flipud(np.array(nc2.variables["dhdt"][:, :, i]))
    dhdt[mask == 0] = np.nan
    dhdt_sum += dhdt

DPIs = [200]

plt.close()
plt.imshow(dhdt_sum, vmin=-350, vmax=150)
plt.title(datestring)
plt.axis("off")
plt.colorbar()

ly = "x"
if ly == "p":
    # os.system('mkdir -p '+'./Figs/daily/max in sep to oct range/')
    # figpath='./Figs/daily/max in sep to oct range/'
    figpath = "/Users/jason/0_dat/Surface_Elevation_Change/Figs/"
    os.system("mkdir -p " + figpath)

    for DPI in DPIs:
        plt.savefig(figpath + datestring + ".png", bbox_inches="tight", dpi=DPI)

#%%

import numpy as np
import pandas as pd
import os
from pyproj import Proj, transform
import xarray as xr
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from datetime import datetime


# %% CARRA coordinates


def lon360_to_lon180(lon360):

    # reduce the angle
    lon180 = lon360 % 360

    # force it to be the positive remainder, so that 0 <= angle < 360
    lon180 = (lon180 + 360) % 360

    # force into the minimum absolute value residue class, so that -180 < angle <= 180
    lon180[lon180 > 180] -= 360

    return lon180


# CARRA West grid dims
ni = 1269
nj = 1069

# read lat lon arrays
fn = "/Users/jason/Dropbox/CARRA/CARRA_ERA5_events/meta/CARRA/2.5km_CARRA_west_lat_1269x1069.npy"
lat = np.fromfile(fn, dtype=np.float32)
lat_mat = lat.reshape(ni, nj)[::-1]
clat_mat = lat.reshape(ni, nj)

fn = "/Users/jason/Dropbox/CARRA/CARRA_ERA5_events/meta/CARRA/2.5km_CARRA_west_lon_1269x1069.npy"
lon = np.fromfile(fn, dtype=np.float32)
lon_mat = lon.reshape(ni, nj)[::-1]

lon_pn = lon360_to_lon180(lon)
clon_mat = lon_pn.reshape(ni, nj)

fn = "/Users/jason/Dropbox/CARRA/CARRA_ERA5_events/meta/CARRA/2.5km_CARRA_west_elev_1269x1069.npy"
elev = np.fromfile(fn, dtype=np.float32)
elev = elev.reshape(ni, nj)


# fn='./meta/CARRA/CARRA_W_domain_ice_mask.nc'
# ds=xr.open_dataset(fn)
# mask = np.array(ds.z)

# %% reproject 4326 (lat/lon) CARRA coordinates to 3413 (orth polar projection in meters)

# from lat/lon to meters
inProj = Proj(init="epsg:4326")
outProj = Proj(init="epsg:3413")

x1, y1 = clon_mat.flatten(), clat_mat.flatten()
cx, cy = transform(inProj, outProj, x1, y1)
cx_mat = cx.reshape(ni, nj)
cy_mat = cy.reshape(ni, nj)

cols, rows = np.meshgrid(
    np.arange(np.shape(clat_mat)[1]), np.arange(np.shape(clat_mat)[0])
)

CARRA_positions = pd.DataFrame(
    {
        "rowc": rows.ravel(),
        "colc": cols.ravel(),
        "xc": cx_mat.ravel(),
        "yc": cy_mat.ravel(),
    }
)

# ,
# 'maskc': mask.flatten()}
# import CARRA datset
# ds = xr.open_dataset(raw_path+'tp_2012.nc')
# CARRA_data = np.array(ds.tp[0, :, :]).flatten()

#%%
# import sys

# CARRA_grid = np.dstack((lon_mat, lat_mat,
#                         elev))


# sys.path.append('/Users/jason/Dropbox/geomatcher/')
# import geomatcher.geomatcher as gm
# CARRA_grid_ll = gm.convert_grid_coordinates(CARRA_grid)

# %% load dhdt info

fn = "./C3S_GrIS_RA_SEC_25km_vers3_2022-08-20.nc"
ds = xr.open_dataset(fn)

# from lat/lon to meters
inProj = Proj(init="epsg:4326")
outProj = Proj(init="epsg:3413")

niE = np.shape(ds.lat.values)[1]
njE = np.shape(ds.lon.values)[0]
print(np.shape(ds.lon))

# niE = 1269 ; njE = 1069

lat_mesh, lon_mesh = np.meshgrid(ds.lat.values, ds.lon.values)
x1, y1 = lon360_to_lon180(lon_mesh.flatten()), lat_mesh.flatten()
ex, ey = transform(inProj, outProj, x1, y1)
ex_mat = ex.reshape(niE, njE)
ey_mat = ey.reshape(niE, njE)

cols2, rows2 = np.meshgrid(
    np.arange(np.shape(ds.lat.values)[0]), np.arange(np.shape(ds.lon.values)[1])
)
lat_e, lon_e = np.meshgrid(ds.lat.values, ds.lon.values)
ERA5_positions = pd.DataFrame(
    {
        "row_e": rows2.ravel(),
        "col_e": cols2.ravel(),
        "lon_e": lon_e.ravel(),
        "lat_e": lat_e.ravel(),
    }
)

#%%

steps = ["-21", "-00", "-03", "-06"]
steps = ["00", "01", "02", "03"]
times = ds.variables["time"]
dates = []
# date_strings=[]
# str_dates = [i.strftime("%Y-%m-%dT%H:%M") for i in time]
for i, time in enumerate(times):
    for step in range(4):
        # print(str(np.array(time)))
        timex = pd.to_datetime(times[i].to_numpy())
        timex = pd.date_range(timex) + pd.Timedelta(hours=step * 3)
        timex = timex[0].strftime("%Y-%m-%d-%H")
        print(timex)

        # dates.append(str(np.array(time)))
        # temp=str(np.array(time))[0:13]

        # date_strings.append(temp)
        # dates.append(datetime.strptime(temp, "%Y-%m-%dT%H"))
        # print(date_time_obj)

        # print(x.strftime("%Y-%m-%dT%H:%M"))
        # print(date_strings)

        # dtime=pd.to_datetime(date_strings,format="%Y-%m-%dT%H")

        choice = "tp"
        choice = "mtpr"

        if choice == "t2m":
            ERA_data = np.array(ds.t2m[i, :, :]) - 273.15
        if choice == "tp":
            ERA_data = np.array(ds.tp[i, step, :, :]) * 1000
        if choice == "mtpr":
            ERA_data = np.array(ds.mtpr[i, step, :, :]) * 3 * 3600
        # print(np.shape(ERA_data))

        #  nearest neighbours

        # dataset to be upscaled -> ERA5
        nA = np.column_stack((ex_mat.ravel(), ey_mat.ravel()))
        # dataset to provide the desired grid -> CARRA
        nB = np.column_stack((cx_mat.ravel(), cy_mat.ravel()))

        btree = cKDTree(nA)  # train dataset
        dist, idx = btree.query(nB, k=1)  # apply on grid

        # collocate matching cells
        CARRA_cells_for_ERA5 = ERA5_positions.iloc[idx]

        # Output resampling key ERA5 data in CARRA grid
        path_tools = "/tmp/"
        CARRA_cells_for_ERA5.to_pickle(path_tools + "resampling_key_ERA5_to_CARRA.pkl")

        outpath = "/Users/jason/0_dat/ERA5/events/resampled/" + choice + "/"
        os.system("mkdir -p " + outpath)

        #  visualisation

        new_grid = ERA_data[CARRA_cells_for_ERA5.col_e, CARRA_cells_for_ERA5.row_e]
        new_grid = np.rot90(new_grid.reshape(ni, nj).T)
        plt.close()
        plt.imshow(new_grid, vmin=0, vmax=6)
        plt.axis("off")
        plt.title(timex)
        plt.colorbar()
        DPI = 200
        ly = "p"

        if ly == "p":
            figpath = "/Users/jason/0_dat/ERA5/events/Figs/"
            # figpath='/Users/jason/Dropbox/CARRA/CARRA_ERA5_events/Figs/ERA5/'
            # os.system('mkdir -p '+figpath)
            plt.savefig(
                figpath + str(timex) + ".png",
                bbox_inches="tight",
                pad_inches=0.04,
                dpi=DPI,
                facecolor="w",
                edgecolor="k",
            )
            # plt.savefig(figpath+select_period+'JJA_'+hgt+'z_anom.eps', bbox_inches='tight')

        new_grid.astype(dtype=np.float16).tofile(
            outpath + "/" + str(timex) + "_1269x1069.npy",
        )
