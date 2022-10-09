#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Adrien Wehrlé, University of Zurich, Switzerland

"""

import xarray as xr
import dateutil.relativedelta
import numpy as np
import datetime
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
# print(np.shape(ds.lon))

lat_dhdt = ds.lat.values
lon_dhdt = ds.lon.values

n_months = np.shape(ds.variables["dhdt"])[2]

mask = np.flipud(np.array(ds.variables["land_mask"]))

start = datetime.date(1990, 1, 1)  # This is the "days since" part

nj = 123
ni = 65
dhdt_sum = np.zeros((nj, ni))

datestrings=[]
for i in range(n_months):

    delta = dateutil.relativedelta.relativedelta(
        months=+i
    )  # Create a time delta object from the number of days

    offset = start + delta  # Add the specified number of days to 1990

    datestring = offset.strftime("%Y-%m-%d")
    datestrings.append(datestring)
    # print(datestring)  # >>>  2015-12-01

    dhdt = np.flipud(np.array(ds.variables["dhdt"][:, :, i]))
    dhdt[mask == 0] = np.nan
    dhdt_sum += dhdt
    
print(np.nanmin(dhdt_sum))

#%%
import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap, LinearSegmentedColormap
# from mpl_toolkits.axes_grid1 import make_axes_locatable

font_size=12

fig, ax = plt.subplots(figsize=(9,11))

im=ax.imshow(dhdt_sum,cmap='bwr',vmin=-350,vmax=150)
ax.set_title(datestrings[0]+' to '+datestring)
ax.axis('off')

clb = plt.colorbar(im,shrink=0.7, pad=0.04)
clb.ax.set_title('m',fontsize=14,c='k')

clb.ax.tick_params(labelsize=font_size) 
clb.ax.set_title('m',fontsize=font_size)
ly='p'
if ly =='p':
    # os.system('mkdir -p '+'./Figs/daily/max in sep to oct range/')
    # figpath='./Figs/daily/max in sep to oct range/'
    figpath='./Figs/'
    os.system('mkdir -p '+figpath)

    DPIs=[200]

    for DPI in DPIs:
        plt.savefig(figpath+datestring+'.png', bbox_inches='tight', dpi=DPI)

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
#%%

font_size=12

fig, ax = plt.subplots(figsize=(9,11))

# im=ax.imshow(dhdt_sum,cmap='bwr',vmin=-350,vmax=150)
im=ax.imshow(elev_CARRA,cmap='magam')
ax.set_title(datestrings[0]+' to '+datestring)
ax.axis('off')

clb = plt.colorbar(im,shrink=0.7, pad=0.04)
clb.ax.set_title('m',fontsize=14,c='k')

clb.ax.tick_params(labelsize=font_size) 
clb.ax.set_title('m',fontsize=font_size)
ly='x'
if ly =='p':
    # os.system('mkdir -p '+'./Figs/daily/max in sep to oct range/')
    # figpath='./Figs/daily/max in sep to oct range/'
    figpath='./Figs/'
    os.system('mkdir -p '+figpath)

    DPIs=[200]

    for DPI in DPIs:
        plt.savefig(figpath+datestring+'.png', bbox_inches='tight', dpi=DPI)
#%%
print(np.shape(indexes))

# dhdt_on_CARRA = dhdt_grid_m[:, :, 2][indexes[:, :, 0], indexes[:, :, 1]]
dhdt_on_CARRA = dhdt_grid_m[:, :, 2].flatten()[indexes]
#%%
plt.imshow(dhdt_on_CARRA)
