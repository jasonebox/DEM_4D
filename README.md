# DEM_4D
DEM_4D

Current code:

resample_25km-dhdt_output_time_vsarying_DEM.py

Aims to resample dhdt from 25 km grid to CARRA 2.5. However, we want a finer DEM. So, BedMachineGreenland-2017-09-20_surface_500m.tif can be found in ./raw

dhdt should be summed (like already in resample_25km-dhdt_output_time_vsarying_DEM.py) and the last value resampled added to the DEM.

The code should output a geotiff of the changed DEM.

Since the perimeter of the dhdt is coarse resolution, we may want to extrapolate dhdt outward.

Jason

#Notes
data documentation in ./raw
Perhaps better is the 100 m DEM Bolli Palmason from IMO has created, though it seems pretty smoothed.
