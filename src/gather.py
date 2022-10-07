#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 13:51:46 2022

@author: jason
"""

import cdsapi

c = cdsapi.Client()

c.retrieve(
    'satellite-ice-sheet-elevation-change',
    {
        'variable': 'all',
        'format': 'zip',
        'domain': 'greenland',
        'climate_data_record_type': 'icdr',
        'version': '3_0',
    },
    '/Users/jason/Dropbox/Surface_Elevation_Change/download.zip')