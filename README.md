# QARTA: An ML-based System for Accurate Map Services

This repository contains the code for experiments of QARTA components along with the code for the query calibration module.

To start, create a conda envrionment based on 'environment file' in this repo as follows:

`conda env create -f environment.yml`

Next, prepare your trips dataset to be as follows:

`TripStartTime,PickupLon,PickupLat,DropLon,DropLat,GT_Distance,GT_Duration,OsrmDistance,OsrmDuration`

The date should be fomrated as (month-day-year hour:minutes), python:('%m-%d-%y %H:%M'). A sample file for NYC trips is given in the data folder. 

The last two column `OsrmDistance` and `OsrmDuration` can be obtained by querying OSRM. A dockerized version is abvailable in thie repo.

Next, run `data_prep.py` to prepare the spatial zoning and obtain OSRM results. make sure to update the file names for your data, and the zones files. Also make sure you configure the dockerfile to download the map of the state/country that you are working on.


Once prepared, run the `query_calibration.py` on top of your trips. 