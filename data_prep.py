# %%
import pandas as pd 
trips = pd.read_csv('data/nyc_1k.csv')


# %%
# zonning
# credits: from https://github.com/vipyoung/stad/blob/master/trip_location_to_zone.py

from rtree import index
import geopandas
from shapely.geometry import Point
from tqdm.auto import tqdm
tqdm.pandas(desc="progress bar!")

# load ShapeFile file for zones
zone_file = 'data/administrative_zones/nyct2010_wgs84.shp'
zones = geopandas.read_file(zone_file)


# Indexing the polygons
zone_idx = index.Index()
for i, shape in zones.geometry.iteritems():
	zone_idx.insert(i, shape.bounds)
	

def get_zone_optimized(lonlat):
	point = Point(lonlat)
	for j in zone_idx.intersection(lonlat):
		if(point.within(zones.geometry[j])):
			return j
	return -1


trips['SourceZone'] = trips[['PickupLon','PickupLat']].progress_apply(lambda x : get_zone_optimized(x.tolist()), axis=1)
trips['DestZone'] = trips[['DropLon','DropLat']].progress_apply(lambda x : get_zone_optimized(x.tolist()), axis = 1)


# %%
import requests
import json

# see the docker file and docker-compose to know more.

# uncomment this section if you have a running instance of OSRM

# API_endpoint_nearest = 'http://127.0.0.1:5000/route/v1/driving/'

# def request_function(query):
#     response =  requests.get(API_endpoint_nearest + query)
#     json_data = json.loads(response.text)
#     if 'routes' in json_data.keys():
#         return tuple([json_data['routes'][0]['distance'],
#                     json_data['routes'][0]['duration']])
#     else:
#         return None

# query_data =  trips.PickupLon.astype(str) + ',' + \
#                 trips.PickupLat.astype(str) = ';' + \
#                 trips.DropLon.astype(str) + ',' + \
#                 trips.DropLat.astype(str)

# distances, durations = zip(*query_data.progress_apply(request_function).values)
# trips['OsrmDistance'] = distances
# trips['OsrmDuration'] = durations




# %%
trips.to_csv('data/nyc_1k_zoned.csv', index=False)
# %%
