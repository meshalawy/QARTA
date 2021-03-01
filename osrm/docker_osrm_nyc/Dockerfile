FROM osrm/osrm-backend as pre_processing
WORKDIR /data
RUN apt-get update && \
    apt-get -y --no-install-recommends install wget 

RUN wget http://download.geofabrik.de/north-america/us/new-york-latest.osm.pbf


FROM osrm/osrm-backend as running_instance
COPY --from=pre_processing /data/new-york-latest.osm.pbf /data/new-york-latest.osm.pbf

RUN osrm-extract -p /opt/car.lua /data/new-york-latest.osm.pbf
RUN  osrm-partition /data/new-york-latest.osrm
RUN  osrm-customize /data/new-york-latest.osrm


ENTRYPOINT osrm-routed --algorithm mld /data/new-york-latest.osrm