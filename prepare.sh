#!/bin/sh
PYTHON="./osm-network-generator-venv/bin/python"
$PYTHON collect_cities.py --worldcities-file worldcities.csv --output-dir $INSTANCE_DIR/cities --country NL --max-workers 8
$PYTHON subsample.py -i $INSTANCE_DIR/cities/ -o $INSTANCE_DIR/base --sample-size 1000 --dist-min 120 --dist-max 180 --dist-step 60 --max-workers 8
