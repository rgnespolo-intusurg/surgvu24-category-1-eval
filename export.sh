#!/usr/bin/env bash

./build.sh

docker save surgtoolloc | gzip -c > SurgToolLoc.tar.gz
