#!/bin/bash
install_base=$1
io_base=$2

store_path="$install_base/backbone"
bbone_file="$store_path/resnet50-19c8e357.pth"
cache_path="/root/.cache/torch/hub/checkpoints"
cache_file="$cache_path/resnet50-19c8e357.pth"

if [ -f $bbone_file ]
then
	mkdir -p $cache_path
	cp -f $bbone_file $cache_path
fi

python3 dbnet.py -i $io_base -vu

if [ ! -d $store_path ]
then
	mkdir -p $store_path
	cp -f $cache_file $store_path
fi
