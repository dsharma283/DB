#!/bin/bash

in_dir=$1
out_dir=$2

for dir in `ls $in_dir`
do
	idir="$in_dir/$dir/images"
	odir="$out_dir/$dir"
	#echo "CUDA_VISIBLE_DEVICES=0 python dbnet.py -i $idir -r $odir -j -v"
	echo "Starting $dir"
	CUDA_VISIBLE_DEVICES=0 python dbnet.py -i $idir -r $odir -j -p
	sleep 10
done
