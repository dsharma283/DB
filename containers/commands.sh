#!/bin/bash

install_base="/data/dbnet"
deform_install_path="$install_base/lib/python"
pretrained_path="$install_base/pretrained"
build_base="/app/assets/ops/dcn"
bbone_path="$install_base/backbone"
app_pretrained="/app/pretrained"
visualize_op="$install_base/viz_output"
input_imgs="$install_base/images"

if [ ! -d $deform_install_path ]
then
	mkdir -p $deform_install_path
fi

export PYTHONPATH=$deform_install_path
cd $build_base
python3 setup.py install --home=$install_base
cd -

if [ ! -d $pretrained_path ]
then
	mv -f pretrained $install_base
fi

if [ ! -d $bbone_path ]
then
	mkdir -p $bbone_path
fi

if [ ! -d $visualize_op ]
then
	mkdir -p $visualize_op
fi

if [ ! -d $input_imgs ]
then
	mkdir -p $input_imgs
fi
