#!/bin/bash
if [ ! -d "/app/inputs/lib/python" ]
then
	mkdir -p /app/inputs/lib/python
fi

export PYTHONPATH=/app/inputs/lib/python
cd assets/ops/dcn
python3 setup.py install --home=/app/inputs
cd -
if [ ! -d "/app/inputs/pretrained" ]
then
	mv -f pretrained inputs
fi

if [ ! -d "/app/inputs/backbone" ]
then
	mkdir -p /app/inputs/backbone
fi
