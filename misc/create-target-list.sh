#!/bin/bash

if [ $# -ne 2 ]
then
	ll="all"
else
	ll=$1
	ul=$2
fi

if [ $ll == "all" ]
then
	echo Creating target list for all images
	for img in `cat image-list.txt`
	do
		base=`echo $img|awk -F '.' '{print $1}'`
		cnt=`cat images/coordinates/$base/*.json|grep coordinates|wc -l`
		echo "$base, $img, $cnt"
		echo "$base, $img, $cnt" >> "target-list-all.txt"
	done
else
	rl=$((ll+1))
	ru=$((ul-1))
	echo "Creating target list for range ${rl} to ${ru}"

	for img in `cat image-list.txt`
	do
		base=`echo $img|awk -F '.' '{print $1}'`
		cnt=`cat images/coordinates/$base/*.json|grep coordinates|wc -l`
		if [ $cnt -gt $ll ] && [ $cnt -lt $ul ]
		then
			echo "$base, $img, $cnt"
			echo "$base, $img, $cnt" >> "target-list-${rl}-${ru}.txt"
		fi
	done
fi
