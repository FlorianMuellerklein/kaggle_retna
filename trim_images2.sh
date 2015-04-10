#!/bin/bash

IMAGES="/Volumes/Mildred/Kaggle/Retna/Data/Backup/train/*.jpeg"
for file in $IMAGES
do
	echo "$file"
	convert $file -fuzz 10% -trim +repage -resize 512x512 -gravity center -background black -extent 512x512 $file
	
done

