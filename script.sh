#!/bin/bash

path_to_data='sample_data'

echo $path_to_data | python script.py

count=0
prev='nothing'

for i in $(cat $path_to_data'/tmp.txt'); do
    if ((count == 1))
        then
            original_file=$path_to_data'/inputs/'$prev'.JPG'
            output_file=$path_to_data'/outputs/'$prev'_'$i'_exif.json'
            exiftool -json $original_file > $output_file
    fi
    count=($count+1)%2
    prev=$i;
done
