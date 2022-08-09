#!/bin/bash

dataset=$1

cpp_file=$dataset/nas_all.cpp
h_file=$dataset/nas_all.h

> $cpp_file
> $h_file

echo '#include "nas_all.h"' >> $cpp_file
echo "" >> $cpp_file
for folder in $dataset/models/*; do
    if [ -f $folder/pass.txt ]; then
        echo $folder
        f=$folder/model.tflite
        flash=$(du -hsk $f | awk '{print $1}')

        if [ "$flash" -lt "600" ]; then #Only include relatively small models
            #echo $f
            xxd -i $f >> $cpp_file
            echo "" >> $cpp_file
            sed -i 's/^unsigned char/alignas(8) const unsigned char/' $cpp_file
            sed -i 's/unsigned int/const int/' $cpp_file
        fi
    fi
done

values=($(grep -r tflite $cpp_file | awk 'NF{print $(NF-2)}'))

for ((i=0; i<"${#values[@]}"; i+=2)); do
    echo "extern const unsigned char ${values[$i]};" >> $h_file
    echo "extern const int ${values[$((i+1))]};" >> $h_file
    echo "" >> $h_file
done
