#!/bin/bash

#set -x

models=($(grep -r "unsigned char" dummy_all.h | awk 'NF{ print $NF }' | sed 's/\[\]\;//g'))

for model in ${models[@]}; do
    cur=`grep -r "GetModel" dnn.cpp | sed "s/.*(\(.*\))/\1/" | sed "s/;//g"`
    sed -i "s/$cur/$model/g" dnn.cpp
    
    echo $(grep -r "GetModel" dnn.cpp)
    
    cd build
    make -j4
    sudo picotool load -f dnn.uf2
    sudo picotool reboot
    cd ..
    
    sleep 30
done
