#!/bin/bash

for entry in */
do
    cd "$entry"
    cd "cold_dry"
    ./*.sh
    cd ../..
done

    