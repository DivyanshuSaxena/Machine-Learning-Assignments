#!/bin/bash

if [[ $1 == 1 ]]; then
    python linear-regression.py $2 $3 $4 $5 
elif [[ $1 == 2 ]]; then
    python locally-weighted.py $2 $3 $4 
elif [[ $1 == 3 ]]; then
    python logistic-regression.py $2 $3 
elif [[ $1 == 4 ]]; then
    python gaussian-discriminant.py $2 $3 $4
fi