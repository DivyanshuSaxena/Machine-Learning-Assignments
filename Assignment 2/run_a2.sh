#!/bin/bash

if [[ $1 == 1 ]]; then
    python text-classification.py $2 $3 $4 
elif [[ $1 == 2 ]]; then
    python handwritten-digit.py $2 $3 $4 $5
fi