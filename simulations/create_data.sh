#!/bin/bash -v

mkdir data
mkdir data/elegant_files
mkdir data/elegant_files/train
mkdir data/elegant_files/validate
mkdir data/tmp
mkdir data/training_data
mkdir data/validation_data

python create_parfiles.py

echo "Start simulations for training data"

for FILE in ./data/elegant_files/train/*.ele
do 
    echo "Processing $FILE file..."
    echo "$(basename $FILE)"
    cp "$FILE" ./data/tmp
    elegant ./data/tmp/$(basename $FILE)
    mkdir ./data/training_data/$(basename -s .ele $FILE)
    mv ./data/tmp/*.out ./data/training_data/$(basename -s .ele $FILE)/
    rm ./data/tmp/*.*
done

echo "Start simulations for validation data"

for FILE in ./data/elegant_files/validate/*.ele
do 
    echo "Processing $FILE file..."
    echo "$(basename $FILE)"
    cp "$FILE" ./data/tmp
    elegant ./data/tmp/$(basename $FILE)
    mkdir ./data/validation_data/$(basename -s .ele $FILE)
    mv ./data/tmp/*.out ./data/validation_data/$(basename -s .ele $FILE)/
    rm ./data/tmp/*.*
done

mkdir data/data_npy

python save_files.py